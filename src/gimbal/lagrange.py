"""Equazioni di Lagrange del modello a 2 GdL (sezione 6 della specifica).

Le equazioni **non sono codificate a mano**: si scrivono energia cinetica,
energia potenziale e funzione di dissipazione, e SymPy deriva

.. math:: \\frac{d}{dt}\\frac{\\partial\\mathcal L}{\\partial\\dot q}
          - \\frac{\\partial\\mathcal L}{\\partial q}
          + \\frac{\\partial\\mathcal D}{\\partial\\dot q} = Q,
          \\qquad \\mathcal L = T - V.

Il risultato viene poi riportato nella forma matriciale

.. math:: M(q)\\ddot q + C(q,\\dot q)\\dot q + B\\dot q
          + \\frac{\\partial V}{\\partial q} = \\tau,

con ``C`` costruita dai simboli di Christoffel, e si verifica che
``dM/dt - 2C`` sia antisimmetrica (proprieta' di passivita').

Il tensore di B e' completo: tutti e sei gli elementi ``I_xx ... I_yz``
entrano nelle equazioni. Le forme chiuse della specifica si ottengono come
caso particolare ponendo a zero i prodotti d'inerzia.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Callable, Final

import numpy as np
import sympy as sp

from . import kinematics as kin
from .inertia import ModelloInerziale

# ---------------------------------------------------------------------------
# Simboli dei parametri
# ---------------------------------------------------------------------------

J_A = sp.Symbol("J_A", positive=True)
I_xx, I_yy, I_zz = sp.symbols("I_xx I_yy I_zz", positive=True)
I_xy, I_xz, I_yz = sp.symbols("I_xy I_xz I_yz", real=True)
m_B = sp.Symbol("m_B", positive=True)
g = sp.Symbol("g", positive=True)
d_x, d_y, d_z = sp.symbols("d_x d_y d_z", real=True)
b_psi, b_theta = sp.symbols("b_psi b_theta", nonnegative=True)
tau_psi, tau_theta = sp.symbols("tau_psi tau_theta", real=True)

SIMBOLI_PARAMETRI: Final[tuple[sp.Symbol, ...]] = (
    J_A, I_xx, I_yy, I_zz, I_xy, I_xz, I_yz, m_B, g, d_x, d_y, d_z, b_psi, b_theta,
)

PRODOTTI_NULLI: Final[dict[sp.Symbol, int]] = {I_xy: 0, I_xz: 0, I_yz: 0}
"""Sostituzione che annulla i prodotti d'inerzia (target di regressione)."""


def tensore_B() -> sp.Matrix:
    """Tensore ``I_B,O|_B`` simbolico, completo e simmetrico."""
    return sp.Matrix(
        [
            [I_xx, I_xy, I_xz],
            [I_xy, I_yy, I_yz],
            [I_xz, I_yz, I_zz],
        ]
    )


# ---------------------------------------------------------------------------
# Modello simbolico
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ModelloSimbolico:
    """Equazioni di moto simboliche e loro forma matriciale.

    Attributes
    ----------
    T, V, D, L
        Energia cinetica, potenziale, funzione di dissipazione, lagrangiana.
    eom
        Residuo delle equazioni di Lagrange, ``eom = 0`` (vettore 2x1),
        nella forma ``M ddq + C dq + B dq + dV/dq - tau``.
    M
        Matrice di massa ``M(theta)``, 2x2 simmetrica.
    C
        Matrice di Coriolis dai simboli di Christoffel, 2x2.
    gradV
        Gradiente dell'energia potenziale, 2x1.
    Bmat
        Matrice di smorzamento viscoso ``diag(b_psi, b_theta)``.
    residuo
        ``M ddq + C dq + B dq + dV/dq`` (senza le coppie), 2x1: e' la
        funzione di cui si prendono i jacobiani nella linearizzazione.
    J_psi, M_12
        Elementi ``M[0,0]`` e ``M[0,1]``, esposti per leggibilita'.
    """

    T: sp.Expr
    V: sp.Expr
    D: sp.Expr
    L: sp.Expr
    eom: sp.Matrix
    M: sp.Matrix
    C: sp.Matrix
    gradV: sp.Matrix
    Bmat: sp.Matrix
    residuo: sp.Matrix
    omega_B_B: sp.Matrix
    J_psi: sp.Expr
    M_12: sp.Expr


def _christoffel(M: sp.Matrix, q: sp.Matrix, dq: sp.Matrix) -> sp.Matrix:
    """Matrice di Coriolis dai simboli di Christoffel di prima specie.

    .. math:: C_{kj} = \\sum_i \\frac12\\left(
              \\frac{\\partial M_{kj}}{\\partial q_i}
              + \\frac{\\partial M_{ki}}{\\partial q_j}
              - \\frac{\\partial M_{ij}}{\\partial q_k}\\right)\\dot q_i

    Con questa scelta ``dM/dt - 2C`` e' antisimmetrica.
    """
    n = len(q)
    C = sp.zeros(n, n)
    for k in range(n):
        for j in range(n):
            somma = sp.S.Zero
            for i in range(n):
                c_kji = sp.Rational(1, 2) * (
                    sp.diff(M[k, j], q[i]) + sp.diff(M[k, i], q[j]) - sp.diff(M[i, j], q[k])
                )
                somma += c_kji * dq[i]
            C[k, j] = sp.simplify(somma)
    return C


@lru_cache(maxsize=4)
def costruisci_modello_simbolico() -> ModelloSimbolico:
    """Deriva le equazioni di moto con SymPy. Il risultato e' memoizzato."""
    theta, dpsi, dtheta = kin.theta, kin.dpsi, kin.dtheta
    q, dq, ddq = kin.q, kin.dq, kin.ddq

    # --- cinematica ---------------------------------------------------------
    cin = kin.costruisci_cinematica()
    omega = cin.omega_poisson  # (-dpsi sin th, dtheta, dpsi cos th)

    # --- energia cinetica ---------------------------------------------------
    # A ruota solo in psi: contribuisce con 1/2 J_A dpsi^2.
    # B ruota attorno a O: 1/2 omega^T I_B,O|_B omega, tensore completo.
    T = sp.Rational(1, 2) * J_A * dpsi**2 + sp.Rational(1, 2) * (
        omega.T * tensore_B() * omega
    )[0, 0]
    T = sp.expand(sp.trigsimp(sp.expand(T)))

    # --- energia potenziale -------------------------------------------------
    # V = m_B g (r_OG . Z). In terna B la verticale e' Z|_B = (-sin th, 0, cos th),
    # quindi r_OG . Z = -d_x sin(theta) + d_z cos(theta): d_y non contribuisce
    # perche' y_B e' sempre orizzontale. A non varia la propria quota.
    Z_in_B = kin.R_y(theta).T * sp.Matrix([0, 0, 1])
    r_OG = sp.Matrix([d_x, d_y, d_z])
    V = sp.simplify(m_B * g * (r_OG.T * Z_in_B)[0, 0])

    # --- dissipazione e forze generalizzate ---------------------------------
    D = sp.Rational(1, 2) * b_psi * dpsi**2 + sp.Rational(1, 2) * b_theta * dtheta**2
    Q = sp.Matrix([tau_psi, tau_theta])

    L = T - V

    # --- equazioni di Lagrange ---------------------------------------------
    eom = sp.zeros(2, 1)
    for i in range(2):
        dL_ddq = sp.diff(L, dq[i])
        termine_inerziale = kin.derivata_totale(dL_ddq)
        eom[i] = sp.simplify(
            termine_inerziale - sp.diff(L, q[i]) + sp.diff(D, dq[i]) - Q[i]
        )

    # --- forma matriciale ---------------------------------------------------
    # M si estrae come hessiana di T rispetto a dq: e' esatta perche' T e'
    # quadratica omogenea in dq (nessun termine di trascinamento).
    M = sp.simplify(sp.hessian(T, list(dq)))
    C = _christoffel(M, q, dq)
    gradV = sp.Matrix([sp.simplify(sp.diff(V, q[i])) for i in range(2)])
    Bmat = sp.diag(b_psi, b_theta)

    residuo = sp.simplify(M * ddq + C * dq + Bmat * dq + gradV)

    # Coerenza fra le due derivazioni: eom deve essere residuo - Q.
    scarto = sp.simplify(sp.expand(eom - (residuo - Q)))
    if scarto != sp.zeros(2, 1):
        raise AssertionError(
            "forma matriciale e equazioni di Lagrange non coincidono: " f"{scarto.T}"
        )

    return ModelloSimbolico(
        T=T,
        V=V,
        D=D,
        L=L,
        eom=eom,
        M=M,
        C=C,
        gradV=gradV,
        Bmat=Bmat,
        residuo=residuo,
        omega_B_B=omega,
        J_psi=sp.simplify(M[0, 0]),
        M_12=sp.simplify(M[0, 1]),
    )


# ---------------------------------------------------------------------------
# Verifiche strutturali (sezione 9, test 7 e 8)
# ---------------------------------------------------------------------------


def verifica_struttura_M(mod: ModelloSimbolico | None = None) -> bool:
    """Confronta ``M(theta)`` con la struttura attesa della sezione 6."""
    mod = mod or costruisci_modello_simbolico()
    theta = kin.theta

    J_psi_atteso = (
        J_A + I_xx * sp.sin(theta) ** 2 + I_zz * sp.cos(theta) ** 2 - I_xz * sp.sin(2 * theta)
    )
    M_12_atteso = -I_xy * sp.sin(theta) + I_yz * sp.cos(theta)

    if sp.simplify(mod.M[0, 0] - J_psi_atteso) != 0:
        raise AssertionError(f"M[0,0] != J_psi atteso: {sp.simplify(mod.M[0, 0] - J_psi_atteso)}")
    if sp.simplify(mod.M[0, 1] - M_12_atteso) != 0:
        raise AssertionError(f"M[0,1] != M_12 atteso: {sp.simplify(mod.M[0, 1] - M_12_atteso)}")
    if sp.simplify(mod.M[1, 1] - I_yy) != 0:
        raise AssertionError("M[1,1] deve valere I_yy")
    if sp.simplify(mod.M[0, 1] - mod.M[1, 0]) != 0:
        raise AssertionError("M non e' simmetrica")
    return True


def verifica_forme_chiuse(mod: ModelloSimbolico | None = None) -> bool:
    """Con prodotti nulli le equazioni coincidono con la sezione 6 (test 7).

    Forme attese:

    .. math:: J_\\psi\\ddot\\psi + J_\\psi'\\dot\\theta\\dot\\psi
              + b_\\psi\\dot\\psi = \\tau_\\psi
    .. math:: I_{yy}\\ddot\\theta - \\tfrac12 J_\\psi'\\dot\\psi^2
              + \\partial V/\\partial\\theta + b_\\theta\\dot\\theta = \\tau_\\theta
    """
    mod = mod or costruisci_modello_simbolico()
    theta, dpsi, dtheta = kin.theta, kin.dpsi, kin.dtheta
    ddpsi, ddtheta = kin.ddpsi, kin.ddtheta

    eom0 = mod.eom.subs(PRODOTTI_NULLI)

    J_psi = J_A + I_xx * sp.sin(theta) ** 2 + I_zz * sp.cos(theta) ** 2
    J_psi_primo = (I_xx - I_zz) * sp.sin(2 * theta)
    dV_dtheta = -m_B * g * (d_x * sp.cos(theta) + d_z * sp.sin(theta))

    atteso_psi = J_psi * ddpsi + J_psi_primo * dtheta * dpsi + b_psi * dpsi - tau_psi
    atteso_theta = (
        I_yy * ddtheta
        - sp.Rational(1, 2) * J_psi_primo * dpsi**2
        + dV_dtheta
        + b_theta * dtheta
        - tau_theta
    )

    # Controllo che J_psi' sia davvero la derivata di J_psi.
    if sp.simplify(sp.diff(J_psi, theta) - J_psi_primo) != 0:
        raise AssertionError("J_psi' non e' la derivata di J_psi")

    for nome, ottenuto, atteso in (
        ("psi", eom0[0], atteso_psi),
        ("theta", eom0[1], atteso_theta),
    ):
        differenza = sp.simplify(sp.expand(sp.trigsimp(ottenuto - atteso)))
        if differenza != 0:
            raise AssertionError(
                f"equazione in {nome}: scarto rispetto alla forma chiusa = {differenza}"
            )
    return True


def verifica_antisimmetria(mod: ModelloSimbolico | None = None) -> sp.Matrix:
    """Verifica che ``dM/dt - 2C`` sia antisimmetrica (sezione 9, test 8).

    Returns
    -------
    sympy.Matrix
        La matrice ``dM/dt - 2C``, utile per ispezione.
    """
    mod = mod or costruisci_modello_simbolico()
    Mdot = sp.Matrix(2, 2, lambda i, j: kin.derivata_totale(mod.M[i, j]))
    S = sp.simplify(Mdot - 2 * mod.C)
    if sp.simplify(S + S.T) != sp.zeros(2, 2):
        raise AssertionError(f"dM/dt - 2C non e' antisimmetrica:\n{S}")
    return S


# ---------------------------------------------------------------------------
# Passaggio ai valori numerici
# ---------------------------------------------------------------------------


def sostituzioni_numeriche(modello: ModelloInerziale) -> dict[sp.Symbol, float]:
    """Mappa dai simboli dei parametri ai valori numerici del modello."""
    return {
        J_A: modello.J_A,
        I_xx: modello.I_xx,
        I_yy: modello.I_yy,
        I_zz: modello.I_zz,
        I_xy: modello.I_xy,
        I_xz: modello.I_xz,
        I_yz: modello.I_yz,
        m_B: modello.m_B,
        g: modello.g,
        d_x: modello.d_x,
        d_y: modello.d_y,
        d_z: modello.d_z,
        b_psi: modello.par.b_psi,
        b_theta: modello.par.b_theta,
    }


@dataclass(frozen=True)
class ModelloNumerico:
    """Funzioni numeriche del modello non lineare, ottenute con ``lambdify``.

    Tutte le funzioni accettano e restituiscono unita' SI, con angoli in rad.
    """

    M: Callable[[float], np.ndarray]
    """``M(theta)`` -> matrice di massa 2x2."""

    Cdq: Callable[[float, float, float], np.ndarray]
    """``Cdq(theta, dpsi, dtheta)`` -> vettore di Coriolis ``C dq``, 2x1."""

    gradV: Callable[[float], np.ndarray]
    """``gradV(theta)`` -> gradiente dell'energia potenziale, 2x1."""

    Mdot: Callable[[float, float], np.ndarray]
    """``Mdot(theta, dtheta)`` -> ``dM/dt``, 2x2."""

    T: Callable[[float, float, float], float]
    """``T(theta, dpsi, dtheta)`` -> energia cinetica."""

    V: Callable[[float], float]
    """``V(theta)`` -> energia potenziale."""

    p_psi: Callable[[float, float, float], float]
    """``p_psi(theta, dpsi, dtheta)`` -> momento coniugato a psi."""

    Bmat: np.ndarray
    """Matrice di smorzamento viscoso, 2x2."""

    modello_inerziale: ModelloInerziale
    simbolico: ModelloSimbolico

    # -- dinamica ------------------------------------------------------------

    def accelerazioni(
        self, q: np.ndarray, dq: np.ndarray, tau: np.ndarray
    ) -> np.ndarray:
        """Risolve ``M ddq = tau - C dq - B dq - dV/dq``.

        Parameters
        ----------
        q
            ``[psi, theta]`` in rad.
        dq
            ``[dpsi, dtheta]`` in rad/s.
        tau
            ``[tau_psi, tau_theta]`` in N m.
        """
        theta = float(q[1])
        forzante = (
            np.asarray(tau, dtype=float)
            - self.Cdq(theta, float(dq[0]), float(dq[1]))
            - self.Bmat @ np.asarray(dq, dtype=float)
            - self.gradV(theta)
        )
        return np.linalg.solve(self.M(theta), forzante)

    def energia(self, q: np.ndarray, dq: np.ndarray) -> float:
        """Energia meccanica ``E = T + V``."""
        return float(self.T(float(q[1]), float(dq[0]), float(dq[1])) + self.V(float(q[1])))

    def momento_psi(self, q: np.ndarray, dq: np.ndarray) -> float:
        """Momento coniugato ``p_psi = dL/d(dpsi)``, conservato se tau_psi = 0 e b = 0."""
        return float(self.p_psi(float(q[1]), float(dq[0]), float(dq[1])))

    def coppie_nominali(self, theta0: float, Omega0: float) -> np.ndarray:
        """Coppie di equilibrio nel punto di lavoro (sezione 7).

        .. math:: \\tau_0 = C\\dot q_0 + B\\dot q_0 + \\partial V/\\partial q|_0

        con ``dq_0 = [Omega0, 0]`` e ``ddq_0 = 0``. La seconda componente e'
        la **coppia di mantenimento** dell'asse di elevazione.
        """
        dq0 = np.array([Omega0, 0.0])
        return self.Cdq(theta0, Omega0, 0.0) + self.Bmat @ dq0 + self.gradV(theta0)


def _lambdify(espressione: sp.Expr | sp.Matrix, argomenti: list[sp.Symbol]) -> Callable:
    """``lambdify`` con nomi di simboli non validi come identificatori Python.

    I simboli ``\\dot{\\psi}`` e simili sono scelti per la resa in LaTeX; per
    ``lambdify`` vengono rinominati in identificatori semplici.
    """
    rinomina = {
        kin.dpsi: sp.Symbol("dpsi"),
        kin.dtheta: sp.Symbol("dtheta"),
        kin.ddpsi: sp.Symbol("ddpsi"),
        kin.ddtheta: sp.Symbol("ddtheta"),
    }
    espressione = espressione.subs(rinomina)
    argomenti = [rinomina.get(a, a) for a in argomenti]
    return sp.lambdify(argomenti, espressione, modules="numpy")


def costruisci_modello_numerico(modello: ModelloInerziale) -> ModelloNumerico:
    """Sostituisce i valori numerici e produce le funzioni ``lambdify``."""
    sim = costruisci_modello_simbolico()
    sost = sostituzioni_numeriche(modello)
    theta, dpsi, dtheta = kin.theta, kin.dpsi, kin.dtheta

    M_n = sim.M.subs(sost)
    Cdq_n = (sim.C * kin.dq).subs(sost)
    gradV_n = sim.gradV.subs(sost)
    Mdot_n = sp.Matrix(2, 2, lambda i, j: kin.derivata_totale(sim.M[i, j])).subs(sost)
    T_n = sim.T.subs(sost)
    V_n = sim.V.subs(sost)
    p_psi_n = sp.diff(sim.T - sim.V, dpsi).subs(sost)

    f_M = _lambdify(M_n, [theta])
    f_Cdq = _lambdify(Cdq_n, [theta, dpsi, dtheta])
    f_gradV = _lambdify(gradV_n, [theta])
    f_Mdot = _lambdify(Mdot_n, [theta, dtheta])
    f_T = _lambdify(T_n, [theta, dpsi, dtheta])
    f_V = _lambdify(V_n, [theta])
    f_p = _lambdify(p_psi_n, [theta, dpsi, dtheta])

    return ModelloNumerico(
        M=lambda th: np.asarray(f_M(th), dtype=float).reshape(2, 2),
        Cdq=lambda th, dp, dt: np.asarray(f_Cdq(th, dp, dt), dtype=float).reshape(2),
        gradV=lambda th: np.asarray(f_gradV(th), dtype=float).reshape(2),
        Mdot=lambda th, dt: np.asarray(f_Mdot(th, dt), dtype=float).reshape(2, 2),
        T=lambda th, dp, dt: float(f_T(th, dp, dt)),
        V=lambda th: float(f_V(th)),
        p_psi=lambda th, dp, dt: float(f_p(th, dp, dt)),
        Bmat=np.diag([modello.par.b_psi, modello.par.b_theta]),
        modello_inerziale=modello,
        simbolico=sim,
    )


# ---------------------------------------------------------------------------
# Esportazione LaTeX (sezione 6, output del modulo)
# ---------------------------------------------------------------------------


def esporta_latex(percorso: str | Path, mod: ModelloSimbolico | None = None) -> Path:
    """Scrive le equazioni di moto in LaTeX in ``results/latex/eom.tex``."""
    mod = mod or costruisci_modello_simbolico()
    percorso = Path(percorso)
    percorso.parent.mkdir(parents=True, exist_ok=True)

    theta = kin.theta
    eom0 = mod.eom.subs(PRODOTTI_NULLI)
    S = verifica_antisimmetria(mod)

    def eq(espressione: sp.Expr | sp.Matrix) -> str:
        return sp.latex(espressione)

    righe: list[str] = [
        "% Equazioni di moto del modello rigido a 2 GdL.",
        "% Generato automaticamente da gimbal.lagrange.esporta_latex.",
        "% Notazione: sezione 2 della specifica. Angoli in rad, unita' SI.",
        "",
        r"\section*{Modello rigido a 2 gradi di libert\`a}",
        "",
        r"\paragraph{Coordinate e cinematica.}",
        r"\begin{equation}",
        r"  \mathbf q = \begin{bmatrix}\psi \\ \theta\end{bmatrix}, \qquad",
        r"  \boldsymbol\omega_B\big|_B = " + eq(mod.omega_B_B),
        r"\end{equation}",
        "",
        r"\paragraph{Energie e dissipazione.}",
        r"\begin{align}",
        r"  T &= " + eq(mod.T) + r" \\",
        r"  V &= " + eq(mod.V) + r" \\",
        r"  \mathcal D &= " + eq(mod.D),
        r"\end{align}",
        "",
        r"\paragraph{Forma matriciale.}",
        r"\begin{equation}",
        r"  \mathbf M(\mathbf q)\ddot{\mathbf q} + \mathbf C(\mathbf q,\dot{\mathbf q})"
        r"\dot{\mathbf q} + \mathbf B\dot{\mathbf q}"
        r" + \frac{\partial V}{\partial \mathbf q} = \boldsymbol\tau",
        r"\end{equation}",
        r"\begin{equation}",
        r"  \mathbf M(\theta) = " + eq(mod.M),
        r"\end{equation}",
        r"\begin{equation}",
        r"  \mathbf C(\mathbf q,\dot{\mathbf q}) = " + eq(mod.C),
        r"\end{equation}",
        r"\begin{equation}",
        r"  \frac{\partial V}{\partial \mathbf q} = " + eq(mod.gradV),
        r"\qquad \mathbf B = " + eq(mod.Bmat),
        r"\end{equation}",
        "",
        r"\paragraph{Equazioni di Lagrange complete.}",
        r"\begin{align}",
        r"  " + eq(mod.eom[0]) + r" &= 0 \\",
        r"  " + eq(mod.eom[1]) + r" &= 0",
        r"\end{align}",
        "",
        r"\paragraph{Caso con prodotti d'inerzia nulli.}",
        r"Con $I_{xy} = I_{xz} = I_{yz} = 0$ le equazioni si riducono a",
        r"\begin{align}",
        r"  " + eq(sp.collect(sp.expand(eom0[0]), [kin.ddpsi, kin.dpsi])) + r" &= 0 \\",
        r"  " + eq(sp.collect(sp.expand(eom0[1]), [kin.ddtheta, kin.dpsi])) + r" &= 0",
        r"\end{align}",
        r"con",
        r"\begin{equation}",
        r"  J_\psi(\theta) = " + eq(sp.simplify(mod.J_psi)) + r", \qquad",
        r"  J_\psi'(\theta)\big|_{I_{xz}=0} = "
        + eq(sp.simplify(sp.diff(mod.J_psi.subs(PRODOTTI_NULLI), theta))),
        r"\end{equation}",
        "",
        r"\paragraph{Passivit\`a.}",
        r"La matrice $\dot{\mathbf M} - 2\mathbf C$ risulta antisimmetrica:",
        r"\begin{equation}",
        r"  \dot{\mathbf M} - 2\mathbf C = " + eq(S),
        r"\end{equation}",
        "",
    ]
    percorso.write_text("\n".join(righe), encoding="utf-8")
    return percorso
