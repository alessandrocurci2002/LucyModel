"""Linearizzazione, spazio di stato e funzioni di trasferimento (sezione 7).

Le matrici linearizzate si ottengono come **jacobiani simbolici del residuo**

.. math:: \\mathbf r = \\mathbf M\\ddot{\\mathbf q} + \\mathbf C\\dot{\\mathbf q}
          + \\mathbf B\\dot{\\mathbf q} + \\partial V/\\partial\\mathbf q,

valutati nel punto di lavoro ``theta = theta_0``, ``dpsi = Omega_0``,
``dtheta = 0``, ``ddq = 0``:

.. math:: \\mathbf M_0 = \\partial\\mathbf r/\\partial\\ddot{\\mathbf q},\\quad
          \\mathbf C_0 = \\partial\\mathbf r/\\partial\\dot{\\mathbf q},\\quad
          \\mathbf K_0 = \\partial\\mathbf r/\\partial\\mathbf q.

``psi`` e' ciclica, quindi la prima colonna di ``K_0`` e' identicamente nulla
e il sistema ha un polo nell'origine per l'asse di azimut.

La matrice di trasferimento e' ``G(s) = P(s)^{-1}`` con
``P(s) = M_0 s^2 + C_0 s + K_0``; le FRF sono valutate invertendo
``P(j omega)`` direttamente, non tramite i polinomi, per accuratezza numerica.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Final, Sequence

import numpy as np
import sympy as sp

from . import kinematics as kin
from . import lagrange as lg
from .inertia import ModelloInerziale

# ---------------------------------------------------------------------------
# Simboli del punto di lavoro
# ---------------------------------------------------------------------------

theta0 = sp.Symbol("theta_0", real=True)
Omega0 = sp.Symbol("Omega_0", real=True)
s = sp.Symbol("s")

# Abbreviazioni per la forma compatta delle FdT: sono le tre grandezze che
# governano la dinamica linearizzata quando i prodotti d'inerzia sono nulli.
J_psi_s = sp.Symbol("J_psi", positive=True)
gamma_s = sp.Symbol("gamma", real=True)
k_theta_s = sp.Symbol("k_theta", real=True)

ETICHETTE_FDT: Final[dict[tuple[int, int], str]] = {
    (0, 0): "G11 = Psi/T_psi (collocata, azimut)",
    (0, 1): "G12 = Psi/T_theta (incrociata)",
    (1, 0): "G21 = Theta/T_psi (incrociata)",
    (1, 1): "G22 = Theta/T_theta (collocata, elevazione)",
}


# ---------------------------------------------------------------------------
# Linearizzazione simbolica
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ModelloLineareSimbolico:
    """Matrici linearizzate e FdT in forma simbolica.

    Le matrici dipendono dai simboli del punto di lavoro (``theta_0``,
    ``Omega_0``) e dai simboli dei parametri di :mod:`gimbal.lagrange`.

    Attributes
    ----------
    M0, C0, K0
        Matrici linearizzate 2x2.
    P
        ``P(s) = M_0 s^2 + C_0 s + K_0``.
    Delta
        ``det P(s)``, denominatore comune delle quattro FdT.
    G
        ``adj(P)/Delta``, matrice di trasferimento 2x2.
    gamma
        Coefficiente giroscopico ``C_0[0,1]``.
    k_theta
        Rigidezza equivalente dell'asse di elevazione ``K_0[1,1]``.
    """

    M0: sp.Matrix
    C0: sp.Matrix
    K0: sp.Matrix
    P: sp.Matrix
    Delta: sp.Expr
    G: sp.Matrix
    gamma: sp.Expr
    k_theta: sp.Expr
    tau0: sp.Matrix


@lru_cache(maxsize=4)
def costruisci_modello_lineare_simbolico() -> ModelloLineareSimbolico:
    """Deriva ``M_0``, ``C_0``, ``K_0``, ``P(s)`` e ``G(s)``. Memoizzato."""
    sim = lg.costruisci_modello_simbolico()
    residuo = sim.residuo

    q, dq, ddq = kin.q, kin.dq, kin.ddq

    # Jacobiani simbolici, prima della valutazione nel punto di lavoro.
    dr_dddq = residuo.jacobian(ddq)
    dr_ddq = residuo.jacobian(dq)
    dr_dq = residuo.jacobian(q)

    # Punto di lavoro: theta = theta_0, dpsi = Omega_0, dtheta = 0, ddq = 0.
    # psi e' ciclica: psi_0 non compare da nessuna parte.
    punto = {
        kin.theta: theta0,
        kin.dpsi: Omega0,
        kin.dtheta: 0,
        kin.ddpsi: 0,
        kin.ddtheta: 0,
    }

    M0 = sp.simplify(dr_dddq.subs(punto))
    C0 = sp.simplify(dr_ddq.subs(punto))
    K0 = sp.simplify(dr_dq.subs(punto))

    if sp.simplify(K0[:, 0]) != sp.zeros(2, 1):
        raise AssertionError(
            "la prima colonna di K_0 deve essere nulla (psi e' ciclica): " f"{K0[:, 0].T}"
        )

    # Coppie nominali nel punto di lavoro.
    tau0 = sp.simplify(residuo.subs(punto))

    P = sp.expand(M0 * s**2 + C0 * s + K0)
    Delta = sp.expand(P.det())
    adjP = sp.Matrix([[P[1, 1], -P[0, 1]], [-P[1, 0], P[0, 0]]])
    G = adjP / Delta

    return ModelloLineareSimbolico(
        M0=M0,
        C0=C0,
        K0=K0,
        P=P,
        Delta=Delta,
        G=G,
        gamma=sp.simplify(C0[0, 1]),
        k_theta=sp.simplify(K0[1, 1]),
        tau0=tau0,
    )


def verifica_target_regressione() -> bool:
    """Verifica i target di regressione della sezione 7 (prodotti nulli).

    Attesi:

    .. math:: \\mathbf C_0 = \\begin{bmatrix} b_\\psi & \\gamma \\\\
              -\\gamma & b_\\theta\\end{bmatrix},\\qquad
              \\mathbf K_0 = \\begin{bmatrix}0 & 0\\\\ 0 & k_\\theta\\end{bmatrix}
    .. math:: \\gamma = (I_{xx} - I_{zz})\\,\\Omega_0\\sin 2\\theta_0
    .. math:: k_\\theta = -(I_{xx}-I_{zz})\\Omega_0^2\\cos 2\\theta_0
              + m_B g (d_x\\sin\\theta_0 - d_z\\cos\\theta_0)
    """
    lin = costruisci_modello_lineare_simbolico()
    M0 = lin.M0.subs(lg.PRODOTTI_NULLI)
    C0 = lin.C0.subs(lg.PRODOTTI_NULLI)
    K0 = lin.K0.subs(lg.PRODOTTI_NULLI)

    gamma_atteso = (lg.I_xx - lg.I_zz) * Omega0 * sp.sin(2 * theta0)
    k_c = -(lg.I_xx - lg.I_zz) * Omega0**2 * sp.cos(2 * theta0)
    k_g = lg.m_B * lg.g * (lg.d_x * sp.sin(theta0) - lg.d_z * sp.cos(theta0))

    J_psi0 = lg.J_A + lg.I_xx * sp.sin(theta0) ** 2 + lg.I_zz * sp.cos(theta0) ** 2
    M0_atteso = sp.Matrix([[J_psi0, 0], [0, lg.I_yy]])
    C0_atteso = sp.Matrix([[lg.b_psi, gamma_atteso], [-gamma_atteso, lg.b_theta]])
    K0_atteso = sp.Matrix([[0, 0], [0, k_c + k_g]])

    for nome, ottenuto, atteso in (
        ("M_0", M0, M0_atteso),
        ("C_0", C0, C0_atteso),
        ("K_0", K0, K0_atteso),
    ):
        differenza = sp.simplify(ottenuto - atteso)
        if differenza != sp.zeros(2, 2):
            raise AssertionError(f"{nome} non coincide con il target:\n{differenza}")
    return True


def forma_compatta() -> dict[str, sp.Expr | sp.Matrix]:
    """FdT nella forma compatta in ``J_psi``, ``gamma``, ``k_theta``.

    Con prodotti d'inerzia nulli la dinamica linearizzata dipende solo da tre
    grandezze oltre agli smorzamenti, e le funzioni di trasferimento si
    scrivono in poche righe invece che in poche pagine. E' la forma citabile
    in tesi.

    Le espressioni non sono scritte a mano: si parte dalle matrici ricavate
    per jacobiani, se ne riconoscono le tre grandezze e si ricalcolano
    ``Delta``, ``adj P`` e ``G``. La funzione verifica anche che, risostituendo
    le definizioni, si ritrovi esattamente la forma estesa.

    Returns
    -------
    dict
        ``P``, ``Delta``, ``D`` (il fattore cubico), ``G`` e ``definizioni``.
    """
    lin = costruisci_modello_lineare_simbolico()
    M0 = lin.M0.subs(lg.PRODOTTI_NULLI)
    C0 = lin.C0.subs(lg.PRODOTTI_NULLI)
    K0 = lin.K0.subs(lg.PRODOTTI_NULLI)

    definizioni: dict[sp.Symbol, sp.Expr] = {
        J_psi_s: sp.simplify(M0[0, 0]),
        gamma_s: sp.simplify(C0[0, 1]),
        k_theta_s: sp.simplify(K0[1, 1]),
    }

    # Riscrittura delle matrici con le abbreviazioni.
    M_c = sp.Matrix([[J_psi_s, 0], [0, lg.I_yy]])
    C_c = sp.Matrix([[lg.b_psi, gamma_s], [-gamma_s, lg.b_theta]])
    K_c = sp.Matrix([[0, 0], [0, k_theta_s]])

    # Controllo: sostituendo le definizioni si ritorna alle matrici estese.
    for nome, compatta, estesa in (("M_0", M_c, M0), ("C_0", C_c, C0), ("K_0", K_c, K0)):
        if sp.simplify(compatta.subs(definizioni) - estesa) != sp.zeros(2, 2):
            raise AssertionError(f"{nome}: la forma compatta non riproduce quella estesa")

    P = sp.expand(M_c * s**2 + C_c * s + K_c)
    Delta = sp.factor(sp.expand(P.det()), s)

    # Delta = s * D(s): il fattore s e' dovuto alla coordinata ciclica psi.
    D = sp.simplify(sp.cancel(Delta / s))
    if sp.simplify(sp.expand(Delta - s * D)) != 0:
        raise AssertionError("Delta non si fattorizza come s * D(s)")

    G = sp.Matrix(
        2, 2,
        lambda i, j: sp.cancel(
            sp.together(sp.Matrix([[P[1, 1], -P[0, 1]], [-P[1, 0], P[0, 0]]])[i, j] / Delta)
        ),
    )
    return {"P": P, "Delta": Delta, "D": sp.collect(sp.expand(D), s), "G": G,
            "definizioni": definizioni}


def esporta_latex(percorso: str | Path) -> Path:
    """Scrive ``tf.tex``: ``P(s)``, ``Delta(s)`` e le quattro ``G_ij(s)``.

    Si esportano tre livelli di dettaglio:

    1. la forma **generale**, in termini degli elementi di ``M_0``, ``C_0`` e
       ``K_0``, valida anche con prodotti d'inerzia non nulli;
    2. la forma **compatta** con prodotti nulli, scritta in ``J_psi``,
       ``gamma`` e ``k_theta``: e' quella citabile in tesi;
    3. il **caso di verifica** ``Omega_0 = 0``, ``d = 0``, che riproduce i
       target di regressione della sezione 7.

    La forma estesa in tutti i parametri non viene esportata: occuperebbe
    pagine e non e' leggibile. Resta accessibile da codice con
    ``costruisci_modello_lineare_simbolico().G``.
    """
    lin = costruisci_modello_lineare_simbolico()
    compatta = forma_compatta()
    percorso = Path(percorso)
    percorso.parent.mkdir(parents=True, exist_ok=True)

    G0 = compatta["G"]
    senza_rotazione = {gamma_s: 0, k_theta_s: 0}
    G_target = sp.Matrix(
        2, 2, lambda i, j: sp.cancel(sp.factor(G0[i, j].subs(senza_rotazione)))
    )
    definizioni = compatta["definizioni"]

    L = sp.latex
    righe: list[str] = [
        "% Funzioni di trasferimento coppia -> angolo del modello rigido a 2 GdL.",
        "% Generato automaticamente da gimbal.linearize.esporta_latex.",
        "",
        r"\section*{Funzioni di trasferimento $G_{ij}(s;\theta_0,\Omega_0)$}",
        "",
        r"\paragraph{Modello linearizzato.}",
        r"\begin{equation}",
        r"  \mathbf M_0\,\delta\ddot{\mathbf q} + \mathbf C_0\,\delta\dot{\mathbf q}"
        r" + \mathbf K_0\,\delta\mathbf q = \delta\boldsymbol\tau",
        r"\end{equation}",
        r"\begin{equation}",
        r"  \mathbf M_0 = " + L(lin.M0),
        r"\end{equation}",
        r"\begin{equation}",
        r"  \mathbf C_0 = " + L(lin.C0),
        r"\end{equation}",
        r"\begin{equation}",
        r"  \mathbf K_0 = " + L(lin.K0),
        r"\end{equation}",
        r"La prima colonna di $\mathbf K_0$ \`e nulla perch\'e $\psi$ \`e ciclica.",
        "",
        r"\paragraph{Coppie nominali nel punto di lavoro.}",
        r"La seconda componente \`e la coppia di mantenimento dell'asse di elevazione.",
        r"\begin{equation}",
        r"  \boldsymbol\tau_0 = " + L(lin.tau0),
        r"\end{equation}",
        "",
        r"\paragraph{Matrice di trasferimento.}",
        r"\begin{equation}",
        r"  \mathbf P(s) = \mathbf M_0 s^2 + \mathbf C_0 s + \mathbf K_0, \qquad",
        r"  \mathbf G(s) = \mathbf P(s)^{-1} = \frac{\operatorname{adj}\mathbf P(s)}{\Delta(s)}",
        r"\end{equation}",
        r"\begin{align}",
        r"  G_{11} &= \frac{P_{22}}{\Delta}, &"
        r"  G_{12} &= -\frac{P_{12}}{\Delta}, \\",
        r"  G_{21} &= -\frac{P_{21}}{\Delta}, &"
        r"  G_{22} &= \frac{P_{11}}{\Delta}",
        r"\end{align}",
        "",
        r"\paragraph{Forma compatta, con prodotti d'inerzia nulli.}",
        r"Con $I_{xy} = I_{xz} = I_{yz} = 0$ la dinamica linearizzata dipende, oltre",
        r"agli smorzamenti, da tre sole grandezze:",
        r"\begin{align}",
        r"  J_\psi &= " + L(definizioni[J_psi_s]) + r" \\",
        r"  \gamma &= " + L(definizioni[gamma_s])
        + r" &&\text{(accoppiamento giroscopico)} \\",
        r"  k_\theta &= " + L(definizioni[k_theta_s])
        + r" &&\text{(rigidezza centrifuga + gravitazionale)}",
        r"\end{align}",
        r"con cui",
        r"\begin{equation}",
        r"  \mathbf P(s) = " + L(compatta["P"]) + r", \qquad",
        r"  \Delta(s) = s\,D(s)",
        r"\end{equation}",
        r"\begin{equation}",
        r"  D(s) = " + L(compatta["D"]),
        r"\end{equation}",
        r"Il fattore $s$ in $\Delta$ \`e dovuto alla coordinata ciclica $\psi$; i",
        r"numeratori di $G_{12}$, $G_{21}$ e $G_{22}$ lo condividono e lo cancellano,",
        r"mentre $G_{11}$ lo conserva come polo.",
        r"\begin{align}",
        r"  G_{11}(s) &= " + L(G0[0, 0]) + r" \\",
        r"  G_{12}(s) &= " + L(G0[0, 1]) + r" \\",
        r"  G_{21}(s) &= " + L(G0[1, 0]) + r" \\",
        r"  G_{22}(s) &= " + L(G0[1, 1]),
        r"\end{align}",
        "",
        r"\paragraph{Verifica: $\Omega_0 = 0$ e $\mathbf d = 0$, cio\`e $\gamma = k_\theta = 0$.}",
        r"\begin{align}",
        r"  G_{11}(s) &= " + L(G_target[0, 0]) + r", &"
        r"  G_{12}(s) &= " + L(G_target[0, 1]) + r", \\",
        r"  G_{21}(s) &= " + L(G_target[1, 0]) + r", &"
        r"  G_{22}(s) &= " + L(G_target[1, 1]),
        r"\end{align}",
        "",
    ]
    percorso.write_text("\n".join(righe), encoding="utf-8")
    return percorso


# ---------------------------------------------------------------------------
# Modello lineare numerico
# ---------------------------------------------------------------------------


def polinomio(M: float, C: float, K: float) -> np.ndarray:
    """Coefficienti di ``M s^2 + C s + K``, in ordine decrescente."""
    return np.array([M, C, K], dtype=float)


def polinomio_delta(M0: np.ndarray, C0: np.ndarray, K0: np.ndarray) -> np.ndarray:
    """Coefficienti di ``Delta(s) = det(M_0 s^2 + C_0 s + K_0)``."""
    p11 = polinomio(M0[0, 0], C0[0, 0], K0[0, 0])
    p12 = polinomio(M0[0, 1], C0[0, 1], K0[0, 1])
    p21 = polinomio(M0[1, 0], C0[1, 0], K0[1, 0])
    p22 = polinomio(M0[1, 1], C0[1, 1], K0[1, 1])
    return np.polysub(np.polymul(p11, p22), np.polymul(p12, p21))


def _pulisci(p: np.ndarray, tol: float = 0.0) -> np.ndarray:
    """Rimuove gli zeri iniziali di un polinomio."""
    p = np.atleast_1d(np.asarray(p, dtype=float))
    scala = float(np.max(np.abs(p))) if p.size else 0.0
    soglia = max(tol * scala, 0.0)
    nz = np.nonzero(np.abs(p) > soglia)[0]
    if nz.size == 0:
        return np.array([0.0])
    return p[nz[0] :]


@dataclass(frozen=True)
class Cancellazione:
    """Radice comune cancellata fra numeratore e denominatore di una FdT."""

    fdt: str
    radice: complex
    scarto: float

    def __str__(self) -> str:  # pragma: no cover - solo presentazione
        return (
            f"{self.fdt}: cancellata la radice comune s = {self.radice:.6g} "
            f"(scarto {self.scarto:.2e})"
        )


def cancella_radici_comuni(
    num: np.ndarray, den: np.ndarray, *, tol: float = 1e-8, etichetta: str = ""
) -> tuple[np.ndarray, np.ndarray, list[Cancellazione]]:
    """Cancella le radici comuni a numeratore e denominatore entro ``tol``.

    La cancellazione e' documentata: ogni radice rimossa viene restituita in
    un elenco di :class:`Cancellazione`, riportato poi nel report. Il caso
    tipico e' il fattore ``s`` dovuto alla coordinata ciclica ``psi``, che
    compare in ``Delta`` e nei numeratori di ``G_12``, ``G_21`` e ``G_22``.

    Parameters
    ----------
    tol
        Tolleranza **assoluta** sulla distanza fra radici, riscalata sulla
        dimensione tipica delle radici stesse.
    """
    num = _pulisci(num, tol=1e-14)
    den = _pulisci(den, tol=1e-14)
    cancellazioni: list[Cancellazione] = []

    if num.size <= 1 and abs(float(num[0])) == 0.0:
        return num, den, cancellazioni  # numeratore identicamente nullo

    radici_num = list(np.roots(num)) if num.size > 1 else []
    radici_den = list(np.roots(den)) if den.size > 1 else []
    if not radici_num or not radici_den:
        return num, den, cancellazioni

    scala = max(
        1.0,
        float(np.max(np.abs(radici_num))) if radici_num else 1.0,
        float(np.max(np.abs(radici_den))) if radici_den else 1.0,
    )
    soglia = tol * scala

    num_residue: list[complex] = []
    den_rimaste = list(radici_den)
    for rn in radici_num:
        distanze = [abs(rn - rd) for rd in den_rimaste]
        if distanze and min(distanze) <= soglia:
            k = int(np.argmin(distanze))
            cancellazioni.append(
                Cancellazione(fdt=etichetta, radice=complex(rn), scarto=float(distanze[k]))
            )
            den_rimaste.pop(k)
        else:
            num_residue.append(rn)

    if not cancellazioni:
        return num, den, cancellazioni

    guadagno_num = float(num[0])
    guadagno_den = float(den[0])
    nuovo_num = guadagno_num * np.real(np.poly(num_residue)) if num_residue else np.array(
        [guadagno_num]
    )
    nuovo_den = guadagno_den * np.real(np.poly(den_rimaste)) if den_rimaste else np.array(
        [guadagno_den]
    )
    return np.atleast_1d(nuovo_num), np.atleast_1d(nuovo_den), cancellazioni


@dataclass(frozen=True)
class FunzioneDiTrasferimento:
    """FdT SISO in forma di rapporto di polinomi."""

    nome: str
    num: np.ndarray
    den: np.ndarray
    cancellazioni: tuple[Cancellazione, ...] = ()

    @property
    def zeri(self) -> np.ndarray:
        """Zeri della FdT (radici del numeratore)."""
        n = _pulisci(self.num, tol=1e-14)
        if n.size <= 1:
            return np.array([], dtype=complex)
        return np.roots(n)

    @property
    def poli(self) -> np.ndarray:
        """Poli della FdT (radici del denominatore)."""
        d = _pulisci(self.den, tol=1e-14)
        if d.size <= 1:
            return np.array([], dtype=complex)
        return np.roots(d)

    @property
    def nulla(self) -> bool:
        """``True`` se la FdT e' identicamente nulla."""
        return bool(np.all(np.abs(self.num) == 0.0))

    def valuta(self, s_val: np.ndarray) -> np.ndarray:
        """Valuta la FdT su valori complessi di ``s``."""
        return np.polyval(self.num, s_val) / np.polyval(self.den, s_val)


@dataclass(frozen=True)
class ModelloLineare:
    """Modello linearizzato numerico in un punto di lavoro.

    Attributes
    ----------
    theta0
        Punto di lavoro in elevazione, in rad.
    Omega0
        Velocita' di inseguimento in azimut, in rad/s.
    M0, C0, K0
        Matrici linearizzate.
    A, Bu, Cy, Dd
        Spazio di stato con ``x = [dpsi, dtheta, d(dpsi), d(dtheta)]``,
        ingresso ``u = dtau`` e uscita ``y = dq``.
    tau0
        Coppie nominali; ``tau0[1]`` e' la coppia di mantenimento.
    """

    theta0: float
    Omega0: float
    M0: np.ndarray
    C0: np.ndarray
    K0: np.ndarray
    A: np.ndarray
    Bu: np.ndarray
    Cy: np.ndarray
    Dd: np.ndarray
    tau0: np.ndarray
    modello_inerziale: ModelloInerziale
    fdt: dict[tuple[int, int], FunzioneDiTrasferimento] = field(default_factory=dict)

    # -- poli e zeri ---------------------------------------------------------

    @property
    def poli(self) -> np.ndarray:
        """Autovalori di ``A``."""
        return np.linalg.eigvals(self.A)

    @property
    def delta(self) -> np.ndarray:
        """Coefficienti di ``Delta(s) = det P(s)``."""
        return polinomio_delta(self.M0, self.C0, self.K0)

    @property
    def radici_delta(self) -> np.ndarray:
        """Radici di ``Delta(s)``; devono coincidere con gli autovalori di A."""
        return np.roots(_pulisci(self.delta, tol=1e-14))

    @property
    def gamma(self) -> float:
        """Coefficiente giroscopico ``C_0[0,1]``."""
        return float(self.C0[0, 1])

    @property
    def k_theta(self) -> float:
        """Rigidezza equivalente dell'asse di elevazione ``K_0[1,1]``."""
        return float(self.K0[1, 1])

    def P(self, s_val: complex | np.ndarray) -> np.ndarray:
        """``P(s) = M_0 s^2 + C_0 s + K_0`` per un valore scalare di ``s``."""
        return self.M0 * s_val**2 + self.C0 * s_val + self.K0

    def frf(self, omega: np.ndarray) -> np.ndarray:
        """FRF ``G(j omega) = P(j omega)^{-1}``, valutata invertendo ``P``.

        Parameters
        ----------
        omega
            Pulsazioni in rad/s.

        Returns
        -------
        np.ndarray
            Array complesso di forma ``(len(omega), 2, 2)``.
        """
        omega = np.atleast_1d(np.asarray(omega, dtype=float))
        jw = 1j * omega
        # Costruzione vettorizzata di P(jw) e inversione in blocco.
        Pw = (
            self.M0[None, :, :] * (jw**2)[:, None, None]
            + self.C0[None, :, :] * jw[:, None, None]
            + self.K0[None, :, :]
        )
        return np.linalg.inv(Pw)

    def frf_da_spazio_di_stato(self, omega: np.ndarray) -> np.ndarray:
        """FRF calcolata da ``C (jw I - A)^{-1} B + D``, per controllo incrociato."""
        omega = np.atleast_1d(np.asarray(omega, dtype=float))
        n = self.A.shape[0]
        I = np.eye(n)
        risultato = np.empty((omega.size, 2, 2), dtype=complex)
        for k, w in enumerate(omega):
            risultato[k] = self.Cy @ np.linalg.solve(1j * w * I - self.A, self.Bu) + self.Dd
        return risultato

    def zeri_di_blocco(self) -> dict[str, np.ndarray]:
        """Zeri delle quattro FdT, cioe' le radici degli elementi di ``P``.

        Gli zeri di ``G_11`` sono le radici di ``P_22`` (antirisonanze
        dell'asse di azimut), quelli di ``G_22`` le radici di ``P_11``.
        """
        return {
            etichetta: _radici_sicure(
                polinomio(self.M0[i, j], self.C0[i, j], self.K0[i, j])
            )
            for etichetta, (i, j) in (
                ("P11 (zeri di G22)", (0, 0)),
                ("P22 (zeri di G11)", (1, 1)),
                ("P12 (zeri di G12)", (0, 1)),
                ("P21 (zeri di G21)", (1, 0)),
            )
        }

    @property
    def cancellazioni(self) -> tuple[Cancellazione, ...]:
        """Tutte le cancellazioni documentate nelle quattro FdT."""
        return tuple(c for f in self.fdt.values() for c in f.cancellazioni)


def _radici_sicure(p: np.ndarray) -> np.ndarray:
    """Radici di un polinomio, gestendo il caso identicamente nullo."""
    p = _pulisci(p, tol=1e-14)
    if p.size <= 1:
        return np.array([], dtype=complex)
    return np.roots(p)


def linearizza(
    modello: ModelloInerziale,
    theta0_val: float,
    Omega0_val: float,
    *,
    tol_cancellazione: float = 1e-8,
) -> ModelloLineare:
    """Costruisce il modello lineare numerico in ``(theta_0, Omega_0)``.

    Parameters
    ----------
    modello
        Modello inerziale.
    theta0_val
        Punto di lavoro in elevazione, in rad.
    Omega0_val
        Velocita' di inseguimento in azimut, in rad/s.
    tol_cancellazione
        Tolleranza per la cancellazione delle radici comuni nelle FdT.
    """
    lin = costruisci_modello_lineare_simbolico()
    sost = dict(lg.sostituzioni_numeriche(modello))
    sost[theta0] = float(theta0_val)
    sost[Omega0] = float(Omega0_val)

    M0 = np.array(lin.M0.subs(sost), dtype=float)
    C0 = np.array(lin.C0.subs(sost), dtype=float)
    K0 = np.array(lin.K0.subs(sost), dtype=float)
    tau0 = np.array(lin.tau0.subs(sost), dtype=float).reshape(2)

    M0_inv = np.linalg.inv(M0)
    A = np.block(
        [
            [np.zeros((2, 2)), np.eye(2)],
            [-M0_inv @ K0, -M0_inv @ C0],
        ]
    )
    Bu = np.vstack([np.zeros((2, 2)), M0_inv])
    Cy = np.hstack([np.eye(2), np.zeros((2, 2))])
    Dd = np.zeros((2, 2))

    # FdT numeriche: adj(P)/Delta, con cancellazione documentata.
    den = polinomio_delta(M0, C0, K0)
    numeratori = {
        (0, 0): polinomio(M0[1, 1], C0[1, 1], K0[1, 1]),
        (0, 1): -polinomio(M0[0, 1], C0[0, 1], K0[0, 1]),
        (1, 0): -polinomio(M0[1, 0], C0[1, 0], K0[1, 0]),
        (1, 1): polinomio(M0[0, 0], C0[0, 0], K0[0, 0]),
    }
    fdt: dict[tuple[int, int], FunzioneDiTrasferimento] = {}
    for indice, num in numeratori.items():
        etichetta = ETICHETTE_FDT[indice].split(" ")[0]
        n, d, canc = cancella_radici_comuni(
            num, den, tol=tol_cancellazione, etichetta=etichetta
        )
        fdt[indice] = FunzioneDiTrasferimento(
            nome=etichetta, num=n, den=d, cancellazioni=tuple(canc)
        )

    return ModelloLineare(
        theta0=float(theta0_val),
        Omega0=float(Omega0_val),
        M0=M0,
        C0=C0,
        K0=K0,
        A=A,
        Bu=Bu,
        Cy=Cy,
        Dd=Dd,
        tau0=tau0,
        modello_inerziale=modello,
        fdt=fdt,
    )


def griglia_modelli(
    modello: ModelloInerziale,
    theta0_vals: Sequence[float],
    Omega0_vals: Sequence[float],
) -> dict[tuple[float, float], ModelloLineare]:
    """Linearizza su una griglia di punti di lavoro."""
    return {
        (float(th), float(om)): linearizza(modello, float(th), float(om))
        for th in theta0_vals
        for om in Omega0_vals
    }
