"""Cinematica simbolica della catena N -> A -> B (sezione 6 della specifica).

La velocita' angolare del telescopio si ricava dalla **formula di Poisson**,

.. math:: [\\omega_B|_B]_\\times = R_{NB}^{\\mathsf T}\\,\\dot R_{NB},

e non viene scritta a mano: il modulo la deriva con SymPy e la confronta con
l'espressione in forma chiusa della specifica. Il confronto e' esposto come
funzione (:func:`verifica_poisson`) ed e' uno dei test della sezione 9.

Convenzione vincolante (sezione 2): ``R_NB = R_z(psi) R_y(theta)``, con
``theta > 0`` che abbassa il puntamento.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final

import sympy as sp

# ---------------------------------------------------------------------------
# Simboli
# ---------------------------------------------------------------------------

t: Final[sp.Symbol] = sp.Symbol("t", real=True)
"""Tempo."""

# Coordinate come funzioni del tempo: servono per derivare R_NB(t).
psi_t: Final[sp.Expr] = sp.Function("psi", real=True)(t)
theta_t: Final[sp.Expr] = sp.Function("theta", real=True)(t)

# Coordinate come simboli indipendenti: servono nelle equazioni algebriche.
psi, theta = sp.symbols("psi theta", real=True)
dpsi, dtheta = sp.symbols(r"\dot{\psi} \dot{\theta}", real=True)
ddpsi, ddtheta = sp.symbols(r"\ddot{\psi} \ddot{\theta}", real=True)

q: Final[sp.Matrix] = sp.Matrix([psi, theta])
"""Coordinate generalizzate ``q = [psi, theta]^T``."""

dq: Final[sp.Matrix] = sp.Matrix([dpsi, dtheta])
ddq: Final[sp.Matrix] = sp.Matrix([ddpsi, ddtheta])


# ---------------------------------------------------------------------------
# Rotazioni elementari
# ---------------------------------------------------------------------------


def R_z(angolo: sp.Expr) -> sp.Matrix:
    """Rotazione attorno a z (azimut), regola della mano destra."""
    c, s = sp.cos(angolo), sp.sin(angolo)
    return sp.Matrix([[c, -s, 0], [s, c, 0], [0, 0, 1]])


def R_y(angolo: sp.Expr) -> sp.Matrix:
    """Rotazione attorno a y nella convenzione della sezione 2.

    .. math:: R_y(\\theta) = \\begin{bmatrix} c & 0 & s \\\\ 0 & 1 & 0 \\\\
              -s & 0 & c \\end{bmatrix}
    """
    c, s = sp.cos(angolo), sp.sin(angolo)
    return sp.Matrix([[c, 0, s], [0, 1, 0], [-s, 0, c]])


def R_NB(angolo_psi: sp.Expr, angolo_theta: sp.Expr) -> sp.Matrix:
    """Assetto del telescopio rispetto alla terna inerziale N."""
    return R_z(angolo_psi) * R_y(angolo_theta)


def vee(S: sp.Matrix) -> sp.Matrix:
    """Inversa dell'operatore ``[.]_x``: da matrice antisimmetrica a vettore.

    Con la convenzione ``[w]_x v = w x v`` si ha
    ``w = (S[2,1], S[0,2], S[1,0])``.
    """
    return sp.Matrix([S[2, 1], S[0, 2], S[1, 0]])


def hat(w: sp.Matrix) -> sp.Matrix:
    """Operatore ``[.]_x``: da vettore a matrice antisimmetrica."""
    return sp.Matrix([[0, -w[2], w[1]], [w[2], 0, -w[0]], [-w[1], w[0], 0]])


# ---------------------------------------------------------------------------
# Sostituzioni fra forma temporale e forma algebrica
# ---------------------------------------------------------------------------

SOSTITUZIONI_A_SIMBOLI: Final[dict[sp.Expr, sp.Symbol]] = {
    sp.Derivative(psi_t, (t, 2)): ddpsi,
    sp.Derivative(theta_t, (t, 2)): ddtheta,
    sp.Derivative(psi_t, t): dpsi,
    sp.Derivative(theta_t, t): dtheta,
    psi_t: psi,
    theta_t: theta,
}
"""Da ``psi(t), theta(t)`` e derivate ai simboli algebrici.

L'ordine e' significativo: le derivate di ordine piu' alto vanno sostituite
prima, altrimenti ``psi(t) -> psi`` renderebbe le derivate nulle.
"""

SOSTITUZIONI_A_FUNZIONI: Final[dict[sp.Symbol, sp.Expr]] = {
    ddpsi: sp.Derivative(psi_t, (t, 2)),
    ddtheta: sp.Derivative(theta_t, (t, 2)),
    dpsi: sp.Derivative(psi_t, t),
    dtheta: sp.Derivative(theta_t, t),
    psi: psi_t,
    theta: theta_t,
}
"""Sostituzione inversa, usata per derivare rispetto al tempo."""


def a_simboli(espressione: sp.Expr | sp.Matrix) -> sp.Expr | sp.Matrix:
    """Converte un'espressione dalla forma temporale a quella algebrica."""
    risultato = espressione
    for vecchio, nuovo in SOSTITUZIONI_A_SIMBOLI.items():
        risultato = risultato.subs(vecchio, nuovo)
    return risultato


def a_funzioni(espressione: sp.Expr | sp.Matrix) -> sp.Expr | sp.Matrix:
    """Converte un'espressione dalla forma algebrica a quella temporale."""
    risultato = espressione
    for vecchio, nuovo in SOSTITUZIONI_A_FUNZIONI.items():
        risultato = risultato.subs(vecchio, nuovo)
    return risultato


def derivata_totale(espressione: sp.Expr | sp.Matrix) -> sp.Expr | sp.Matrix:
    """Derivata totale rispetto al tempo di un'espressione in forma algebrica.

    Passa in forma temporale, deriva, e torna in forma algebrica: in questo
    modo la regola della catena su ``psi`` e ``theta`` e' gestita da SymPy.
    """
    return a_simboli(sp.diff(a_funzioni(espressione), t))


# ---------------------------------------------------------------------------
# Velocita' angolare
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Cinematica:
    """Risultati della derivazione cinematica.

    Attributes
    ----------
    R
        ``R_NB(psi, theta)`` in forma algebrica.
    omega_poisson
        ``omega_B|_B`` ricavata dalla formula di Poisson.
    omega_forma_chiusa
        ``omega_B|_B`` nella forma chiusa della specifica,
        ``(-dpsi sin(theta), dtheta, dpsi cos(theta))``.
    jacobiano
        Matrice ``J(theta)`` tale che ``omega_B|_B = J(theta) dq``.
    """

    R: sp.Matrix
    omega_poisson: sp.Matrix
    omega_forma_chiusa: sp.Matrix
    jacobiano: sp.Matrix


def costruisci_cinematica() -> Cinematica:
    """Deriva ``omega_B|_B`` con Poisson e la forma chiusa di confronto."""
    R_t = R_NB(psi_t, theta_t)
    S = sp.simplify(R_t.T * sp.diff(R_t, t))  # [omega_B|_B]_x
    omega_poisson = sp.simplify(a_simboli(vee(S)))

    # Forma chiusa della specifica: composizione delle due rotazioni.
    omega_chiusa = sp.simplify(
        R_y(theta).T * sp.Matrix([0, 0, dpsi]) + sp.Matrix([0, dtheta, 0])
    )

    # omega e' lineare in dq: il jacobiano si estrae per derivazione.
    J = sp.simplify(omega_poisson.jacobian(dq))

    return Cinematica(
        R=R_NB(psi, theta),
        omega_poisson=omega_poisson,
        omega_forma_chiusa=omega_chiusa,
        jacobiano=J,
    )


def verifica_poisson(cin: Cinematica | None = None) -> bool:
    """Verifica che Poisson e forma chiusa coincidano (sezione 9, test 6).

    Controlla anche l'antisimmetria di ``R^T dR/dt``, che e' la condizione di
    consistenza della formula di Poisson.
    """
    cin = cin or costruisci_cinematica()

    differenza = sp.simplify(cin.omega_poisson - cin.omega_forma_chiusa)
    if differenza != sp.zeros(3, 1):
        raise AssertionError(
            f"omega da Poisson e forma chiusa differiscono: {differenza.T}"
        )

    R_t = R_NB(psi_t, theta_t)
    S = sp.simplify(R_t.T * sp.diff(R_t, t))
    if sp.simplify(S + S.T) != sp.zeros(3, 3):
        raise AssertionError("R^T dR/dt non e' antisimmetrica")

    atteso = sp.Matrix([-dpsi * sp.sin(theta), dtheta, dpsi * sp.cos(theta)])
    if sp.simplify(cin.omega_poisson - atteso) != sp.zeros(3, 1):
        raise AssertionError("omega non coincide con l'espressione della sezione 6")

    return True
