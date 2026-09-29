"""Test della sezione 9, punto 6: cinematica e formula di Poisson."""

from __future__ import annotations

import numpy as np
import pytest
import sympy as sp

from gimbal import kinematics as kin
from gimbal import params as pr


def test_poisson_coincide_con_forma_chiusa() -> None:
    """``omega`` da Poisson = forma chiusa della sezione 6 (test 6)."""
    assert kin.verifica_poisson() is True


def test_omega_esplicita() -> None:
    """``omega_B|_B = (-dpsi sin(theta), dtheta, dpsi cos(theta))``."""
    cin = kin.costruisci_cinematica()
    atteso = sp.Matrix(
        [-kin.dpsi * sp.sin(kin.theta), kin.dtheta, kin.dpsi * sp.cos(kin.theta)]
    )
    assert sp.simplify(cin.omega_poisson - atteso) == sp.zeros(3, 1)


def test_jacobiano_cinematico() -> None:
    """``omega = J(theta) dq`` con ``J`` indipendente da ``psi``."""
    cin = kin.costruisci_cinematica()
    J = cin.jacobiano
    assert J.shape == (3, 2)
    assert kin.psi not in J.free_symbols  # psi e' ciclica
    assert sp.simplify(J * kin.dq - cin.omega_poisson) == sp.zeros(3, 1)


def test_R_NB_ortonormale_simbolica() -> None:
    """``R_NB`` e' una rotazione per ogni ``psi`` e ``theta``."""
    R = kin.R_NB(kin.psi, kin.theta)
    assert sp.simplify(R.T * R - sp.eye(3)) == sp.zeros(3, 3)
    assert sp.simplify(R.det() - 1) == 0


def test_convenzione_segno_theta() -> None:
    """``theta > 0`` abbassa il puntamento, ``theta = -90 deg`` punta allo zenit.

    E' la convenzione vincolante della sezione 2: un errore di segno qui
    invertirebbe tutta la dinamica gravitazionale.
    """
    x_B = pr.R_NB(0.0, 0.0) @ np.array([1.0, 0.0, 0.0])
    assert x_B == pytest.approx([1.0, 0.0, 0.0])  # orizzontale

    x_B_giu = pr.R_NB(0.0, np.deg2rad(30.0)) @ np.array([1.0, 0.0, 0.0])
    assert x_B_giu[2] < 0.0  # theta > 0 abbassa

    x_B_zenit = pr.R_NB(0.0, np.deg2rad(-90.0)) @ np.array([1.0, 0.0, 0.0])
    assert x_B_zenit == pytest.approx([0.0, 0.0, 1.0], abs=1e-12)  # zenit


def test_asse_di_elevazione_sempre_orizzontale() -> None:
    """``y_B`` resta orizzontale: per questo ``d_y`` non entra in ``V``."""
    for psi in np.linspace(0.0, 2 * np.pi, 9):
        for theta in np.linspace(-np.pi / 2, 0.0, 7):
            y_B = pr.R_NB(psi, theta) @ np.array([0.0, 1.0, 0.0])
            assert y_B[2] == pytest.approx(0.0, abs=1e-15)


def test_derivata_totale() -> None:
    """La derivata totale applica la regola della catena su psi e theta."""
    espressione = sp.sin(kin.theta) * kin.dpsi
    attesa = sp.cos(kin.theta) * kin.dtheta * kin.dpsi + sp.sin(kin.theta) * kin.ddpsi
    assert sp.simplify(kin.derivata_totale(espressione) - attesa) == 0


def test_hat_vee_inverse() -> None:
    """``vee`` e ``hat`` sono l'una l'inversa dell'altra."""
    w = sp.Matrix(sp.symbols("w1 w2 w3", real=True))
    assert kin.vee(kin.hat(w)) == w


def test_costruzione_terne_assert_incoerente() -> None:
    """Se ``x_A0`` ed ``elev_axis_W`` sono incoerenti, l'assert deve scattare."""
    with pytest.raises(AssertionError, match="non e' parallelo"):
        pr.costruisci_terne(pr.direzione("+X"), pr.direzione("X"), 0.0)
    # La combinazione coerente passa.
    y, R_WA0, R_WB0 = pr.costruisci_terne(pr.direzione("+X"), pr.direzione("Y"), 0.0)
    assert y == pytest.approx([0.0, 1.0, 0.0])
    assert R_WA0 == pytest.approx(np.eye(3))
    assert R_WB0 == pytest.approx(np.eye(3))


def test_x_A0_deve_essere_orizzontale() -> None:
    with pytest.raises(ValueError, match="orizzontale"):
        pr.costruisci_terne(pr.direzione("Z"), pr.direzione("X"), 0.0)
