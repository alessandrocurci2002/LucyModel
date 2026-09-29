"""Test della sezione 9, punti 7 e 8: equazioni di Lagrange."""

from __future__ import annotations

import numpy as np
import pytest
import sympy as sp

from gimbal import kinematics as kin
from gimbal import lagrange as lg


# --- 7. riduzione alle forme chiuse con prodotti nulli ---------------------


def test_forme_chiuse_con_prodotti_nulli(simbolico) -> None:
    """Le equazioni simboliche coincidono con quelle della sezione 6."""
    assert lg.verifica_forme_chiuse(simbolico) is True


def test_struttura_matrice_di_massa(simbolico) -> None:
    """``M(theta)`` ha la struttura attesa e non dipende da ``psi``."""
    assert lg.verifica_struttura_M(simbolico) is True
    assert kin.psi not in simbolico.M.free_symbols


def test_energia_potenziale(simbolico) -> None:
    """``V = m_B g (-d_x sin(theta) + d_z cos(theta))``; ``d_y`` non compare."""
    atteso = lg.m_B * lg.g * (
        -lg.d_x * sp.sin(kin.theta) + lg.d_z * sp.cos(kin.theta)
    )
    assert sp.simplify(simbolico.V - atteso) == 0
    assert lg.d_y not in simbolico.V.free_symbols


def test_gradiente_potenziale(simbolico) -> None:
    """``dV/dpsi = 0`` e ``dV/dtheta = -m_B g (d_x cos + d_z sin)``."""
    assert sp.simplify(simbolico.gradV[0]) == 0
    atteso = -lg.m_B * lg.g * (
        lg.d_x * sp.cos(kin.theta) + lg.d_z * sp.sin(kin.theta)
    )
    assert sp.simplify(simbolico.gradV[1] - atteso) == 0


def test_psi_e_ciclica(simbolico) -> None:
    """``psi`` non compare in T, V, D: e' una coordinata ciclica."""
    for espressione in (simbolico.T, simbolico.V, simbolico.D):
        assert kin.psi not in espressione.free_symbols


def test_energia_cinetica_quadratica(simbolico) -> None:
    """``T = 1/2 dq^T M dq``: nessun termine lineare o costante in ``dq``."""
    T_da_M = (kin.dq.T * simbolico.M * kin.dq / 2)[0, 0]
    assert sp.simplify(sp.expand(simbolico.T - T_da_M)) == 0


# --- 8. antisimmetria di dM/dt - 2C ----------------------------------------


def test_antisimmetria_Mdot_meno_2C(simbolico) -> None:
    """``dM/dt - 2C`` e' antisimmetrica (passivita')."""
    S = lg.verifica_antisimmetria(simbolico)
    assert sp.simplify(S + S.T) == sp.zeros(2, 2)
    assert sp.simplify(S[0, 0]) == 0 and sp.simplify(S[1, 1]) == 0


def test_coriolis_coincide_con_lagrange(simbolico) -> None:
    """``C dq`` dai simboli di Christoffel = termine di Coriolis di Lagrange."""
    Q = sp.Matrix([lg.tau_psi, lg.tau_theta])
    assert sp.simplify(sp.expand(simbolico.eom - (simbolico.residuo - Q))) == sp.zeros(2, 1)


# --- funzioni numeriche ----------------------------------------------------


def test_modello_numerico_coincide_con_simbolico(modello_attivo) -> None:
    """Le funzioni ``lambdify`` riproducono le espressioni simboliche."""
    mn = lg.costruisci_modello_numerico(modello_attivo)
    sost = lg.sostituzioni_numeriche(modello_attivo)
    sim = lg.costruisci_modello_simbolico()

    for theta, dpsi, dtheta in [(-0.8, 0.1, -0.05), (0.0, 0.0, 0.0), (-1.5, -0.2, 0.3)]:
        punto = {kin.theta: theta, kin.dpsi: dpsi, kin.dtheta: dtheta}
        M_atteso = np.array(sim.M.subs(sost).subs(punto), dtype=float)
        assert mn.M(theta) == pytest.approx(M_atteso)
        Cdq_atteso = np.array(
            (sim.C * kin.dq).subs(sost).subs(punto), dtype=float
        ).reshape(2)
        assert mn.Cdq(theta, dpsi, dtheta) == pytest.approx(Cdq_atteso)
        gradV_atteso = np.array(sim.gradV.subs(sost).subs(punto), dtype=float).reshape(2)
        assert mn.gradV(theta) == pytest.approx(gradV_atteso)


def test_matrice_di_massa_definita_positiva(modello_attivo) -> None:
    """``M(theta)`` deve essere simmetrica e definita positiva ovunque."""
    mn = lg.costruisci_modello_numerico(modello_attivo)
    for theta in np.linspace(-np.pi / 2, 0.0, 19):
        M = mn.M(theta)
        assert M == pytest.approx(M.T)
        assert np.min(np.linalg.eigvalsh(M)) > 0.0


def test_J_psi_analitico_e_derivate(modello_attivo) -> None:
    """``J_psi``, ``J_psi'`` e ``J_psi''`` sono coerenti fra loro.

    Il confronto usa i soli punti **interni**: agli estremi ``np.gradient``
    ripiega su una formula unilaterale, il cui errore di troncamento e' di un
    ordine di grandezza superiore a quello delle differenze centrate e non
    dice nulla sulla correttezza delle espressioni analitiche.
    """
    theta = np.linspace(-np.pi / 2, 0.0, 200)
    interni = slice(1, -1)
    J = modello_attivo.J_psi(theta)
    dJ_numerica = np.gradient(J, theta)
    assert modello_attivo.J_psi_prime(theta)[interni] == pytest.approx(
        dJ_numerica[interni], rel=1e-3, abs=1e-4
    )
    d2J_numerica = np.gradient(modello_attivo.J_psi_prime(theta), theta)
    assert modello_attivo.J_psi_second(theta)[interni] == pytest.approx(
        d2J_numerica[interni], rel=1e-3, abs=1e-3
    )


def test_coppie_nominali_sono_un_equilibrio(modello_attivo) -> None:
    """Con ``tau = tau_0`` il punto di lavoro ha accelerazione nulla."""
    mn = lg.costruisci_modello_numerico(modello_attivo)
    for theta0 in (0.0, -0.5, -1.2):
        for Omega0 in (0.0, np.deg2rad(10.0)):
            tau0 = mn.coppie_nominali(theta0, Omega0)
            ddq = mn.accelerazioni(
                np.array([0.0, theta0]), np.array([Omega0, 0.0]), tau0
            )
            assert ddq == pytest.approx(np.zeros(2), abs=1e-12)


def test_esporta_latex_eom(tmp_path, simbolico) -> None:
    """L'esportazione LaTeX produce un file non vuoto e con le voci attese."""
    percorso = lg.esporta_latex(tmp_path / "eom.tex", simbolico)
    testo = percorso.read_text(encoding="utf-8")
    assert r"\mathbf M(\theta)" in testo
    assert r"J_\psi(\theta)" in testo
    assert len(testo) > 1000
