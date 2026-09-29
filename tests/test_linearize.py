"""Test della sezione 9, punti 9 e 10: linearizzazione e coerenza."""

from __future__ import annotations

import numpy as np
import pytest
import sympy as sp

from gimbal import analysis as an
from gimbal import inertia as inz
from gimbal import lagrange as lg
from gimbal import linearize as lz
from gimbal import params as pr


# --- 9. target di regressione della sezione 7 ------------------------------


def test_target_regressione_matrici(lineare_simbolico) -> None:
    """``M_0``, ``C_0``, ``K_0`` coincidono con i target con prodotti nulli."""
    assert lz.verifica_target_regressione() is True


def test_prima_colonna_di_K0_nulla(lineare_simbolico) -> None:
    """``psi`` e' ciclica: la prima colonna di ``K_0`` e' identicamente nulla."""
    assert sp.simplify(lineare_simbolico.K0[:, 0]) == sp.zeros(2, 1)


def test_gamma_e_k_theta_simbolici(lineare_simbolico) -> None:
    """Espressioni di ``gamma`` e ``k_theta`` con prodotti nulli."""
    gamma = sp.simplify(lineare_simbolico.gamma.subs(lg.PRODOTTI_NULLI))
    atteso = (lg.I_xx - lg.I_zz) * lz.Omega0 * sp.sin(2 * lz.theta0)
    assert sp.simplify(gamma - atteso) == 0

    k = sp.simplify(lineare_simbolico.k_theta.subs(lg.PRODOTTI_NULLI))
    k_c = -(lg.I_xx - lg.I_zz) * lz.Omega0**2 * sp.cos(2 * lz.theta0)
    k_g = lg.m_B * lg.g * (lg.d_x * sp.sin(lz.theta0) - lg.d_z * sp.cos(lz.theta0))
    assert sp.simplify(k - (k_c + k_g)) == 0


def test_C0_antisimmetrica_senza_smorzamento(lineare_simbolico) -> None:
    """Senza smorzamento ``C_0`` e' antisimmetrica: e' un termine giroscopico."""
    C0 = lineare_simbolico.C0.subs({lg.b_psi: 0, lg.b_theta: 0})
    assert sp.simplify(C0 + C0.T) == sp.zeros(2, 2)


def test_fdt_target_omega_nullo_e_d_nullo() -> None:
    """``Omega_0 = 0``, prodotti nulli, ``d = 0``: le FdT della sezione 7.

    Attese ``G_11 = 1/(s(J_psi s + b_psi))``, ``G_22 = 1/(s(I_yy s + b_theta))``
    e ``G_12 = G_21 = 0``.
    """
    lin = lz.costruisci_modello_lineare_simbolico()
    sost = {lz.Omega0: 0, lg.d_x: 0, lg.d_y: 0, lg.d_z: 0, **lg.PRODOTTI_NULLI}
    G = sp.Matrix(2, 2, lambda i, j: sp.cancel(sp.together(lin.G[i, j].subs(sost))))

    J_psi = lg.J_A + lg.I_xx * sp.sin(lz.theta0) ** 2 + lg.I_zz * sp.cos(lz.theta0) ** 2
    atteso_11 = 1 / (lz.s * (J_psi * lz.s + lg.b_psi))
    atteso_22 = 1 / (lz.s * (lg.I_yy * lz.s + lg.b_theta))

    assert sp.simplify(G[0, 0] - atteso_11) == 0
    assert sp.simplify(G[1, 1] - atteso_22) == 0
    assert sp.simplify(G[0, 1]) == 0
    assert sp.simplify(G[1, 0]) == 0


def test_fdt_target_con_sbilanciamento() -> None:
    """Con ``d != 0`` e ``Omega_0 = 0``: ``G_22 = 1/(I_yy s^2 + b_theta s + k_g)``."""
    lin = lz.costruisci_modello_lineare_simbolico()
    sost = {lz.Omega0: 0, **lg.PRODOTTI_NULLI}
    G22 = sp.cancel(sp.together(lin.G[1, 1].subs(sost)))
    k_g = lg.m_B * lg.g * (lg.d_x * sp.sin(lz.theta0) - lg.d_z * sp.cos(lz.theta0))
    atteso = 1 / (lg.I_yy * lz.s**2 + lg.b_theta * lz.s + k_g)
    assert sp.simplify(G22 - atteso) == 0


def test_valori_numerici_gamma_k_theta(modello_attivo) -> None:
    """Le formule di ``gamma`` e ``k_theta`` valgono anche numericamente."""
    m = modello_attivo
    for theta0 in (0.0, -0.4, -np.pi / 2):
        for Omega0 in (0.0, np.deg2rad(5.0), np.deg2rad(10.0)):
            ml = lz.linearizza(m, theta0, Omega0)
            # Con prodotti non nulli i target valgono a meno dei termini in I_xz.
            k_g = m.m_B * m.g * (m.d_x * np.sin(theta0) - m.d_z * np.cos(theta0))
            k_c = -0.5 * float(m.J_psi_second(theta0)) * Omega0**2
            assert ml.k_theta == pytest.approx(k_c + k_g, rel=1e-10, abs=1e-12)
            assert ml.gamma == pytest.approx(
                float(m.J_psi_prime(theta0)) * Omega0, rel=1e-10, abs=1e-12
            )


def test_coppie_nominali_linearizzazione(modello_attivo) -> None:
    """``tau_0`` del modello lineare = quelle del modello non lineare."""
    mn = lg.costruisci_modello_numerico(modello_attivo)
    for theta0 in (0.0, -0.7):
        for Omega0 in (0.0, np.deg2rad(10.0)):
            ml = lz.linearizza(modello_attivo, theta0, Omega0)
            assert ml.tau0 == pytest.approx(mn.coppie_nominali(theta0, Omega0))


# --- 10. coerenza fra le rappresentazioni ----------------------------------


@pytest.mark.parametrize("theta0_deg", [0.0, -30.0, -60.0, -90.0])
@pytest.mark.parametrize("omega0_deg_s", [0.0, 10.0])
def test_coerenza_rappresentazioni(modello_attivo, theta0_deg, omega0_deg_s) -> None:
    """Autovalori di ``A`` = radici di ``Delta``; FRF da stato = ``P(jw)^-1``."""
    ml = lz.linearizza(
        modello_attivo, float(np.deg2rad(theta0_deg)), float(np.deg2rad(omega0_deg_s))
    )
    scarti = an.verifica_coerenza(ml)
    assert scarti["scarto_poli"] < 1e-8
    assert scarti["scarto_frf"] < 1e-8


def test_frf_coincide_con_inversa_di_P(modello_attivo) -> None:
    """``G(jw) P(jw) = I``: la FRF e' davvero l'inversa di ``P``."""
    ml = lz.linearizza(modello_attivo, -0.7, np.deg2rad(5.0))
    omega = np.logspace(-3, 2, 40) * 2 * np.pi
    G = ml.frf(omega)
    for k, w in enumerate(omega):
        assert G[k] @ ml.P(1j * w) == pytest.approx(np.eye(2), abs=1e-9)


def test_fdt_polinomiali_coincidono_con_frf(modello_attivo) -> None:
    """Le FdT come rapporto di polinomi coincidono con la FRF matriciale."""
    ml = lz.linearizza(modello_attivo, -0.7, np.deg2rad(5.0))
    omega = np.logspace(-2, 1, 30) * 2 * np.pi
    G = ml.frf(omega)
    for (i, j), fdt in ml.fdt.items():
        if fdt.nulla:
            assert np.max(np.abs(G[:, i, j])) == pytest.approx(0.0, abs=1e-12)
            continue
        atteso = fdt.valuta(1j * omega)
        scala = np.max(np.abs(G[:, i, j]))
        assert np.max(np.abs(atteso - G[:, i, j])) / scala < 1e-9


# --- cancellazione delle radici comuni -------------------------------------


def test_cancellazione_del_polo_nell_origine(modello_attivo) -> None:
    """``G_12``, ``G_21`` e ``G_22`` cancellano il fattore ``s`` di ``Delta``.

    ``psi`` e' ciclica, quindi ``Delta`` ha una radice nell'origine che quei
    numeratori condividono; ``G_11`` invece la conserva come polo.
    """
    ml = lz.linearizza(modello_attivo, -0.7, np.deg2rad(5.0))
    grado_delta = lz._pulisci(ml.delta, 1e-14).size - 1
    assert ml.fdt[(0, 0)].den.size - 1 == grado_delta  # nessuna cancellazione
    for indice in ((0, 1), (1, 0), (1, 1)):
        assert ml.fdt[indice].cancellazioni
        assert all(abs(c.radice) < 1e-9 for c in ml.fdt[indice].cancellazioni)


def test_cancella_radici_comuni_su_caso_noto() -> None:
    """(s+2) si cancella fra numeratore e denominatore."""
    num = np.polymul([1.0, 2.0], [1.0, 5.0])
    den = np.polymul([1.0, 2.0], [1.0, 3.0])
    n, d, canc = lz.cancella_radici_comuni(num, den, etichetta="prova")
    assert len(canc) == 1
    assert canc[0].radice == pytest.approx(-2.0)
    assert np.roots(n) == pytest.approx([-5.0])
    assert np.roots(d) == pytest.approx([-3.0])


def test_nessuna_cancellazione_fuori_tolleranza() -> None:
    """Radici vicine ma distinte non vanno cancellate."""
    num = [1.0, 2.0]
    den = np.polymul([1.0, 2.0 + 1e-3], [1.0, 3.0])
    _, _, canc = lz.cancella_radici_comuni(num, den, tol=1e-8)
    assert canc == []


# --- classificazione delle radici ------------------------------------------


def test_classificazione_radici() -> None:
    """Le radici sono classificate secondo la sezione 8.1."""
    assert an.classifica_radice(0.0).tipo == "nullo"
    assert an.classifica_radice(-2.0).tipo == "reale_stabile"
    assert an.classifica_radice(+2.0).tipo == "reale_instabile"
    complessa = an.classifica_radice(-0.5 + 3.0j)
    assert complessa.tipo == "complesso"
    assert complessa.f_hz == pytest.approx(abs(-0.5 + 3.0j) / (2 * np.pi))
    assert complessa.zeta == pytest.approx(0.5 / abs(-0.5 + 3.0j))
    # La classificazione dichiara l'origine fisica: niente modi strutturali.
    assert "strutturale" in complessa.origine


def test_poli_nulli_per_coordinata_ciclica(modello_riferimento) -> None:
    """Senza smorzamento e senza gravita' restano poli nell'origine."""
    par = pr.parametri_di_riferimento(z_O_cm=None)
    mod = inz.costruisci_modello_inerziale(par)
    ml = lz.linearizza(mod, 0.0, 0.0)
    radici = an.classifica(ml.poli)
    assert sum(1 for r in radici if r.tipo == "nullo") >= 2


# --- forma compatta delle FdT ----------------------------------------------


def test_forma_compatta_riproduce_quella_estesa() -> None:
    """La forma in ``J_psi``, ``gamma``, ``k_theta`` equivale a quella estesa.

    E' la forma esportata in ``tf.tex`` e citata in tesi: deve essere
    un'abbreviazione esatta, non un'approssimazione.
    """
    compatta = lz.forma_compatta()
    lin = lz.costruisci_modello_lineare_simbolico()
    definizioni = compatta["definizioni"]

    G_esteso = sp.Matrix(
        2, 2, lambda i, j: sp.cancel(sp.together(lin.G[i, j].subs(lg.PRODOTTI_NULLI)))
    )
    G_compatto = compatta["G"].subs(definizioni)
    for i in range(2):
        for j in range(2):
            assert sp.simplify(G_compatto[i, j] - G_esteso[i, j]) == 0


def test_delta_compatto_si_fattorizza_con_s() -> None:
    """``Delta = s D(s)``: il polo nell'origine della coordinata ciclica."""
    compatta = lz.forma_compatta()
    assert sp.simplify(sp.expand(compatta["Delta"] - lz.s * compatta["D"])) == 0
    assert compatta["Delta"].subs(lz.s, 0) == 0
    # D(0) = b_psi k_theta: non e' nullo in generale, quindi la radice e' semplice.
    assert sp.simplify(compatta["D"].subs(lz.s, 0) - lg.b_psi * lz.k_theta_s) == 0


def test_esporta_latex_tf(tmp_path) -> None:
    """``tf.tex`` contiene la forma compatta e non quella illeggibile estesa."""
    percorso = lz.esporta_latex(tmp_path / "tf.tex")
    testo = percorso.read_text(encoding="utf-8")
    assert r"J_\psi" in testo and r"\gamma" in testo and r"k_\theta" in testo
    assert r"\Delta(s) = s\,D(s)" in testo
    # Nessuna riga sproporzionata: la forma estesa ne produrrebbe da pagine.
    assert max(len(r) for r in testo.splitlines()) < 600
