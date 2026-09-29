"""Test della sezione 9, punti 11 e 12: simulazioni non lineari."""

from __future__ import annotations

import numpy as np
import pytest

from gimbal import lagrange as lg
from gimbal import linearize as lz
from gimbal import simulate as sim


# --- 11. scenario S4: conservazione di energia e momento -------------------


def test_S4_conserva_energia_e_momento(modello_riferimento) -> None:
    """Con ``b = 0`` e ``tau = 0`` la deriva relativa resta sotto ``1e-8``.

    E' il controllo piu' severo sulla correttezza delle equazioni: un errore
    di segno o di derivazione romperebbe la conservazione.
    """
    r = sim.S4(modello_riferimento, n_punti=2001, t_finale=20.0)
    assert r.diagnostica["deriva_relativa_energia"] < 1e-8
    assert r.diagnostica["deriva_relativa_momento"] < 1e-8
    assert r.esito == "ok"


def test_S4_conserva_anche_con_forte_sbilanciamento(modello_attivo) -> None:
    """La conservazione non dipende dalla configurazione scelta."""
    r = sim.S4(modello_attivo, n_punti=2001, t_finale=10.0)
    assert r.diagnostica["deriva_relativa_energia"] < 1e-8
    assert r.diagnostica["deriva_relativa_momento"] < 1e-8


def test_momento_psi_e_energia_costanti_lungo_la_traiettoria(modello_riferimento) -> None:
    """Verifica diretta punto per punto, non solo sulla diagnostica."""
    mod0, mn0 = sim.modello_conservativo(modello_riferimento)
    r = sim.S4(modello_riferimento, n_punti=801, t_finale=8.0)
    E = np.array([mn0.energia(r.q[k], r.dq[k]) for k in range(r.t.size)])
    p = np.array([mn0.momento_psi(r.q[k], r.dq[k]) for k in range(r.t.size)])
    assert np.max(np.abs(E - E[0])) / abs(E[0]) < 1e-8
    assert np.max(np.abs(p - p[0])) / abs(p[0]) < 1e-8


def test_smorzamento_dissipa_energia(modello_riferimento) -> None:
    """Controprova: con ``b > 0`` l'energia deve calare, non conservarsi."""
    from dataclasses import replace

    par = replace(modello_riferimento.par, b_psi=0.5, b_theta=0.5)
    mod = replace(modello_riferimento, par=par)
    mn = lg.costruisci_modello_numerico(mod)
    t = np.linspace(0.0, 5.0, 501)
    q, dq = sim.integra_nonlineare(
        mn, t, np.array([0.0, -1.0]), np.array([0.3, 0.2]), lambda _: np.zeros(2)
    )
    E = np.array([mn.energia(q[k], dq[k]) for k in range(t.size)])
    assert E[-1] < E[0]
    assert np.all(np.diff(E) <= 1e-9)


# --- 12. scenari S1 e S3: accordo lineare/non lineare ----------------------


def test_S1_accordo_lineare(modello_riferimento) -> None:
    """S1: errore sulla coordinata guidata sotto l'1 % (sezione 8.4)."""
    r = sim.S1(modello_riferimento, n_punti=2001)
    assert r.errori["guidata"] < 0.01
    assert r.diagnostica["picco_dpsi_deg"] < 1.0  # vincolo |delta psi| < 1 deg
    assert r.esito == "ok"


def test_S3_accordo_lineare(modello_riferimento) -> None:
    """S3: inseguimento a Omega_0 massima, errore sotto l'1 % sulla guidata.

    Sulla coordinata **incrociata** il criterio dell'1 % non si applica a
    un'ampiezza qualsiasi: il rapporto fra la risposta di secondo ordine
    (termine centrifugo, quadratico in ``dpsi``) e quella di primo ordine
    (accoppiamento giroscopico, proporzionale a ``gamma = J_psi' Omega_0``)
    vale circa ``dpsi/(2 Omega_0)``, quindi cresce con l'ampiezza della
    perturbazione. Il test verifica allora che l'1 % sia raggiunto riducendo
    l'ampiezza, il che e' esattamente cio' che significa "linearizzazione
    corretta".
    """
    r = sim.S3(modello_riferimento, n_punti=2001)
    assert r.errori["guidata"] < 0.01
    assert r.esito == "ok"

    ridotto = sim.scenario_doppietto(
        modello_riferimento,
        nome="S3-ridotto",
        descrizione="S3 con ampiezza ridotta",
        theta0=float(np.deg2rad(-45.0)),
        Omega0=modello_riferimento.par.omega0_max,
        canale=0,
        coordinata=0,
        bersaglio_deg=0.005,
        n_punti=2001,
    )
    assert ridotto.errori["guidata"] < 0.01
    assert ridotto.errori["incrociata"] < 0.01


def test_S2_asse_instabile_segnalato(modello_attivo) -> None:
    """Asse theta instabile ad anello aperto: orizzonte ridotto e nota.

    Con i dati CAD correnti l'instabilita' non viene da ``d_z`` (il telescopio
    e' quasi bilanciato in verticale, ``d_z`` vale -0.44 cm) ma da ``d_x``,
    l'offset di quasi 8 cm lungo l'asse di puntamento: a theta negativi il
    baricentro finisce sopra l'asse di elevazione e il sistema e' un pendolo
    inverso. Il test guarda quindi la rigidezza equivalente, non il segno di
    un singolo offset.
    """
    ml = lz.linearizza(modello_attivo, float(np.deg2rad(-45.0)), 0.0)
    assert ml.k_theta < 0.0
    assert np.max(np.real(ml.poli)) > 0.0

    r = sim.S2(modello_attivo, n_punti=1501)
    assert r.diagnostica["t_finale_s"] < 20.0
    assert any("instabile ad anello aperto" in nota for nota in r.note)


@pytest.mark.parametrize("Omega0_deg_s", [0.0, 10.0])
def test_convergenza_di_ordine_uno(modello_riferimento, Omega0_deg_s: float) -> None:
    """L'errore relativo scala linearmente con l'ampiezza della perturbazione.

    E' la verifica quantitativa che ``M_0``, ``C_0``, ``K_0`` siano corrette:
    una linearizzazione esatta al primo ordine lascia un residuo
    ``O(ampiezza^2)``, quindi un errore relativo ``O(ampiezza)``.
    """
    c = sim.convergenza_lineare(
        modello_riferimento,
        Omega0=float(np.deg2rad(Omega0_deg_s)),
        nome="prova",
        n_punti=1501,
    )
    assert 0.7 < c.pendenza_incrociata < 1.3
    assert c.errore_incrociata[-1] < c.errore_incrociata[0]


# --- integrazione e forzanti ----------------------------------------------


def test_doppietto_a_media_nulla() -> None:
    """Il doppietto ha integrale nullo: non sposta il punto di lavoro."""
    t = np.linspace(0.0, 10.0, 100001)
    u = sim.doppietto(t, t_inizio=1.0, durata=0.5, ampiezza=3.0)
    assert np.trapezoid(u, t) == pytest.approx(0.0, abs=1e-3)
    assert np.max(u) == pytest.approx(3.0)
    assert np.min(u) == pytest.approx(-3.0)


def test_integrazione_conserva_il_punto_di_lavoro(modello_attivo) -> None:
    """Con solo il feedforward nominale il punto di lavoro non si muove."""
    mn = lg.costruisci_modello_numerico(modello_attivo)
    theta0, Omega0 = -0.7, float(np.deg2rad(10.0))
    tau0 = mn.coppie_nominali(theta0, Omega0)
    t = np.linspace(0.0, 1.0, 201)
    q, dq = sim.integra_nonlineare(
        mn, t, np.array([0.0, theta0]), np.array([Omega0, 0.0]), lambda _: tau0
    )
    assert np.max(np.abs(q[:, 1] - theta0)) < 1e-9
    assert q[:, 0] == pytest.approx(Omega0 * t, abs=1e-9)


def test_orizzonte_sicuro_riduce_solo_se_instabile(modello_attivo, modello_riferimento) -> None:
    """L'orizzonte viene accorciato solo in presenza di un polo instabile."""
    ml_instabile = lz.linearizza(modello_attivo, -0.7, 0.0)
    t, note = sim.orizzonte_sicuro(ml_instabile, 20.0)
    assert t < 20.0 and note

    ml_marginale = lz.linearizza(modello_riferimento, 0.0, 0.0)
    if np.max(np.real(ml_marginale.poli)) <= 1e-9:
        t2, note2 = sim.orizzonte_sicuro(ml_marginale, 20.0)
        assert t2 == 20.0 and not note2


def test_salvataggio_csv(modello_riferimento, tmp_path) -> None:
    """Le storie temporali vengono esportate in CSV con l'intestazione attesa."""
    r = sim.S1(modello_riferimento, n_punti=201)
    percorso = r.salva_csv(tmp_path / "S1.csv")
    righe = percorso.read_text(encoding="utf-8").splitlines()
    assert righe[0].startswith("t_s,psi_deg,theta_deg")
    assert "dpsi_lin_deg" in righe[0]
    assert len(righe) == 202
