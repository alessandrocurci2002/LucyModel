"""Test dell'estensione flessibile a 4 GdL (sezione 10).

I parametri usati qui sono **segnaposto** e servono solo a verificare
proprieta' **qualitative** del modello (interlacciamento, limite rigido,
antirisonanze), che non dipendono dai valori numerici. Le frequenze che ne
risultano non sono frequenze della montatura e non vanno citate.
"""

from __future__ import annotations

import numpy as np
import pytest

from gimbal import flex4dof as fx
from gimbal.params import ParametriFlex


@pytest.fixture(scope="module")
def flex_segnaposto() -> ParametriFlex:
    """Parametri di prova, dichiaratamente non fisici."""
    return ParametriFlex(
        J_m_psi=0.02, J_m_theta=0.01,
        k_psi=2.0e5, k_theta=8.0e5,
        c_psi=5.0, c_theta=3.0,
    )


@pytest.fixture(scope="module")
def modello_flex(modello_riferimento, flex_segnaposto) -> fx.ModelloFlex:
    return fx.costruisci(
        modello_riferimento, float(np.deg2rad(-45.0)),
        flex=flex_segnaposto, consenti_segnaposto=True,
    )


# --- rifiuto senza parametri reali -----------------------------------------


def test_rifiuta_senza_parametri(modello_riferimento) -> None:
    """Senza parametri reali il modulo deve rifiutarsi di produrre risultati."""
    with pytest.raises(fx.ParametriFlexMancanti) as errore:
        fx.costruisci(modello_riferimento, 0.0)
    messaggio = str(errore.value)
    assert "J_m_psi" in messaggio
    assert "datasheet" in messaggio
    assert "non e' un risultato" in messaggio


def test_esegui_restituisce_none_senza_parametri(modello_riferimento, capsys) -> None:
    """``esegui`` stampa un messaggio chiaro e non produce numeri."""
    assert fx.esegui(modello_riferimento, 0.0) is None
    catturato = capsys.readouterr().out
    assert "non calcolabile" in catturato
    assert "identificazione sperimentale" in catturato


def test_segnaposto_richiede_consenso_esplicito(modello_riferimento, flex_segnaposto) -> None:
    """I parametri segnaposto passano solo con ``consenti_segnaposto=True``."""
    mf = fx.costruisci(
        modello_riferimento, 0.0, flex=flex_segnaposto, consenti_segnaposto=True
    )
    assert mf.segnaposto is True


# --- test della sezione 10 -------------------------------------------------


def test_interlacciamento_poli_zeri(modello_flex) -> None:
    """Le FRF collocate non smorzate hanno poli e zeri alternati."""
    assert fx.verifica_interlacciamento(modello_flex) == {
        "azimut": True, "elevazione": True
    }


def test_antirisonanze_uguali_alle_frequenze_a_rotore_bloccato(modello_flex) -> None:
    """Le antirisonanze collocate = frequenze proprie a rotore bloccato."""
    assert fx.verifica_antirisonanze(modello_flex) < 1e-9


@pytest.mark.parametrize("configurazione", ["riferimento", "attiva"])
def test_limite_rigido(
    modello_riferimento, modello_attivo, flex_segnaposto, configurazione: str
) -> None:
    """Nel limite ``k -> infinito`` si ritrovano le FdT del modello rigido.

    I termini diagonali convergono; il residuo sui fuori diagonale deve
    coincidere con l'accoppiamento ``M_12``, che il modello della sezione 10
    non contiene. Il confronto e' contro la previsione, non contro una soglia
    tarata: vale quindi per entrambe le configurazioni, benche' il residuo
    relativo cambi di tre ordini di grandezza fra le due.
    """
    modello = modello_riferimento if configurazione == "riferimento" else modello_attivo
    scarti = fx.verifica_limite_rigido(
        modello, float(np.deg2rad(-45.0)), flex_segnaposto
    )
    assert scarti["scarto_diagonale"] < 1e-6
    assert scarti["scarto_previsione"] < 1e-6
    assert scarti["scarto_fuori_diagonale"] == pytest.approx(
        scarti["previsione_fuori_diagonale"], rel=1e-6
    )


def test_inerzie_strutturali_tolgono_i_rotori(modello_riferimento, flex_segnaposto) -> None:
    """``J_psi^s = J_psi - J_m_psi`` e ``I_yy^s = I_yy - J_m_theta``."""
    theta0 = float(np.deg2rad(-45.0))
    mf = fx.costruisci(
        modello_riferimento, theta0, flex=flex_segnaposto, consenti_segnaposto=True
    )
    assert mf.M[1, 1] == pytest.approx(
        float(modello_riferimento.J_psi(theta0)) - flex_segnaposto.J_m_psi
    )
    assert mf.M[3, 3] == pytest.approx(
        modello_riferimento.I_yy - flex_segnaposto.J_m_theta
    )
    # La somma rotore + struttura riproduce l'inerzia rigida.
    assert mf.M[0, 0] + mf.M[1, 1] == pytest.approx(float(modello_riferimento.J_psi(theta0)))
    assert mf.M[2, 2] + mf.M[3, 3] == pytest.approx(modello_riferimento.I_yy)


def test_inerzia_rotore_eccessiva_solleva(modello_riferimento, flex_segnaposto) -> None:
    """Un rotore piu' inerte della struttura deve far fallire la costruzione."""
    from dataclasses import replace

    assurdo = replace(flex_segnaposto, J_m_theta=1e3)
    with pytest.raises(ValueError, match="non positiva"):
        fx.costruisci(modello_riferimento, 0.0, flex=assurdo, consenti_segnaposto=True)


def test_modo_rigido_in_azimut(modello_flex) -> None:
    """L'azimut conserva un modo rigido: ``psi`` resta ciclica."""
    frequenze = modello_flex.frequenze_proprie()
    assert np.sum(frequenze < 1e-9) >= 1


def test_frf_rotori_e_strutture_hanno_forma_attesa(modello_flex) -> None:
    """Entrambe le uscite sono matrici 2x2 di FRF."""
    omega = np.logspace(-2, 3, 50) * 2 * np.pi
    assert modello_flex.frf_rotori(omega).shape == (omega.size, 2, 2)
    assert modello_flex.frf_strutture(omega).shape == (omega.size, 2, 2)


def test_equazioni_simboliche_struttura() -> None:
    """La derivazione simbolica produce M, K, C con la struttura della sezione 10."""
    eq = fx.equazioni_simboliche()
    assert eq["M"].shape == (4, 4)
    assert eq["K"].shape == (4, 4)
    # Le coppie agiscono sui rotori, non sulle strutture.
    assert eq["Q"][1] == 0 and eq["Q"][3] == 0
