"""Fixture condivise dalla suite di test.

I test verificano il **codice**, non le scelte fatte in ``config.yaml``: i
valori attesi della sezione 5 della specifica valgono infatti solo nella
configurazione di riferimento (i default della sezione 4). Per questo la
fixture principale e' :func:`par_riferimento`, costruita da
:func:`gimbal.params.parametri_di_riferimento`, mentre la configurazione
effettivamente usata per l'analisi e' esposta a parte da :func:`par_attivo`.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

RADICE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RADICE / "src"))

from gimbal import inertia as inz  # noqa: E402
from gimbal import lagrange as lg  # noqa: E402
from gimbal import linearize as lz  # noqa: E402
from gimbal import params as pr  # noqa: E402


@pytest.fixture(scope="session")
def par_riferimento() -> pr.Parametri:
    """Configurazione di riferimento: i default della sezione 4 della specifica."""
    return pr.parametri_di_riferimento()


@pytest.fixture(scope="session")
def modello_riferimento(par_riferimento: pr.Parametri) -> inz.ModelloInerziale:
    """Modello inerziale nella configurazione di riferimento."""
    return inz.costruisci_modello_inerziale(par_riferimento)


@pytest.fixture(scope="session")
def par_attivo() -> pr.Parametri:
    """Configurazione effettiva del progetto, letta da ``config.yaml``."""
    return pr.parametri_da_config(RADICE / "config.yaml")


@pytest.fixture(scope="session")
def modello_attivo(par_attivo: pr.Parametri) -> inz.ModelloInerziale:
    """Modello inerziale nella configurazione effettiva."""
    return inz.costruisci_modello_inerziale(par_attivo)


@pytest.fixture(scope="session")
def simbolico() -> lg.ModelloSimbolico:
    """Modello simbolico. La derivazione con SymPy e' memoizzata."""
    return lg.costruisci_modello_simbolico()


@pytest.fixture(scope="session")
def lineare_simbolico() -> lz.ModelloLineareSimbolico:
    """Modello lineare simbolico."""
    return lz.costruisci_modello_lineare_simbolico()
