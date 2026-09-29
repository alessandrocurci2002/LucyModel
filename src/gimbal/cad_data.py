"""Dati grezzi del CAD Onshape (sezione 3 della specifica). Immutabili.

Questo modulo non esegue alcun calcolo e non applica alcuna conversione: e'
la sola fonte dei dati di massa e inerzia, riportati esattamente come
mostrati dallo strumento *Mass properties* di Onshape.

Contesto della misura, da cui dipende l'interpretazione dei numeri:

* nessun *mate connector* e' stato selezionato come riferimento, quindi le
  inerzie sono calcolate **rispetto al baricentro delle parti selezionate**,
  con **assi paralleli alla terna W** (comportamento documentato da Onshape);
* l'origine di W coincide con il **centro della base dell'albero azimutale**:
  l'asse di azimut passa quindi per l'origine per costruzione, e le quote z
  si leggono come altezze sopra la base dell'albero. Nella posa del CAD il
  telescopio punta in direzione e verso **-Y**;
* unita' dei dati grezzi: cm per le lunghezze, kg per le masse,
  kg*cm^2 per i momenti d'inerzia;
* la convenzione di segno dei prodotti d'inerzia non e' nota a priori: e'
  governata dal parametro ``onshape_products_sign`` di ``config.yaml`` e
  applicata in :mod:`gimbal.inertia`, non qui.

I due corpi del modello:

* **corpo A**: forcella e albero di azimut, ruota solo in psi;
* **corpo B**: telescopio e tutte le parti che ruotano in elevazione.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Final

import numpy as np

# ---------------------------------------------------------------------------
# Fattori di conversione verso il SI (sezione 5, punto 1)
# ---------------------------------------------------------------------------

CM_TO_M: Final[float] = 1e-2
"""Da cm a m."""

KG_CM2_TO_KG_M2: Final[float] = 1e-4
"""Da kg*cm^2 a kg*m^2."""


def _immutabile(a: list[float] | list[list[float]]) -> np.ndarray:
    """Restituisce un array numpy in sola lettura, per impedire modifiche."""
    arr = np.array(a, dtype=float)
    arr.setflags(write=False)
    return arr


@dataclass(frozen=True)
class CorpoCAD:
    """Proprieta' di massa di un corpo, come lette da Onshape.

    Parameters
    ----------
    nome
        Etichetta del corpo ("A" o "B").
    descrizione
        Descrizione in italiano delle parti incluse.
    massa_kg
        Massa in kg.
    r_G_cm
        Baricentro in terna W, in cm.
    I_G_kg_cm2
        Tensore d'inerzia nel baricentro, assi paralleli a W, in kg*cm^2,
        **come visualizzato da Onshape** (segno dei prodotti non ancora
        normalizzato).
    istanze
        Istanze selezionate in Onshape. La lista e' troncata negli
        screenshot disponibili: serve solo a documentare la selezione e a
        individuare i componenti condivisi tra i due corpi.
    """

    nome: str
    descrizione: str
    massa_kg: float
    r_G_cm: np.ndarray
    I_G_kg_cm2: np.ndarray
    istanze: tuple[str, ...] = field(default_factory=tuple)

    @property
    def massa(self) -> float:
        """Massa in kg (il SI coincide con il dato grezzo)."""
        return self.massa_kg

    @property
    def r_G_W(self) -> np.ndarray:
        """Baricentro in terna W, in m."""
        return self.r_G_cm * CM_TO_M

    @property
    def I_G_W_grezzo(self) -> np.ndarray:
        """Tensore nel baricentro, assi // W, in kg*m^2, segni non normalizzati.

        Il nome porta il suffisso ``_grezzo`` perche' la convenzione di segno
        dei prodotti d'inerzia non e' ancora stata applicata: quello lo fa
        :func:`gimbal.inertia.normalizza_prodotti`.
        """
        return self.I_G_kg_cm2 * KG_CM2_TO_KG_M2


# ---------------------------------------------------------------------------
# Corpo A: forcella e albero di azimut (ruota solo in psi)
# ---------------------------------------------------------------------------

CORPO_A: Final[CorpoCAD] = CorpoCAD(
    nome="A",
    descrizione="Forcella e albero di azimut (ruota solo in psi)",
    massa_kg=22.535,
    r_G_cm=_immutabile([-0.240, 0.001, 41.944]),
    I_G_kg_cm2=_immutabile(
        [
            [6602.470, -0.172, 6.604],
            [-0.172, 15363.593, 0.031],
            [6.604, 0.031, 10431.336],
        ]
    ),
    istanze=(
        "B-10 Albero di trasmissione <1>",
        "B-04 Sede anello interno <1>",
        "Part 1 <2>",
        "INNENRING <1>",
        "...(lista troncata nello screenshot)",
    ),
)

# ---------------------------------------------------------------------------
# Corpo B: telescopio e parti che ruotano in elevazione
# ---------------------------------------------------------------------------

CORPO_B: Final[CorpoCAD] = CorpoCAD(
    nome="B",
    descrizione="Telescopio e parti che ruotano in elevazione",
    massa_kg=24.057,
    r_G_cm=_immutabile([1.369, -7.965, 54.460]),
    I_G_kg_cm2=_immutabile(
        [
            [8932.534, -263.414, -10.194],
            [-263.414, 5624.207, 86.696],
            [-10.194, 86.696, 11222.053],
        ]
    ),
    istanze=(
        "Telescopio LAMERA <1>",
        "T-02 Anello di registrazione del telescopio <1>",
        "I04 - Albero condotto <1>",
        "PH Countersunk flat head screw M3x0.5 x 12 <2>",
        "...(lista troncata nello screenshot)",
    ),
)

# ---------------------------------------------------------------------------
# Componenti presenti in entrambe le selezioni: rischio di doppio conteggio
# ---------------------------------------------------------------------------

ISTANZE_CONDIVISE: Final[tuple[str, ...]] = tuple(
    sorted((set(CORPO_A.istanze) & set(CORPO_B.istanze)) - {"...(lista troncata nello screenshot)"})
)
"""Istanze che compaiono in entrambe le selezioni di Onshape.

Con la revisione dei dati del 2026-09-27 questo insieme e' **vuoto**: nelle
liste visibili non c'e' alcun componente condiviso. Le liste degli screenshot
sono pero' troncate, quindi la sovrapposizione non e' esclusa e l'avviso sul
doppio conteggio resta. Vedi ``shared_component`` in ``config.yaml`` e
:func:`gimbal.inertia.sottrai_componente_condiviso`.
"""

# ---------------------------------------------------------------------------
# Valori di riferimento della sezione 5 della specifica
# ---------------------------------------------------------------------------

VALORI_RIFERIMENTO: Final[dict[str, float]] = {
    "J_A": 1.043263,
    "I_B_G_ZZ_piu_steiner_azimut": 1.279335,
    "J_psi_theta_cad_0": 2.322598,
    "I_xx_O_dz0": 0.566929,
    "I_yy_O_dz0": 1.045874,
    "I_zz_O_dz0": 1.279335,
}
"""Valori attesi (SI) della tabella della sezione 5, usati nei test.

Si riferiscono alla **configurazione di riferimento** della specifica, cioe'
ai default della sezione 4: ``x_A0_W = "-Y"``, ``elev_axis_W = "X"``,
``theta_cad_deg = 0``, ``z_O_cm = null`` (quindi ``d_z = 0``).
``J_A`` e ``J_psi(theta_cad = 0)`` sono invece indipendenti da ``z_O`` e
dalla convenzione dei prodotti.

Con questa revisione dei dati vale ``I_xx,O < I_yy,O < I_zz,O``: l'asse di
puntamento e' quello a inerzia minima, come dev'essere per un tubo ottico.
Di conseguenza ``J_psi`` **decresce** dall'orizzonte allo zenit.
"""

Z_O_STIMATO_CM: Final[float] = 54.9125
"""Quota dell'asse di elevazione stimata dai soli dati CAD, in cm.

Il tubo e' un corpo di rivoluzione attorno al proprio asse, quindi rispetto
al punto O il suo tensore dev'essere principale: imponendo ``I_xz,O = 0`` si
ricava ``z_O`` senza misurarlo. Vedi
:func:`gimbal.inertia.stima_z_O_da_prodotti`.
"""

AUTOVALORI_RIFERIMENTO: Final[dict[str, tuple[float, float, float]]] = {
    "A": (6602.46, 10431.35, 15363.59),
    "B": (5602.06, 8953.25, 11223.48),
}
"""Autovalori attesi dei tensori baricentrici come visualizzati, in kg*cm^2."""
