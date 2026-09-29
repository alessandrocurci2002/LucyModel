"""Preprocessing inerziale: dal CAD alle grandezze del modello (sezione 5).

Il modulo esegue, nell'ordine della specifica:

1. conversione in SI;
2. applicazione della convenzione di segno dei prodotti d'inerzia;
3. controlli fisici nel baricentro di A e B;
4. definizione del punto O;
5. trasporto di B in O (Huygens-Steiner tensoriale) e rotazione in terna B;
6. offset del baricentro di B in terna B;
7. inerzia di A attorno all'asse di azimut;
8. riepilogo su stdout e su ``results/tables/inertia_summary.csv``.

Tutti i tensori sono in **convenzione tensoriale**, cioe'
``I_ij = -int(x_i x_j dm)`` per ``i != j``: con questa convenzione il
trasporto e le rotazioni sono le usuali operazioni tensoriali e l'energia
cinetica e' ``T = 1/2 w^T I w`` senza segni aggiuntivi.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Final

import numpy as np

from . import cad_data
from .params import Avviso, Parametri

# ---------------------------------------------------------------------------
# Operazioni elementari sui tensori
# ---------------------------------------------------------------------------


def normalizza_prodotti(I: np.ndarray, convenzione: str) -> np.ndarray:
    """Porta il tensore in convenzione tensoriale (sezione 5, punto 2).

    Parameters
    ----------
    I
        Tensore 3x3 come visualizzato da Onshape.
    convenzione
        ``"tensor"``: i fuori diagonale sono gia' ``-int(x_i x_j dm)`` e il
        tensore viene restituito invariato.
        ``"positive"``: sono ``+int(x_i x_j dm)`` e vanno negati.
    """
    I = np.asarray(I, dtype=float)
    if convenzione == "tensor":
        return I.copy()
    if convenzione == "positive":
        J = I.copy()
        fuori_diagonale = ~np.eye(3, dtype=bool)
        J[fuori_diagonale] = -J[fuori_diagonale]
        return J
    raise ValueError(f"convenzione '{convenzione}' non riconosciuta")


def trasporta(I_G: np.ndarray, m: float, r: np.ndarray) -> np.ndarray:
    """Huygens-Steiner tensoriale (sezione 5, punto 5).

    .. math:: I_P = I_G + m\\left(\\|r\\|^2 \\mathbb{1} - r r^{\\mathsf T}\\right),
              \\qquad r = r_G - r_P

    Parameters
    ----------
    I_G
        Tensore nel baricentro, in kg*m^2.
    m
        Massa, in kg.
    r
        Vettore dal punto di destinazione P al baricentro G, in m.
    """
    I_G = np.asarray(I_G, dtype=float)
    r = np.asarray(r, dtype=float)
    return I_G + m * (float(r @ r) * np.eye(3) - np.outer(r, r))


def ruota(I: np.ndarray, R: np.ndarray) -> np.ndarray:
    """Cambio di terna di un tensore: ``I|_2 = R^T I|_1 R`` con ``R = R_12``."""
    I = np.asarray(I, dtype=float)
    R = np.asarray(R, dtype=float)
    return R.T @ I @ R


def momenti_principali(I: np.ndarray) -> np.ndarray:
    """Momenti principali d'inerzia, ordinati in modo crescente."""
    return np.sort(np.linalg.eigvalsh(np.asarray(I, dtype=float)))


# ---------------------------------------------------------------------------
# Controlli fisici (sezione 5, punto 3 e sezione 9, test 3)
# ---------------------------------------------------------------------------


class TensoreNonFisico(ValueError):
    """Il tensore d'inerzia non puo' appartenere a un corpo reale."""


def controlli_fisici(
    I: np.ndarray, etichetta: str, *, tol_simmetria: float = 1e-9, tol_triangolo: float = 1e-9
) -> np.ndarray:
    """Verifica simmetria, definita positivita' e disuguaglianza triangolare.

    Un tensore d'inerzia reale deve essere simmetrico, definito positivo e
    i suoi momenti principali devono soddisfare ``I_1 + I_2 >= I_3`` (con
    ``I_3`` il massimo): e' la disuguaglianza triangolare di Hopf, che vale
    perche' ``I_1 + I_2 - I_3 = 2*int(x_3^2 dm) >= 0``.

    Raises
    ------
    TensoreNonFisico
        Se uno dei tre controlli fallisce.

    Returns
    -------
    np.ndarray
        I momenti principali, ordinati in modo crescente.
    """
    I = np.asarray(I, dtype=float)
    if I.shape != (3, 3):
        raise TensoreNonFisico(f"{etichetta}: forma {I.shape}, attesa (3, 3)")

    scala = max(float(np.max(np.abs(I))), 1e-30)
    asimmetria = float(np.max(np.abs(I - I.T))) / scala
    if asimmetria > tol_simmetria:
        raise TensoreNonFisico(
            f"{etichetta}: tensore non simmetrico (asimmetria relativa {asimmetria:.3e})"
        )

    principali = momenti_principali(I)
    if principali[0] <= 0.0:
        raise TensoreNonFisico(
            f"{etichetta}: tensore non definito positivo, momenti principali {principali}"
        )

    residuo = (principali[0] + principali[1] - principali[2]) / principali[2]
    if residuo < -tol_triangolo:
        raise TensoreNonFisico(
            f"{etichetta}: disuguaglianza triangolare violata, "
            f"I1 + I2 - I3 = {principali[0] + principali[1] - principali[2]:.6e} "
            f"(relativo {residuo:.3e}), momenti principali {principali}"
        )
    return principali


# ---------------------------------------------------------------------------
# Componente condiviso tra le due selezioni (sezione 3, avvisi)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class CorpoRidotto:
    """Massa, baricentro e tensore baricentrico di un corpo, in SI."""

    nome: str
    massa: float
    r_G_W: np.ndarray
    I_G_W: np.ndarray


def sottrai_componente_condiviso(
    corpo: CorpoRidotto, condiviso: dict[str, Any], convenzione: str
) -> CorpoRidotto:
    """Rimuove da un corpo un componente contato due volte.

    La sottrazione avviene riportando entrambi i tensori **allo stesso punto**
    prima di sottrarre, come richiesto dalla specifica: si trasportano i due
    tensori baricentrici nell'origine di W, si sottraggono massa, momento di
    primo grado e tensore, e si ricostruisce il baricentro risultante.

    Parameters
    ----------
    corpo
        Corpo da cui sottrarre.
    condiviso
        Dizionario con ``massa_kg``, ``r_G_cm`` e ``I_G_kg_cm2`` del
        componente condiviso (tensore nel proprio baricentro, assi // W).
    convenzione
        Convenzione di segno dei prodotti, applicata anche al componente.

    Raises
    ------
    ValueError
        Se la massa residua non e' positiva.
    """
    m_c = float(condiviso["massa_kg"])
    r_c = np.asarray(condiviso["r_G_cm"], dtype=float) * cad_data.CM_TO_M
    I_c_G = normalizza_prodotti(
        np.asarray(condiviso["I_G_kg_cm2"], dtype=float) * cad_data.KG_CM2_TO_KG_M2,
        convenzione,
    )

    m_res = corpo.massa - m_c
    if m_res <= 0.0:
        raise ValueError(
            f"{corpo.nome}: massa residua {m_res:.6g} kg non positiva dopo la "
            f"sottrazione del componente condiviso ({m_c:.6g} kg)"
        )

    # Tensori riportati nell'origine di W: additivi e quindi sottraibili.
    I_corpo_O = trasporta(corpo.I_G_W, corpo.massa, corpo.r_G_W)
    I_c_O = trasporta(I_c_G, m_c, r_c)
    I_res_O = I_corpo_O - I_c_O

    # Nuovo baricentro dal momento di primo grado.
    r_res = (corpo.massa * corpo.r_G_W - m_c * r_c) / m_res

    # Ritorno al baricentro residuo (trasporto inverso).
    I_res_G = trasporta(I_res_O, -m_res, r_res)

    return CorpoRidotto(nome=corpo.nome, massa=m_res, r_G_W=r_res, I_G_W=I_res_G)


# ---------------------------------------------------------------------------
# Modello inerziale
# ---------------------------------------------------------------------------


UNITA_RIEPILOGO: Final[dict[str, tuple[str, str]]] = {
    "m_A_kg": ("m_A", "kg"),
    "m_B_kg": ("m_B", "kg"),
    "g_m_s2": ("g", "m/s2"),
    "z_O_cm": ("z_O", "cm"),
    "J_A_kg_m2": ("J_A", "kg m2"),
    "I_xx_O_kg_m2": ("I_xx_O", "kg m2"),
    "I_yy_O_kg_m2": ("I_yy_O", "kg m2"),
    "I_zz_O_kg_m2": ("I_zz_O", "kg m2"),
    "I_xy_O_kg_m2": ("I_xy_O", "kg m2"),
    "I_xz_O_kg_m2": ("I_xz_O", "kg m2"),
    "I_yz_O_kg_m2": ("I_yz_O", "kg m2"),
    "d_x_m": ("d_x", "m"),
    "d_y_m": ("d_y", "m"),
    "d_z_m": ("d_z", "m"),
    "J_psi_theta_cad_kg_m2": ("J_psi_theta_cad", "kg m2"),
    "J_psi_0_kg_m2": ("J_psi_0", "kg m2"),
    "theta_min_deg": ("theta_min", "deg"),
    "theta_max_deg": ("theta_max", "deg"),
    "J_psi_theta_min_kg_m2": ("J_psi_theta_min", "kg m2"),
    "J_psi_theta_max_kg_m2": ("J_psi_theta_max", "kg m2"),
    "k_g_0_N_m_rad": ("k_g_0", "N m/rad"),
    "tau_theta_mantenimento_max_N_m": ("tau_theta_mantenimento_max", "N m"),
}
"""Nome pulito e unita' di ciascuna chiave di :meth:`ModelloInerziale.riepilogo`.

Tenuta esplicita, non ricavata dalla chiave: un taglio euristico
sull'ultimo underscore non distingue un nome con underscore interni
(``k_g_0``) da un'unita' su piu' parole (``N_m_rad``).
"""


@dataclass(frozen=True)
class ModelloInerziale:
    """Grandezze inerziali del modello a 2 GdL, in unita' SI.

    Attributes
    ----------
    J_A
        Inerzia del corpo A attorno all'asse di azimut, in kg*m^2.
    I_B_O_B
        Tensore di B rispetto a O, espresso in terna B, in kg*m^2.
    d
        Offset ``(d_x, d_y, d_z)`` del baricentro di B rispetto a O, in terna
        B, in m. ``d_x`` e' lungo il puntamento, ``d_y`` lungo l'asse di
        elevazione, ``d_z`` perpendicolare a entrambi. Solo ``d_x`` e ``d_z``
        pesano sull'energia potenziale, perche' ``y_B`` resta orizzontale:
        ``V = m_B g (-d_x sin(theta) + d_z cos(theta))``.
    m_A, m_B
        Masse dei due corpi, in kg (al netto dell'eventuale componente
        condiviso).
    """

    J_A: float
    I_B_O_B: np.ndarray
    d: np.ndarray
    m_A: float
    m_B: float
    g: float
    r_O_W: np.ndarray
    I_A_G_W: np.ndarray
    I_B_G_W: np.ndarray
    I_B_O_W: np.ndarray
    r_G_A_W: np.ndarray
    r_G_B_W: np.ndarray
    R_WB0: np.ndarray
    par: Parametri
    avvisi: tuple[Avviso, ...] = field(default_factory=tuple)

    # -- componenti del tensore in terna B ----------------------------------

    @property
    def I_xx(self) -> float:
        """Inerzia attorno all'asse di puntamento ``x_B`` (roll)."""
        return float(self.I_B_O_B[0, 0])

    @property
    def I_yy(self) -> float:
        """Inerzia attorno all'asse di elevazione ``y_B``."""
        return float(self.I_B_O_B[1, 1])

    @property
    def I_zz(self) -> float:
        """Inerzia attorno a ``z_B``; a theta = 0 coincide con l'asse verticale."""
        return float(self.I_B_O_B[2, 2])

    @property
    def I_xy(self) -> float:
        return float(self.I_B_O_B[0, 1])

    @property
    def I_xz(self) -> float:
        return float(self.I_B_O_B[0, 2])

    @property
    def I_yz(self) -> float:
        return float(self.I_B_O_B[1, 2])

    @property
    def d_x(self) -> float:
        return float(self.d[0])

    @property
    def d_y(self) -> float:
        return float(self.d[1])

    @property
    def d_z(self) -> float:
        return float(self.d[2])

    @property
    def prodotti_nulli(self) -> bool:
        """``True`` se i prodotti d'inerzia sono numericamente trascurabili."""
        scala = max(self.I_xx, self.I_yy, self.I_zz)
        return bool(max(abs(self.I_xy), abs(self.I_xz), abs(self.I_yz)) < 1e-12 * scala)

    # -- grandezze dipendenti dal punto di lavoro ---------------------------

    def J_psi(self, theta: float | np.ndarray) -> float | np.ndarray:
        """Inerzia efficace di azimut (sezione 6).

        .. math:: J_\\psi(\\theta) = J_A + I_{xx}\\sin^2\\theta
                  + I_{zz}\\cos^2\\theta - I_{xz}\\sin 2\\theta
        """
        s, c = np.sin(theta), np.cos(theta)
        return self.J_A + self.I_xx * s**2 + self.I_zz * c**2 - self.I_xz * np.sin(2 * theta)

    def J_psi_prime(self, theta: float | np.ndarray) -> float | np.ndarray:
        """Derivata ``dJ_psi/dtheta``; con prodotti nulli vale (I_xx-I_zz) sin2theta."""
        return (self.I_xx - self.I_zz) * np.sin(2 * theta) - 2.0 * self.I_xz * np.cos(2 * theta)

    def J_psi_second(self, theta: float | np.ndarray) -> float | np.ndarray:
        """Derivata seconda ``d^2 J_psi/dtheta^2``."""
        return 2.0 * (self.I_xx - self.I_zz) * np.cos(2 * theta) + 4.0 * self.I_xz * np.sin(
            2 * theta
        )

    def M_12(self, theta: float | np.ndarray) -> float | np.ndarray:
        """Termine di accoppiamento inerziale ``M_12 = -I_xy sin(theta) + I_yz cos(theta)``."""
        return -self.I_xy * np.sin(theta) + self.I_yz * np.cos(theta)

    def V_g(self, theta: float | np.ndarray) -> float | np.ndarray:
        """Energia potenziale gravitazionale di B (sezione 6)."""
        return self.m_B * self.g * (-self.d_x * np.sin(theta) + self.d_z * np.cos(theta))

    def dV_dtheta(self, theta: float | np.ndarray) -> float | np.ndarray:
        """Coppia di gravita' ``dV/dtheta``; a regime e' la coppia di mantenimento."""
        return -self.m_B * self.g * (self.d_x * np.cos(theta) + self.d_z * np.sin(theta))

    def k_g(self, theta: float | np.ndarray) -> float | np.ndarray:
        """Rigidezza gravitazionale ``k_g = d^2V/dtheta^2`` (negativa = instabile)."""
        return self.m_B * self.g * (self.d_x * np.sin(theta) - self.d_z * np.cos(theta))

    # -- presentazione -------------------------------------------------------

    def riepilogo(self) -> dict[str, float]:
        """Dizionario piatto delle grandezze principali, per CSV e report.

        Gli estremi di ``theta`` e la coppia di mantenimento massima sono
        valutati sul **campo operativo configurato** (``par.theta0_grid``),
        non su un intervallo fisso: cosi' restano corretti qualunque sia il
        range di pitch confermato in ``config.yaml``.
        """
        theta_grid = self.par.theta0_grid
        theta_min = float(theta_grid.min())
        theta_max = float(theta_grid.max())
        return {
            "m_A_kg": self.m_A,
            "m_B_kg": self.m_B,
            "g_m_s2": self.g,
            "z_O_cm": float(self.r_O_W[2]) * 100.0,
            "J_A_kg_m2": self.J_A,
            "I_xx_O_kg_m2": self.I_xx,
            "I_yy_O_kg_m2": self.I_yy,
            "I_zz_O_kg_m2": self.I_zz,
            "I_xy_O_kg_m2": self.I_xy,
            "I_xz_O_kg_m2": self.I_xz,
            "I_yz_O_kg_m2": self.I_yz,
            "d_x_m": self.d_x,
            "d_y_m": self.d_y,
            "d_z_m": self.d_z,
            "J_psi_theta_cad_kg_m2": float(self.J_psi(self.par.theta_cad)),
            "J_psi_0_kg_m2": float(self.J_psi(0.0)),
            "theta_min_deg": np.rad2deg(theta_min),
            "theta_max_deg": np.rad2deg(theta_max),
            "J_psi_theta_min_kg_m2": float(self.J_psi(theta_min)),
            "J_psi_theta_max_kg_m2": float(self.J_psi(theta_max)),
            "k_g_0_N_m_rad": float(self.k_g(0.0)),
            "tau_theta_mantenimento_max_N_m": float(
                np.max(np.abs(self.dV_dtheta(theta_grid)))
            ),
        }

    def stampa_riepilogo(self) -> None:
        """Riepilogo leggibile su stdout (sezione 5, punto 8)."""
        larghezza = 78
        print("=" * larghezza)
        print("RIEPILOGO INERZIALE (unita' SI)".center(larghezza))
        print("=" * larghezza)
        print(f"  m_A = {self.m_A:10.6f} kg      m_B = {self.m_B:10.6f} kg")
        print(f"  r_O|W = ({self.r_O_W[0]:.5f}, {self.r_O_W[1]:.5f}, {self.r_O_W[2]:.5f}) m")
        print(f"  J_A = {self.J_A:12.6f} kg m^2   (asse di azimut)")
        print("  Tensore di B rispetto a O, in terna B [kg m^2]:")
        for riga in self.I_B_O_B:
            print("    " + "  ".join(f"{v:12.6f}" for v in riga))
        print(
            f"  I_xx = {self.I_xx:.6f} (puntamento)   "
            f"I_yy = {self.I_yy:.6f} (elevazione)   I_zz = {self.I_zz:.6f}"
        )
        print(f"  I_xy = {self.I_xy:.6e}   I_xz = {self.I_xz:.6e}   I_yz = {self.I_yz:.6e}")
        print(
            f"  d = ({self.d_x:+.6f}, {self.d_y:+.6f}, {self.d_z:+.6f}) m "
            "(offset del baricentro di B in terna B)"
        )
        r = self.riepilogo()
        print(f"  J_psi(theta=0)   = {float(self.J_psi(0.0)):.6f} kg m^2")
        print(
            f"  J_psi(theta={r['theta_min_deg']:.0f} deg) = "
            f"{r['J_psi_theta_min_kg_m2']:.6f} kg m^2   "
            f"J_psi(theta={r['theta_max_deg']:.0f} deg) = "
            f"{r['J_psi_theta_max_kg_m2']:.6f} kg m^2"
        )
        print(f"  k_g(theta=0)     = {float(self.k_g(0.0)):+.3f} N m/rad")
        print(
            f"  coppia di mantenimento max su theta in "
            f"[{r['theta_min_deg']:.0f}, {r['theta_max_deg']:.0f}] deg: "
            f"{r['tau_theta_mantenimento_max_N_m']:.3f} N m"
        )
        print("=" * larghezza)

    def salva_riepilogo(self, percorso: str | Path) -> Path:
        """Scrive ``inertia_summary.csv`` (sezione 5, punto 8).

        Nome e unita' di ciascuna grandezza vengono da :data:`UNITA_RIEPILOGO`,
        non da un taglio euristico della chiave: una chiave come
        ``k_g_0_N_m_rad`` non si presta a essere divisa "all'ultimo
        underscore" (l'unita' e' ``N m/rad``, su tre segmenti).
        """
        percorso = Path(percorso)
        percorso.parent.mkdir(parents=True, exist_ok=True)
        righe = self.riepilogo()
        mancanti = set(righe) - set(UNITA_RIEPILOGO)
        if mancanti:
            raise AssertionError(
                f"UNITA_RIEPILOGO non copre le chiavi {sorted(mancanti)}: "
                "aggiornarla insieme a riepilogo()."
            )
        with percorso.open("w", encoding="utf-8", newline="") as f:
            scrittore = csv.writer(f)
            scrittore.writerow(["grandezza", "valore", "unita"])
            for chiave, valore in righe.items():
                nome, unita = UNITA_RIEPILOGO[chiave]
                scrittore.writerow([nome, f"{valore:.10g}", unita])
            for avviso in self.avvisi:
                scrittore.writerow([f"AVVISO:{avviso.codice}", avviso.severita, avviso.messaggio])
        return percorso


# ---------------------------------------------------------------------------
# Costruzione del modello
# ---------------------------------------------------------------------------


def costruisci_modello_inerziale(par: Parametri, *, verifica: bool = True) -> ModelloInerziale:
    """Esegue i punti 1..7 della sezione 5 e restituisce il modello inerziale.

    Parameters
    ----------
    par
        Parametri gia' convertiti in SI.
    verifica
        Se ``True`` esegue i controlli fisici su A e B, nel baricentro e in O.
    """
    avvisi: list[Avviso] = list(par.avvisi)

    # 1-2. Conversione in SI e convenzione dei prodotti.
    corpo_A = CorpoRidotto(
        nome="A",
        massa=cad_data.CORPO_A.massa,
        r_G_W=np.array(cad_data.CORPO_A.r_G_W),
        I_G_W=normalizza_prodotti(
            cad_data.CORPO_A.I_G_W_grezzo, par.onshape_products_sign
        ),
    )
    corpo_B = CorpoRidotto(
        nome="B",
        massa=cad_data.CORPO_B.massa,
        r_G_W=np.array(cad_data.CORPO_B.r_G_W),
        I_G_W=normalizza_prodotti(
            cad_data.CORPO_B.I_G_W_grezzo, par.onshape_products_sign
        ),
    )

    # Doppio conteggio: sottrazione del componente condiviso, se dichiarato.
    if par.shared_component is not None:
        da = str(par.shared_component.get("sottrai_da", "B")).strip().upper()
        if da not in ("A", "B"):
            raise ValueError("shared_component.sottrai_da deve essere 'A' oppure 'B'")
        if da == "A":
            corpo_A = sottrai_componente_condiviso(
                corpo_A, par.shared_component, par.onshape_products_sign
            )
        else:
            corpo_B = sottrai_componente_condiviso(
                corpo_B, par.shared_component, par.onshape_products_sign
            )
        avvisi.append(
            Avviso(
                "DOPPIO_CONTEGGIO_CORRETTO",
                "informazione",
                f"Componente condiviso '{par.shared_component.get('nome', '?')}' "
                f"({float(par.shared_component['massa_kg']):.4f} kg) sottratto dal corpo {da}.",
            )
        )

    # 3. Controlli fisici nel baricentro.
    if verifica:
        controlli_fisici(corpo_A.I_G_W, "I_A,G|W")
        controlli_fisici(corpo_B.I_G_W, "I_B,G|W")

    # 4. Punto O.
    r_O_W = np.array(par.r_O_W, dtype=float)

    # 5. Trasporto di B in O e rotazione in terna B.
    r = corpo_B.r_G_W - r_O_W
    I_B_O_W = trasporta(corpo_B.I_G_W, corpo_B.massa, r)
    if verifica:
        controlli_fisici(I_B_O_W, "I_B,O|W")
    I_B_O_B = ruota(I_B_O_W, par.R_WB0)

    # 6. Offset del baricentro di B in terna B.
    d = par.R_WB0.T @ r

    # 7. Inerzia di A attorno all'asse di azimut.
    #    A ruota solo attorno a Z: gli altri elementi del suo tensore
    #    producono solo carichi sui cuscinetti e non entrano nella dinamica.
    dx_A = corpo_A.r_G_W[0] - r_O_W[0]
    dy_A = corpo_A.r_G_W[1] - r_O_W[1]
    J_A = float(corpo_A.I_G_W[2, 2] + corpo_A.massa * (dx_A**2 + dy_A**2))

    if verifica:
        I_A_O_W = trasporta(corpo_A.I_G_W, corpo_A.massa, corpo_A.r_G_W - r_O_W)
        controlli_fisici(I_A_O_W, "I_A,O|W")

    # Controllo di coerenza della posa CAD: per un telescopio l'asse di
    # puntamento e' l'asse di rotolamento del tubo, quindi deve essere quello
    # con il momento d'inerzia baricentrico **minimo**. Se la posa scelta non
    # lo soddisfa, con ogni probabilita' x_A0_W e' orientato male: la cosa va
    # segnalata, non corretta in silenzio.
    I_B_G_B = ruota(corpo_B.I_G_W, par.R_WB0)
    diagonale = np.diag(I_B_G_B)
    asse_minimo = int(np.argmin(diagonale))
    if asse_minimo != 0:
        nomi = ("x_B (puntamento)", "y_B (elevazione)", "z_B")
        avvisi.append(
            Avviso(
                "POSA_CAD_SOSPETTA",
                "incoerenza",
                f"Con x_A0_W scelto, il momento d'inerzia baricentrico di B "
                f"attorno all'asse di puntamento vale {diagonale[0]:.6f} kg m^2, "
                f"ma il minimo e' {diagonale[asse_minimo]:.6f} kg m^2 attorno a "
                f"{nomi[asse_minimo]}. Per un tubo ottico l'asse di puntamento "
                "e' l'asse di rotolamento e dovrebbe essere quello a inerzia "
                "minima: la posa CAD (x_A0_W, elev_axis_W, theta_cad_deg) e' "
                "probabilmente orientata male. Con questa scelta i valori attesi "
                "della sezione 5 per I_xx,O e I_yy,O risultano scambiati.",
            )
        )

    # Verifica indipendente di z_O: la quota che rende principale il tensore
    # di B rispetto a O e' determinata dai soli dati CAD.
    if par.z_O_misurato:
        z_O_stimato = stima_z_O_da_prodotti(par)
        if z_O_stimato is not None:
            scarto_cm = (float(r_O_W[2]) - z_O_stimato) * 100.0
            if abs(scarto_cm) > 1.0:
                avvisi.append(
                    Avviso(
                        "Z_O_INCOERENTE",
                        "incoerenza",
                        f"z_O_cm = {float(r_O_W[2]) * 100:.4f} cm, ma la quota che "
                        f"rende principale il tensore di B rispetto a O "
                        f"(condizione I_xz,O = 0) vale {z_O_stimato * 100:.4f} cm: "
                        f"scarto {scarto_cm:+.2f} cm. Poiche' il tubo e' un corpo "
                        "di rivoluzione, le due quote dovrebbero coincidere: "
                        "verificare z_O_cm, la posa CAD o i prodotti d'inerzia.",
                    )
                )
            else:
                avvisi.append(
                    Avviso(
                        "Z_O_CONFERMATO",
                        "informazione",
                        f"z_O_cm = {float(r_O_W[2]) * 100:.4f} cm coincide entro "
                        f"{abs(scarto_cm) * 10:.2f} mm con la quota stimata dai soli "
                        f"dati CAD imponendo I_xz,O = 0 ({z_O_stimato * 100:.4f} cm). "
                        "E' una conferma indipendente della quota e della posa CAD.",
                    )
                )

    # Sbilanciamento del telescopio. Conta l'offset del baricentro rispetto a
    # O in **entrambe** le direzioni che pesano sulla gravita': d_x, lungo il
    # puntamento, e d_z, perpendicolare. La grandezza da guardare non e'
    # l'offset in se' ma cio' che produce: coppia di mantenimento e, dove la
    # rigidezza gravitazionale e' negativa, un polo reale instabile.
    d_x, d_z = float(d[0]), float(d[2])
    if max(abs(d_x), abs(d_z)) > 5e-3:
        theta = par.theta0_grid
        theta_min_deg = float(np.rad2deg(theta.min()))
        theta_max_deg = float(np.rad2deg(theta.max()))
        tau = -corpo_B.massa * par.g * (d_x * np.cos(theta) + d_z * np.sin(theta))
        k_theta = corpo_B.massa * par.g * (d_x * np.sin(theta) - d_z * np.cos(theta))
        tau_max = float(np.max(np.abs(tau)))
        I_yy = float(I_B_O_B[1, 1])
        instabile = float(np.max(-k_theta)) > 0.0
        dettaglio = ""
        if instabile:
            polo = float(np.sqrt(np.max(-k_theta) / I_yy))
            theta_peggiore = float(np.rad2deg(theta[int(np.argmin(k_theta))]))
            dettaglio = (
                f" Su parte del campo operativo k_theta < 0: l'asse di elevazione "
                f"e' instabile ad anello aperto, con polo reale massimo "
                f"+{polo:.4f} rad/s ({polo / (2 * np.pi):.4f} Hz, costante di tempo "
                f"{1 / polo:.3f} s) a theta = {theta_peggiore:.0f} deg."
            )
        avvisi.append(
            Avviso(
                "SBILANCIAMENTO",
                "informazione" if not instabile else "incoerenza",
                f"Baricentro di B spostato rispetto a O di d_x = {d_x * 100:+.3f} cm "
                f"(lungo il puntamento) e d_z = {d_z * 100:+.3f} cm (perpendicolare). "
                f"Coppia di mantenimento massima su theta in "
                f"[{theta_min_deg:.0f}, {theta_max_deg:.0f}] deg (campo operativo "
                f"configurato): {tau_max:.3f} N m." + dettaglio,
            )
        )

    return ModelloInerziale(
        J_A=J_A,
        I_B_O_B=I_B_O_B,
        d=np.asarray(d, dtype=float),
        m_A=corpo_A.massa,
        m_B=corpo_B.massa,
        g=par.g,
        r_O_W=r_O_W,
        I_A_G_W=corpo_A.I_G_W,
        I_B_G_W=corpo_B.I_G_W,
        I_B_O_W=I_B_O_W,
        r_G_A_W=corpo_A.r_G_W,
        r_G_B_W=corpo_B.r_G_W,
        R_WB0=np.array(par.R_WB0, dtype=float),
        par=par,
        avvisi=tuple(avvisi),
    )


def I_zz_assieme_asse_azimut(par: Parametri) -> float:
    """Inerzia di A+B attorno all'asse di azimut, dai dati CAD grezzi.

    Calcolo indipendente dal modello: si trasporta il solo termine ZZ di
    ciascun corpo sull'asse verticale di azimut e si sommano. Serve come
    controllo incrociato di ``J_psi(theta_cad)`` (sezione 9, test 5).
    """
    totale = 0.0
    for corpo in (cad_data.CORPO_A, cad_data.CORPO_B):
        I_G = normalizza_prodotti(corpo.I_G_W_grezzo, par.onshape_products_sign)
        dx = corpo.r_G_W[0] - par.r_O_W[0]
        dy = corpo.r_G_W[1] - par.r_O_W[1]
        totale += float(I_G[2, 2] + corpo.massa * (dx**2 + dy**2))
    return totale


def stima_z_O_da_prodotti(par: Parametri) -> float | None:
    r"""Stima la quota dell'asse di elevazione dai soli dati CAD, in m.

    Il tubo ottico e' un corpo di rivoluzione attorno al proprio asse, quindi
    rispetto al punto O il suo tensore dev'essere **principale**: in
    particolare ``I_xz,O = 0``. Questa condizione determina ``z_O`` senza
    bisogno di misurarlo, e fornisce quindi una verifica indipendente del
    valore messo in ``config.yaml``.

    Sviluppando ``I_xz,O = x_B^T I_B,O|_W z_B`` con il trasporto di Steiner, e
    usando ``x_B . z_B = 0``, resta

    .. math:: I_{xz,O} = x_B^{\mathsf T} I_{B,G} z_B - m_B\,d_x\,d_z = 0,

    dove ``d_x`` e ``d_z`` dipendono linearmente da ``z_O``. Ne esce
    un'equazione di secondo grado, che degenera in una lineare quando
    ``theta_cad = 0``.

    Returns
    -------
    float | None
        La quota stimata in m, oppure ``None`` se i dati non la determinano:
        succede quando il baricentro di B e' sull'asse di elevazione
        (``d_x = 0``), perche' allora ``I_xz,O`` non dipende piu' da ``z_O``.
    """
    corpo = cad_data.CORPO_B
    I_G = normalizza_prodotti(corpo.I_G_W_grezzo, par.onshape_products_sign)
    m = corpo.massa

    x_B = par.R_WB0[:, 0]
    z_B = par.R_WB0[:, 2]
    # r0 e' il vettore dal punto O con z_O = 0 al baricentro di B.
    r0 = np.array(corpo.r_G_W) - np.array([par.r_O_W[0], par.r_O_W[1], 0.0])

    c = float(x_B @ I_G @ z_B)
    A = float(x_B @ r0)
    B = float(z_B @ r0)
    sin_cad, cos_cad = np.sin(par.theta_cad), np.cos(par.theta_cad)

    # (m s k) z^2 + m (A k - s B) z + (c - m A B) = 0
    a2 = m * sin_cad * cos_cad
    a1 = m * (A * cos_cad - sin_cad * B)
    a0 = c - m * A * B

    if abs(a2) < 1e-12:
        if abs(a1) < 1e-12:
            return None  # d_x = 0: la condizione non dipende da z_O
        return float(-a0 / a1)

    radici = np.roots([a2, a1, a0])
    reali = [float(np.real(r)) for r in radici if abs(np.imag(r)) < 1e-9]
    if not reali:
        return None
    # Fra le due soluzioni si sceglie quella piu' vicina al baricentro di B:
    # l'altra corrisponde a un punto O assurdo, lontano dal corpo.
    return min(reali, key=lambda z: abs(z - float(corpo.r_G_W[2])))


def modello_con_dz(par: Parametri, d_z: float, *, verifica: bool = True) -> ModelloInerziale:
    """Modello inerziale con lo sbilanciamento ``d_z`` imposto (sezione 8.2).

    Lo sbilanciamento non viene sovrascritto a mano: si sposta il punto O
    lungo la verticale di W finche' ``d_z`` assume il valore richiesto, e si
    rifa' l'intero preprocessing. Cosi' il tensore ``I_B,O`` resta coerente
    con l'offset del baricentro, come impone il teorema di Steiner.

    Raises
    ------
    ValueError
        Se la posa CAD e' tale che ``d_z`` non dipenda da ``z_O``
        (``cos(theta_cad) = 0``, cioe' puntamento verticale nel CAD).
    """
    from dataclasses import replace

    z_B_W = par.R_WB0[:, 2]
    sensibilita = -float(z_B_W[2])  # d(d_z)/d(z_O)
    if abs(sensibilita) < 1e-12:
        raise ValueError(
            "con questa posa CAD d_z non dipende da z_O: usare direttamente z_O_cm"
        )
    r = np.array(cad_data.CORPO_B.r_G_W) - np.array(par.r_O_W)
    d_z_corrente = float(z_B_W @ r)
    delta_z_O = (d_z - d_z_corrente) / sensibilita

    r_O_nuovo = np.array(par.r_O_W, dtype=float) + np.array([0.0, 0.0, delta_z_O])
    par_nuovo = replace(par, r_O_W=r_O_nuovo)
    return costruisci_modello_inerziale(par_nuovo, verifica=verifica)


def tabella_momenti_principali(par: Parametri) -> dict[str, np.ndarray]:
    """Momenti principali dei tensori baricentrici, in kg*cm^2 (controllo CAD)."""
    risultato: dict[str, np.ndarray] = {}
    for corpo in (cad_data.CORPO_A, cad_data.CORPO_B):
        I = normalizza_prodotti(corpo.I_G_W_grezzo, par.onshape_products_sign)
        risultato[corpo.nome] = momenti_principali(I) / cad_data.KG_CM2_TO_KG_M2
    return risultato
