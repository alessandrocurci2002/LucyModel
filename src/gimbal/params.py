"""Lettura di ``config.yaml``, conversione in SI e costruzione delle terne.

Implementa la sezione 4 della specifica. Il modulo e' l'unico punto in cui
entrano i parametri configurabili: tutto il resto del pacchetto riceve un
oggetto :class:`Parametri` gia' in unita' SI, con le terne della posa CAD
gia' costruite e verificate.

Ogni parametro non misurato produce un :class:`Avviso`, che viene propagato
fino a ``report.md`` e stampato a video: la specifica richiede che nessun
risultato dipendente da un segnaposto sia presentato senza avvertenza.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Final, Literal, Sequence

import numpy as np
import yaml

from . import cad_data

# ---------------------------------------------------------------------------
# Avvisi
# ---------------------------------------------------------------------------

Severita = Literal["segnaposto", "incoerenza", "informazione"]


@dataclass(frozen=True)
class Avviso:
    """Avvertenza sui dati di ingresso, da riportare in ogni output.

    Parameters
    ----------
    codice
        Identificativo breve e stabile, citabile nel report.
    severita
        ``"segnaposto"`` per un valore non misurato, ``"incoerenza"`` per un
        dato in conflitto con altri dati, ``"informazione"`` per una scelta
        consapevole da documentare.
    messaggio
        Testo in italiano, autosufficiente.
    """

    codice: str
    severita: Severita
    messaggio: str

    def __str__(self) -> str:  # pragma: no cover - solo presentazione
        etichetta = {
            "segnaposto": "SEGNAPOSTO",
            "incoerenza": "INCOERENZA",
            "informazione": "NOTA",
        }[self.severita]
        return f"[{etichetta}: {self.codice}] {self.messaggio}"


# ---------------------------------------------------------------------------
# Direzioni cardinali della terna W
# ---------------------------------------------------------------------------

_DIREZIONI: Final[dict[str, tuple[float, float, float]]] = {
    "X": (1.0, 0.0, 0.0),
    "+X": (1.0, 0.0, 0.0),
    "-X": (-1.0, 0.0, 0.0),
    "Y": (0.0, 1.0, 0.0),
    "+Y": (0.0, 1.0, 0.0),
    "-Y": (0.0, -1.0, 0.0),
    "Z": (0.0, 0.0, 1.0),
    "+Z": (0.0, 0.0, 1.0),
    "-Z": (0.0, 0.0, -1.0),
}

Z_W: Final[np.ndarray] = np.array([0.0, 0.0, 1.0])
"""Verticale della terna W, verso l'alto. Coincide con l'asse di azimut."""


def direzione(nome: str) -> np.ndarray:
    """Converte ``"X"``, ``"-Y"``, ... nel corrispondente versore di W."""
    chiave = str(nome).strip().upper()
    if chiave not in _DIREZIONI:
        raise ValueError(
            f"direzione '{nome}' non riconosciuta; ammesse: {sorted(_DIREZIONI)}"
        )
    return np.array(_DIREZIONI[chiave], dtype=float)


def R_y(theta: float) -> np.ndarray:
    """Rotazione elementare attorno a y, nella convenzione della sezione 2.

    .. math:: R_y(\\theta) = \\begin{bmatrix} c & 0 & s \\\\ 0 & 1 & 0 \\\\
              -s & 0 & c\\end{bmatrix}

    Attenzione al segno: e' la convenzione **vincolante** della specifica,
    per cui ``theta > 0`` abbassa il puntamento.
    """
    c, s = np.cos(theta), np.sin(theta)
    return np.array([[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]])


def R_z(psi: float) -> np.ndarray:
    """Rotazione elementare attorno a z (azimut), regola della mano destra."""
    c, s = np.cos(psi), np.sin(psi)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def R_NB(psi: float, theta: float) -> np.ndarray:
    """Assetto del telescopio: ``R_NB = R_z(psi) R_y(theta)`` (sezione 2)."""
    return R_z(psi) @ R_y(theta)


# ---------------------------------------------------------------------------
# Parametri
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ParametriFlex:
    """Parametri dell'estensione flessibile a 4 GdL (sezione 10).

    Tutti facoltativi: finche' uno di essi e' ``None``,
    :mod:`gimbal.flex4dof` rifiuta di produrre risultati numerici.
    """

    J_m_psi: float | None = None
    J_m_theta: float | None = None
    k_psi: float | None = None
    k_theta: float | None = None
    c_psi: float | None = None
    c_theta: float | None = None

    @property
    def completo(self) -> bool:
        """``True`` se tutti i parametri sono stati forniti."""
        return all(
            getattr(self, nome) is not None
            for nome in ("J_m_psi", "J_m_theta", "k_psi", "k_theta", "c_psi", "c_theta")
        )

    def mancanti(self) -> tuple[str, ...]:
        """Nomi dei parametri ancora non forniti."""
        return tuple(
            nome
            for nome in ("J_m_psi", "J_m_theta", "k_psi", "k_theta", "c_psi", "c_theta")
            if getattr(self, nome) is None
        )


@dataclass(frozen=True)
class Parametri:
    """Parametri del modello, in unita' SI, con le terne della posa CAD.

    Attributes
    ----------
    r_O_W
        Posizione del punto O (intersezione degli assi) in W, in m.
    z_O_misurato
        ``False`` se ``z_O_cm`` era ``null`` e si e' ricaduti su ``z_G_B``.
    R_WA0, R_WB0
        Assetto di forcella e telescopio nella posa CAD.
    theta_cad
        Elevazione theta nella posa CAD, in rad.
    omega0
        Velocita' di inseguimento in azimut, in rad/s.
    theta0_grid
        Griglia dei punti di lavoro in elevazione, in rad.
    dz_sensitivity
        Sbilanciamenti d_z per la sensibilita', in m.
    avvisi
        Avvertenze raccolte durante la lettura della configurazione.
    """

    # geometria e posa
    r_O_W: np.ndarray
    z_O_misurato: bool
    x_A0_W: np.ndarray
    y_A0_W: np.ndarray
    elev_axis_W: np.ndarray
    R_WA0: np.ndarray
    R_WB0: np.ndarray
    theta_cad: float

    # convenzioni
    onshape_products_sign: str

    # fisica
    g: float
    b_psi: float
    b_theta: float
    b_sensitivity: tuple[float, ...]

    # griglie di analisi
    omega0: tuple[float, ...]
    theta0_grid: np.ndarray
    theta0_bode: tuple[float, ...]
    dz_sensitivity: tuple[float, ...]

    # estensione flessibile
    flex_model: bool
    flex: ParametriFlex

    # doppio conteggio
    shared_component: dict[str, Any] | None

    # tracciabilita'
    sorgente: Path | None
    avvisi: tuple[Avviso, ...] = field(default_factory=tuple)

    # -- proprieta' derivate ------------------------------------------------

    @property
    def omega0_max(self) -> float:
        """Velocita' di inseguimento massima, in rad/s."""
        return max(self.omega0)

    @property
    def b(self) -> np.ndarray:
        """Matrice di smorzamento viscoso ``B = diag(b_psi, b_theta)``."""
        return np.diag([self.b_psi, self.b_theta])

    @property
    def configurazione_di_riferimento(self) -> bool:
        """``True`` se la posa coincide con i default della sezione 4.

        I valori attesi della tabella della sezione 5 valgono solo in questa
        configurazione: ``x_A0_W = "-Y"``, ``theta_cad = 0`` e ``d_z = 0``.
        """
        return (
            bool(np.allclose(self.x_A0_W, [0.0, -1.0, 0.0]))
            and abs(self.theta_cad) < 1e-12
            and not self.z_O_misurato
        )

    def avvisi_per_severita(self, severita: Severita) -> tuple[Avviso, ...]:
        """Sottoinsieme degli avvisi con la severita' richiesta."""
        return tuple(a for a in self.avvisi if a.severita == severita)

    def stampa_avvisi(self) -> None:
        """Stampa gli avvisi in modo visibile, come richiesto dalla sezione 1."""
        if not self.avvisi:
            return
        larghezza = 78
        print("=" * larghezza)
        print("AVVISI SUI DATI DI INGRESSO".center(larghezza))
        print("=" * larghezza)
        for avviso in self.avvisi:
            print(f"  {avviso}")
        print("=" * larghezza)


# ---------------------------------------------------------------------------
# Lettura della configurazione
# ---------------------------------------------------------------------------


def _griglia(spec: Any, nome: str) -> np.ndarray:
    """Interpreta una griglia data come lista oppure come start/stop/step."""
    if isinstance(spec, dict):
        mancanti = {"start", "stop", "step"} - set(spec)
        if mancanti:
            raise ValueError(f"{nome}: chiavi mancanti {sorted(mancanti)}")
        start, stop, step = float(spec["start"]), float(spec["stop"]), float(spec["step"])
        if step == 0.0:
            raise ValueError(f"{nome}: 'step' non puo' essere nullo")
        n = int(round(abs(stop - start) / abs(step))) + 1
        return np.linspace(start, stop, n)
    if isinstance(spec, Sequence):
        return np.array([float(v) for v in spec])
    raise ValueError(f"{nome}: formato non riconosciuto ({type(spec).__name__})")


def carica_config(percorso: str | Path) -> dict[str, Any]:
    """Legge il file YAML e restituisce il dizionario grezzo."""
    percorso = Path(percorso)
    with percorso.open("r", encoding="utf-8") as f:
        dati = yaml.safe_load(f)
    if not isinstance(dati, dict):
        raise ValueError(f"{percorso}: il file non contiene una mappa YAML")
    return dati


def costruisci_terne(
    x_A0_W: np.ndarray, elev_axis_W: np.ndarray, theta_cad: float
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Costruisce le terne della posa CAD (sezione 4 della specifica).

    ``y_A0 = Z_W x x_A0``; si verifica con un assert che ``y_A0`` sia
    parallelo a ``elev_axis_W``; poi
    ``R_WA0 = [x_A0, y_A0, Z_W]`` (vettori come colonne) e
    ``R_WB0 = R_WA0 R_y(theta_cad)``.

    Returns
    -------
    (y_A0_W, R_WA0, R_WB0)
    """
    x_A0_W = np.asarray(x_A0_W, dtype=float)
    x_A0_W = x_A0_W / np.linalg.norm(x_A0_W)
    if abs(float(x_A0_W @ Z_W)) > 1e-12:
        raise ValueError(
            "x_A0_W deve essere orizzontale: e' la proiezione orizzontale del puntamento"
        )

    y_A0_W = np.cross(Z_W, x_A0_W)
    y_A0_W = y_A0_W / np.linalg.norm(y_A0_W)

    # Vincolo della sezione 4: y_A0 e' l'asse di elevazione.
    parallelo = np.linalg.norm(np.cross(y_A0_W, elev_axis_W)) < 1e-9
    assert parallelo, (
        f"y_A0 = {y_A0_W} non e' parallelo a elev_axis_W = {elev_axis_W}: "
        "x_A0_W ed elev_axis_W sono incoerenti (deve valere y_A0 = Z_W x x_A0)"
    )

    R_WA0 = np.column_stack([x_A0_W, y_A0_W, Z_W])
    R_WB0 = R_WA0 @ R_y(theta_cad)

    # Controlli di ortonormalita': una svista qui invaliderebbe tutti i tensori.
    assert np.allclose(R_WA0.T @ R_WA0, np.eye(3), atol=1e-12), "R_WA0 non ortonormale"
    assert np.isclose(np.linalg.det(R_WA0), 1.0, atol=1e-12), "R_WA0 non e' una rotazione"
    assert np.allclose(R_WB0.T @ R_WB0, np.eye(3), atol=1e-12), "R_WB0 non ortonormale"

    return y_A0_W, R_WA0, R_WB0


def parametri_da_config(
    percorso: str | Path | None = None, *, dati: dict[str, Any] | None = None
) -> Parametri:
    """Costruisce :class:`Parametri` da ``config.yaml`` (o da un dizionario).

    Parameters
    ----------
    percorso
        File YAML da leggere. Se ``None`` e ``dati`` e' ``None``, si usa
        ``config.yaml`` nella radice del progetto.
    dati
        Dizionario gia' caricato, utile nei test per costruire
        configurazioni alternative senza scrivere file.
    """
    if dati is None:
        if percorso is None:
            percorso = Path(__file__).resolve().parents[2] / "config.yaml"
        dati = carica_config(percorso)
    else:
        percorso = Path(percorso) if percorso is not None else None

    avvisi: list[Avviso] = []

    # --- asse di azimut nel piano XY ---------------------------------------
    az_xy_cm = np.array([float(v) for v in dati.get("az_axis_xy_cm", [0.0, 0.0])])
    if az_xy_cm.shape != (2,):
        raise ValueError("az_axis_xy_cm deve avere due componenti [x, y] in cm")
    if np.allclose(az_xy_cm, 0.0):
        avvisi.append(
            Avviso(
                "AZ_AXIS_XY",
                "informazione",
                "az_axis_xy_cm = [0, 0]: l'origine di W e' il centro della base "
                "dell'albero azimutale, quindi l'asse di azimut passa per "
                "l'origine per costruzione. Resta implicita l'ipotesi che "
                "l'asse sia verticale, cioe' parallelo a Z_W.",
            )
        )
    else:
        avvisi.append(
            Avviso(
                "AZ_AXIS_XY_SPOSTATO",
                "informazione",
                f"az_axis_xy_cm = {az_xy_cm.tolist()} cm: l'asse di azimut non "
                "passa per l'origine di W, che pero' e' definita come il centro "
                "della base dell'albero azimutale. Verificare quale delle due "
                "informazioni sia corretta.",
            )
        )

    # --- quota del punto O --------------------------------------------------
    z_O_cm_raw = dati.get("z_O_cm", None)
    z_G_B_cm = float(cad_data.CORPO_B.r_G_cm[2])
    z_O_misurato = z_O_cm_raw is not None
    if z_O_misurato:
        z_O_cm = float(z_O_cm_raw)
        # La verifica di coerenza di z_O non si fa qui con un'euristica sulle
        # distanze, ma in gimbal.inertia, confrontando il valore con la quota
        # stimata dai prodotti d'inerzia (condizione I_xz,O = 0): serve la posa
        # CAD, che a questo punto non e' ancora stata costruita.
        pass
    else:
        z_O_cm = z_G_B_cm
        avvisi.append(
            Avviso(
                "Z_O_NON_MISURATO",
                "segnaposto",
                f"z_O_cm = null: si usa z_O = z_G_B = {z_G_B_cm:g} cm, cioe' si "
                "assume il telescopio bilanciato (d_z = 0). Da misurare nel CAD.",
            )
        )

    r_O_W = np.array([az_xy_cm[0], az_xy_cm[1], z_O_cm]) * cad_data.CM_TO_M

    # --- posa CAD ----------------------------------------------------------
    x_A0_W = direzione(dati.get("x_A0_W", "-Y"))
    elev_axis_W = direzione(dati.get("elev_axis_W", "X"))
    theta_cad = float(np.deg2rad(float(dati.get("theta_cad_deg", 0.0))))
    y_A0_W, R_WA0, R_WB0 = costruisci_terne(x_A0_W, elev_axis_W, theta_cad)

    if not np.allclose(x_A0_W, [0.0, -1.0, 0.0]):
        avvisi.append(
            Avviso(
                "X_A0_NON_DEFAULT",
                "informazione",
                f"x_A0_W = {dati.get('x_A0_W')} (default della specifica: '-Y'). "
                "La posa CAD determina quale componente del tensore di B diventa "
                "l'inerzia di puntamento e quale quella di elevazione: i valori "
                "attesi della sezione 5 valgono solo con il default.",
            )
        )
    if abs(theta_cad) > 1e-12:
        avvisi.append(
            Avviso(
                "THETA_CAD_NON_NULLO",
                "informazione",
                f"theta_cad = {np.rad2deg(theta_cad):g} deg: R_WB0 ruota il "
                "tensore di B rispetto alla posa orizzontale.",
            )
        )

    # --- convenzione dei prodotti d'inerzia --------------------------------
    segno = str(dati.get("onshape_products_sign", "tensor")).strip().lower()
    if segno not in ("tensor", "positive"):
        raise ValueError(
            f"onshape_products_sign = '{segno}': ammessi 'tensor' e 'positive'"
        )
    avvisi.append(
        Avviso(
            "ONSHAPE_SEGNO_PRODOTTI",
            "segnaposto",
            f"onshape_products_sign = '{segno}': la convenzione di segno dei "
            "prodotti d'inerzia di Onshape non e' stata verificata con un corpo "
            "di prova. Influenza I_xy, I_xz, I_yz e quindi M_12 e J_psi(theta), "
            "non i momenti diagonali.",
        )
    )

    # --- fisica -------------------------------------------------------------
    g = float(dati.get("g", 9.80665))
    b_psi = float(dati.get("b_psi", 0.0))
    b_theta = float(dati.get("b_theta", 0.0))
    if b_psi == 0.0 and b_theta == 0.0:
        avvisi.append(
            Avviso(
                "SMORZAMENTI_NULLI",
                "segnaposto",
                "b_psi = b_theta = 0: nessuno smorzamento identificato "
                "sperimentalmente. Gli smorzamenti modali riportati derivano "
                "solo dalla dinamica rigida, non da misure.",
            )
        )
    b_sensitivity = tuple(float(v) for v in dati.get("b_sensitivity", (0.0,)))

    # --- griglie ------------------------------------------------------------
    omega0_deg_s = tuple(float(v) for v in dati.get("omega0_deg_s", (0.0, 1.0, 5.0, 10.0)))
    omega0 = tuple(float(np.deg2rad(v)) for v in omega0_deg_s)
    avvisi.append(
        Avviso(
            "OMEGA0_SEGNAPOSTO",
            "segnaposto",
            f"omega0_deg_s = {list(omega0_deg_s)} deg/s: la velocita' di "
            "inseguimento massima reale non e' nota. Ogni risultato che dipende "
            "da Omega_0 (accoppiamento giroscopico gamma, rigidezza centrifuga "
            "k_c) e' condizionato da questo segnaposto.",
        )
    )

    theta0_grid = np.deg2rad(
        _griglia(dati.get("theta0_grid_deg", {"start": -45.0, "stop": 45.0, "step": 1.0}),
                 "theta0_grid_deg")
    )
    theta0_bode = tuple(
        float(np.deg2rad(v))
        for v in dati.get("theta0_bode_deg", (-45.0, -15.0, 15.0, 45.0))
    )
    dz_sensitivity = tuple(
        float(v) * cad_data.CM_TO_M
        for v in dati.get("dz_sensitivity_cm", (-2.0, -1.0, -0.5, 0.0, 0.5, 1.0, 2.0))
    )

    # --- estensione flessibile ---------------------------------------------
    flex_raw = dati.get("flex", {}) or {}
    flex = ParametriFlex(
        J_m_psi=_opt_float(flex_raw.get("J_m_psi")),
        J_m_theta=_opt_float(flex_raw.get("J_m_theta")),
        k_psi=_opt_float(flex_raw.get("k_psi")),
        k_theta=_opt_float(flex_raw.get("k_theta")),
        c_psi=_opt_float(flex_raw.get("c_psi")),
        c_theta=_opt_float(flex_raw.get("c_theta")),
    )
    flex_model = bool(dati.get("flex_model", False))
    if flex_model and not flex.completo:
        avvisi.append(
            Avviso(
                "FLEX_PARAMETRI_MANCANTI",
                "incoerenza",
                f"flex_model = true ma mancano {list(flex.mancanti())}: "
                "flex4dof rifiutera' di produrre risultati numerici.",
            )
        )

    # --- componente condiviso ----------------------------------------------
    shared = dati.get("shared_component", None)
    if shared is None:
        if cad_data.ISTANZE_CONDIVISE:
            avvisi.append(
                Avviso(
                    "DOPPIO_CONTEGGIO",
                    "incoerenza",
                    "shared_component = null: "
                    f"{', '.join(cad_data.ISTANZE_CONDIVISE)} compare in entrambe "
                    "le selezioni di Onshape ed e' quindi contato due volte in "
                    "massa e inerzia. Compilare shared_component in config.yaml "
                    "per sottrarlo da uno dei due corpi.",
                )
            )
        else:
            avvisi.append(
                Avviso(
                    "DOPPIO_CONTEGGIO_NON_ESCLUSO",
                    "informazione",
                    "Nelle liste di istanze visibili non compare alcun componente "
                    "condiviso fra le due selezioni di Onshape, ma le liste degli "
                    "screenshot sono troncate: la sovrapposizione non e' esclusa. "
                    "Se ne emergesse una, compilare shared_component in "
                    "config.yaml per sottrarre il componente da uno dei due corpi.",
                )
            )

    return Parametri(
        r_O_W=r_O_W,
        z_O_misurato=z_O_misurato,
        x_A0_W=x_A0_W,
        y_A0_W=y_A0_W,
        elev_axis_W=elev_axis_W,
        R_WA0=R_WA0,
        R_WB0=R_WB0,
        theta_cad=theta_cad,
        onshape_products_sign=segno,
        g=g,
        b_psi=b_psi,
        b_theta=b_theta,
        b_sensitivity=b_sensitivity,
        omega0=omega0,
        theta0_grid=theta0_grid,
        theta0_bode=theta0_bode,
        dz_sensitivity=dz_sensitivity,
        flex_model=flex_model,
        flex=flex,
        shared_component=shared,
        sorgente=Path(percorso) if percorso is not None else None,
        avvisi=tuple(avvisi),
    )


def _opt_float(v: Any) -> float | None:
    """Converte in float, lasciando passare ``None``."""
    return None if v is None else float(v)


def parametri_di_riferimento(**override: Any) -> Parametri:
    """Configurazione di riferimento della specifica (default della sezione 4).

    Serve ai test: i valori attesi della tabella della sezione 5 valgono solo
    con ``x_A0_W = "-Y"``, ``elev_axis_W = "X"``, ``theta_cad_deg = 0`` e
    ``z_O_cm = null`` (telescopio bilanciato, ``d_z = 0``).
    """
    dati: dict[str, Any] = {
        "az_axis_xy_cm": [0.0, 0.0],
        "z_O_cm": None,
        "elev_axis_W": "X",
        "x_A0_W": "-Y",
        "theta_cad_deg": 0.0,
        "onshape_products_sign": "tensor",
        "g": 9.80665,
        "b_psi": 0.0,
        "b_theta": 0.0,
        "omega0_deg_s": [0.0, 1.0, 5.0, 10.0],
        "theta0_grid_deg": {"start": -45.0, "stop": 45.0, "step": 1.0},
        "dz_sensitivity_cm": [-2.0, -1.0, -0.5, 0.0, 0.5, 1.0, 2.0],
        "flex_model": False,
        "shared_component": None,
    }
    dati.update(override)
    return parametri_da_config(dati=dati)
