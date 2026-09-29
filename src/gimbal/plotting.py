"""Figure del progetto (sezioni 8.2 e 8.3), in stile uniforme.

Ogni figura viene salvata sia in **PDF vettoriale** (per l'inclusione in tesi)
sia in **PNG a 300 dpi** (per le anteprime), con nomi di file descrittivi.
Etichette in italiano con unita' di misura; gradi e Hz nelle presentazioni,
rad e rad/s solo internamente.

Convenzioni grafiche
--------------------
* La palette categorica e' fissa e ordinata, mai ciclata: blu, arancio,
  acqua, giallo. E' stata verificata per separazione sotto deficit di visione
  dei colori (protanopia e deuteranopia) sulle coppie adiacenti.
* Poiche' due dei quattro colori non raggiungono il contrasto 3:1 su fondo
  bianco, il colore non e' mai l'unica codifica: ogni serie ha anche uno stile
  di tratto diverso ed e' presente in legenda. Le stesse curve sono inoltre
  disponibili come CSV in ``results/tables``.
* Griglia e assi sono recessivi; il testo usa i toni dell'inchiostro, non il
  colore della serie.
* Nessun grafico a doppio asse y.
* ``theta`` e' la coordinata del modello; l'elevazione fisica ``epsilon = -theta``
  compare solo come informazione aggiuntiva nelle etichette.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import matplotlib

matplotlib.use("Agg")  # nessuna finestra interattiva: si scrive solo su file

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.figure import Figure
from matplotlib.gridspec import GridSpec

from .analysis import Sweep
from .linearize import ETICHETTE_FDT, ModelloLineare
from .simulate import ConvergenzaLineare, RisultatoScenario

# ---------------------------------------------------------------------------
# Stile
# ---------------------------------------------------------------------------

SERIE: tuple[str, ...] = ("#2a78d6", "#eb6834", "#1baf7a", "#eda100")
"""Palette categorica, in ordine fisso. Verificata per CVD sulle coppie adiacenti."""

TRATTI: tuple[str, ...] = ("-", "--", "-.", ":")
"""Stili di tratto, abbinati alla palette: codifica secondaria obbligatoria."""

INCHIOSTRO: str = "#0b0b0b"
INCHIOSTRO_SECONDARIO: str = "#52514e"
INCHIOSTRO_TENUE: str = "#8a8880"
GRIGLIA: str = "#e3e2de"

DPI_PNG: int = 300


def applica_stile() -> None:
    """Imposta lo stile uniforme di tutte le figure."""
    plt.rcParams.update(
        {
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "savefig.facecolor": "white",
            "font.size": 9.0,
            "axes.titlesize": 9.5,
            "axes.labelsize": 9.0,
            "legend.fontsize": 8.0,
            "xtick.labelsize": 8.0,
            "ytick.labelsize": 8.0,
            "axes.edgecolor": INCHIOSTRO_TENUE,
            "axes.labelcolor": INCHIOSTRO,
            "text.color": INCHIOSTRO,
            "xtick.color": INCHIOSTRO_SECONDARIO,
            "ytick.color": INCHIOSTRO_SECONDARIO,
            "axes.linewidth": 0.7,
            "axes.grid": True,
            "grid.color": GRIGLIA,
            "grid.linewidth": 0.6,
            "grid.alpha": 1.0,
            "lines.linewidth": 1.6,
            "lines.markersize": 4.0,
            "legend.frameon": True,
            "legend.framealpha": 0.92,
            "legend.edgecolor": GRIGLIA,
            "axes.axisbelow": True,
            "figure.autolayout": False,
            "pdf.fonttype": 42,  # font incorporati come TrueType: testo selezionabile
        }
    )


def salva(fig: Figure, cartella: str | Path, nome: str) -> list[Path]:
    """Salva una figura in PDF vettoriale e PNG a 300 dpi."""
    cartella = Path(cartella)
    cartella.mkdir(parents=True, exist_ok=True)
    percorsi: list[Path] = []
    for estensione, kwargs in (("pdf", {}), ("png", {"dpi": DPI_PNG})):
        percorso = cartella / f"{nome}.{estensione}"
        fig.savefig(percorso, bbox_inches="tight", **kwargs)
        percorsi.append(percorso)
    plt.close(fig)
    return percorsi


def _stile_serie(i: int) -> dict[str, object]:
    """Colore e tratto della i-esima serie, in ordine fisso e non ciclato."""
    if i >= len(SERIE):
        raise ValueError(
            f"richieste {i + 1} serie ma la palette ne definisce {len(SERIE)}: "
            "raggruppare le serie in eccesso oppure usare piu' pannelli"
        )
    return {"color": SERIE[i], "linestyle": TRATTI[i]}


def _nota(fig: Figure, testo: str) -> None:
    """Nota a pie' di figura, in tono tenue."""
    fig.text(
        0.005, -0.015, testo, fontsize=7.0, color=INCHIOSTRO_SECONDARIO,
        ha="left", va="top", wrap=True,
    )


# ---------------------------------------------------------------------------
# Diagrammi di Bode (sezione 8.3)
# ---------------------------------------------------------------------------


def _frequenze_hz(n: int = 2000) -> np.ndarray:
    """2000 punti logaritmici fra 1e-3 e 1e2 Hz (sezione 8.3)."""
    return np.logspace(-3.0, 2.0, n)


def bode_2x2(
    modelli: Sequence[ModelloLineare],
    *,
    etichette: Sequence[str],
    titolo: str,
    segna_poli_zeri: bool = True,
    nota: str | None = None,
) -> Figure:
    """Griglia 2x2 delle quattro FdT, con modulo in dB e fase in gradi.

    Ogni cella contiene due assi sovrapposti: il modulo, in dB rispetto a
    1 rad/(N m), e la fase in gradi con ``unwrap``. Poli e zeri sono segnati
    come linee verticali alla frequenza ``|lambda| / 2 pi``.

    Parameters
    ----------
    modelli
        Modelli linearizzati da sovrapporre (al massimo 4).
    etichette
        Etichetta di legenda per ciascun modello.
    """
    f = _frequenze_hz()
    omega = 2.0 * np.pi * f

    fig = plt.figure(figsize=(9.6, 7.4))
    griglia = GridSpec(2, 2, figure=fig, hspace=0.34, wspace=0.24)

    for cella, (indice, descrizione) in enumerate(ETICHETTE_FDT.items()):
        i, j = indice
        interna = griglia[cella // 2, cella % 2].subgridspec(
            2, 1, height_ratios=[2.0, 1.2], hspace=0.08
        )
        ax_mod = fig.add_subplot(interna[0])
        ax_fase = fig.add_subplot(interna[1], sharex=ax_mod)

        nulla_per_tutti = True
        for k, ml in enumerate(modelli):
            G = ml.frf(omega)[:, i, j]
            if np.max(np.abs(G)) <= 0.0:
                continue
            nulla_per_tutti = False
            stile = _stile_serie(k)
            modulo_db = 20.0 * np.log10(np.abs(G))
            fase_deg = np.rad2deg(np.unwrap(np.angle(G)))
            ax_mod.semilogx(f, modulo_db, label=etichette[k], **stile)
            ax_fase.semilogx(f, fase_deg, **stile)

            if segna_poli_zeri:
                _segna_radici(ax_mod, ml, indice, colore=SERIE[k])

        if nulla_per_tutti:
            ax_mod.text(
                0.5, 0.5,
                "identicamente nulla\nin questi punti di lavoro",
                transform=ax_mod.transAxes, ha="center", va="center",
                fontsize=8.5, color=INCHIOSTRO_SECONDARIO,
            )

        ax_mod.set_title(descrizione, loc="left", color=INCHIOSTRO)
        ax_mod.set_ylabel(r"modulo [dB rif. 1 rad/(N$\cdot$m)]")
        ax_mod.tick_params(labelbottom=False)
        ax_fase.set_ylabel("fase [deg]")
        ax_fase.set_xlabel("frequenza [Hz]")
        ax_fase.set_xlim(f[0], f[-1])
        _assi_di_fase(ax_fase)
        if cella == 0 and len(modelli) >= 2:
            ax_mod.legend(loc="lower left", ncol=1)

    fig.suptitle(titolo, y=0.985, fontsize=11.0, color=INCHIOSTRO)
    testo = (
        "Linee verticali tratteggiate: poli; punteggiate: zeri, alla frequenza "
        "|lambda|/2pi. Il modello e' a corpi rigidi: nessuna di queste frequenze "
        "e' un modo strutturale."
    )
    _nota(fig, nota or testo)
    return fig


def _assi_di_fase(ax: plt.Axes) -> None:
    """Limiti e tacche della fase a multipli di 90 deg.

    Senza smorzamento e con ``Omega_0 = 0`` la FRF e' puramente reale, quindi
    la fase e' costante a 0 o +-180 deg: l'autoscala produrrebbe un asse di
    ampiezza 0.1 deg, che sembra un difetto del grafico invece che il
    risultato corretto. Si impone allora una scala minima di mezzo giro.
    """
    basso, alto = ax.get_ylim()
    centro = 0.5 * (basso + alto)
    meta = max(0.5 * (alto - basso), 90.0)
    basso, alto = centro - meta, centro + meta
    ax.set_ylim(np.floor(basso / 90.0) * 90.0, np.ceil(alto / 90.0) * 90.0)
    ax.yaxis.set_major_locator(matplotlib.ticker.MultipleLocator(90.0))


def _segna_radici(
    ax: plt.Axes, ml: ModelloLineare, indice: tuple[int, int], *, colore: str
) -> None:
    """Segna poli e zeri della FdT come linee verticali sulla frequenza."""
    fdt = ml.fdt.get(indice)
    if fdt is None or fdt.nulla:
        return
    for radici, stile in ((fdt.poli, (0, (5, 3))), (fdt.zeri, (0, (1, 2)))):
        for r in np.atleast_1d(radici):
            f_r = float(abs(r)) / (2.0 * np.pi)
            if 1e-3 <= f_r <= 1e2:
                ax.axvline(f_r, color=colore, linestyle=stile, linewidth=0.8, alpha=0.55)


# ---------------------------------------------------------------------------
# Analisi parametriche (sezione 8.2)
# ---------------------------------------------------------------------------


def frequenze_vs_parametro(
    sweeps: Sequence[Sweep],
    *,
    titolo: str,
    nota: str | None = None,
) -> Figure:
    """Frequenze di poli e zeri in funzione del parametro di scansione.

    Una curva per ogni valore di ``Omega_0``. Tre pannelli su un unico asse y
    in Hz: poli, zeri di ``G_11``, zeri di ``G_22``. Le radici reali sono
    riportate come ``|lambda|/2pi``: il pannello dei poli distingue con un
    marcatore le radici reali positive, che sono instabilita' statiche e non
    oscillazioni.
    """
    fig, assi = plt.subplots(3, 1, figsize=(7.2, 8.4), sharex=True)
    pannelli = (
        ("poli (autovalori di A)", "poli"),
        (r"zeri di $G_{11}$ (radici di $P_{22}$)", "zeri_G11"),
        (r"zeri di $G_{22}$ (radici di $P_{11}$)", "zeri_G22"),
    )

    for ax, (titolo_pannello, attributo) in zip(assi, pannelli):
        tracciato = False
        for k, sw in enumerate(sweeps):
            stile = _stile_serie(k)
            radici = getattr(sw, attributo)
            frequenze = np.abs(radici) / (2.0 * np.pi)
            etichetta = rf"$\Omega_0$ = {np.rad2deg(sw.omega0):g} deg/s"
            primo_tracciato = True
            for colonna in range(frequenze.shape[1]):
                y = frequenze[:, colonna]
                # Le radici nell'origine non hanno una frequenza da mostrare in
                # scala logaritmica: vengono dichiarate a parte, non omesse in
                # silenzio.
                if np.all(~np.isfinite(y)) or np.allclose(np.nan_to_num(y), 0.0):
                    continue
                ax.plot(
                    sw.ascissa, y,
                    label=etichetta if primo_tracciato else None,
                    **stile,
                )
                tracciato = True
                primo_tracciato = False
            if attributo == "poli":
                instabili = np.real(radici) > 1e-9
                if np.any(instabili):
                    ax.plot(
                        np.repeat(sw.ascissa[:, None], radici.shape[1], axis=1)[instabili],
                        frequenze[instabili],
                        marker="o", linestyle="none", markersize=3.0,
                        markerfacecolor="none", markeredgewidth=0.9,
                        color=SERIE[k],
                        label="radice reale positiva (instabile)" if k == 0 else None,
                    )
        ax.set_ylabel("frequenza [Hz]")
        ax.set_title(titolo_pannello, loc="left")
        if tracciato:
            ax.set_yscale("log")
        else:
            ax.text(
                0.5, 0.5,
                "tutte le radici sono nell'origine\n"
                "(nessuna frequenza da riportare)",
                transform=ax.transAxes, ha="center", va="center",
                fontsize=8.5, color=INCHIOSTRO_SECONDARIO,
            )
            ax.set_yticks([])

    assi[-1].set_xlabel(sweeps[0].etichetta_ascissa)
    assi[0].legend(loc="best")
    fig.suptitle(titolo, y=0.995, fontsize=11.0, color=INCHIOSTRO)
    fig.tight_layout(rect=(0, 0, 1, 0.98))
    _nota(
        fig,
        nota
        or "Modello a corpi rigidi: le frequenze mostrate hanno origine "
        "gravitazionale, giroscopica o centrifuga, non strutturale.",
    )
    return fig


def J_psi_vs_theta(
    theta_grid: np.ndarray, J: np.ndarray, *, titolo: str | None = None
) -> Figure:
    """``J_psi(theta)`` sul campo operativo di pitch (sezione 8.2).

    Serie unica: nessuna legenda, il titolo nomina la grandezza. In ascissa
    theta, con l'elevazione fisica ``epsilon = -theta`` indicata nell'etichetta.
    L'estensione dell'asse la determina ``theta_grid``, cioe' il campo
    operativo configurato: non e' piu' fissa a [-90, 0] deg.
    """
    fig, ax = plt.subplots(figsize=(6.4, 3.8))
    theta_deg = np.rad2deg(np.asarray(theta_grid, dtype=float))
    ax.plot(theta_deg, J, color=SERIE[0], linestyle=TRATTI[0])

    # Etichetta diretta agli estremi: nessun numero su ogni punto.
    for x, y, allineamento in (
        (theta_deg[0], J[0], "left"),
        (theta_deg[-1], J[-1], "right"),
    ):
        ax.annotate(
            f"{y:.3f}", xy=(x, y), xytext=(0, 7), textcoords="offset points",
            ha=allineamento, fontsize=8.0, color=INCHIOSTRO_SECONDARIO,
        )

    ax.set_xlabel(r"$\theta$ [deg]   ($\varepsilon = -\theta$: 0 = orizzonte, 90 = zenit)")
    ax.set_ylabel(r"$J_\psi$ [kg$\cdot$m$^2$]")
    ax.set_title(
        titolo or r"Inerzia efficace di azimut $J_\psi(\theta)$", loc="left"
    )
    fig.tight_layout()
    return fig


def sensibilita_dz(sw: Sweep, *, titolo: str | None = None) -> Figure:
    """Rigidezza equivalente e coppia di mantenimento in funzione di ``d_z``.

    Due pannelli separati, non un doppio asse y: le grandezze hanno unita'
    diverse (N m/rad e N m).
    """
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(6.4, 5.6), sharex=True)
    ax1.plot(sw.ascissa, sw.k_theta, color=SERIE[0], linestyle=TRATTI[0])
    ax1.axhline(0.0, color=INCHIOSTRO_TENUE, linewidth=0.7)
    ax1.set_ylabel(r"$k_\theta$ [N$\cdot$m/rad]")
    ax1.set_title(
        r"Rigidezza equivalente dell'asse di elevazione ($k_\theta < 0$: instabile)",
        loc="left",
    )

    ax2.plot(sw.ascissa, sw.tau_theta0, color=SERIE[1], linestyle=TRATTI[1])
    ax2.axhline(0.0, color=INCHIOSTRO_TENUE, linewidth=0.7)
    ax2.set_ylabel(r"$\tau_{\theta 0}$ [N$\cdot$m]")
    ax2.set_xlabel(sw.etichetta_ascissa)
    ax2.set_title("Coppia di mantenimento", loc="left")

    fig.suptitle(
        titolo
        or rf"Sensibilit\`a allo sbilanciamento, $\theta_0$ fissato, "
        rf"$\Omega_0$ = {np.rad2deg(sw.omega0):g} deg/s",
        y=0.995, fontsize=11.0, color=INCHIOSTRO,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    return fig


# ---------------------------------------------------------------------------
# Storie temporali (sezione 8.4)
# ---------------------------------------------------------------------------


def storie_temporali(ris: RisultatoScenario) -> Figure:
    """Storie temporali di uno scenario, con sovrapposizione lineare/non lineare.

    Tre pannelli: perturbazione di ``psi``, perturbazione di ``theta``, coppie.
    Dove previsto, la risposta del modello lineare e' sovrapposta a quella non
    lineare; la legenda e' sempre presente quando ci sono due serie.
    """
    dq_nl = ris.dq_nonlineare
    fig, assi = plt.subplots(3, 1, figsize=(7.2, 7.6), sharex=True)

    for ax, colonna, simbolo in (
        (assi[0], 0, r"\delta\psi"),
        (assi[1], 1, r"\delta\theta"),
    ):
        ax.plot(
            ris.t, np.rad2deg(dq_nl[:, colonna]),
            label="non lineare", color=SERIE[0], linestyle=TRATTI[0],
        )
        if ris.dq_lin is not None:
            ax.plot(
                ris.t, np.rad2deg(ris.dq_lin[:, colonna]),
                label="lineare", color=SERIE[1], linestyle=TRATTI[1],
            )
        ax.set_ylabel(rf"${simbolo}$ [deg]")
        if ris.dq_lin is not None:
            ax.legend(loc="best")

    assi[0].set_title(
        r"Perturbazione di azimut $\delta\psi = \psi - \psi_0(t)$", loc="left"
    )
    assi[1].set_title(r"Perturbazione di elevazione $\delta\theta$", loc="left")

    assi[2].plot(ris.t, ris.tau[:, 0], label=r"$\tau_\psi$", color=SERIE[0], linestyle=TRATTI[0])
    assi[2].plot(ris.t, ris.tau[:, 1], label=r"$\tau_\theta$", color=SERIE[1], linestyle=TRATTI[1])
    assi[2].set_ylabel(r"coppia [N$\cdot$m]")
    assi[2].set_xlabel("tempo [s]")
    assi[2].set_title(
        "Coppie applicate (feedforward nominale + perturbazione)", loc="left"
    )
    assi[2].legend(loc="best")

    fig.suptitle(f"{ris.nome} - {ris.descrizione}", y=0.995, fontsize=10.5, color=INCHIOSTRO)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    if ris.errori:
        _nota(
            fig,
            "Errore relativo lineare/non lineare: "
            + ", ".join(f"{k} {v * 100:.3f} %" for k, v in ris.errori.items()
                        if k in ("psi", "theta")),
        )
    return fig


def conservazione(ris: RisultatoScenario, E: np.ndarray, p: np.ndarray) -> Figure:
    """Deriva relativa di energia e momento nello scenario conservativo S4."""
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(6.8, 5.2), sharex=True)
    for ax, valori, nome, unita, k in (
        (ax1, E, r"$E = T + V$", "J", 0),
        (ax2, p, r"$p_\psi = \partial\mathcal{L}/\partial\dot\psi$", r"N$\cdot$m$\cdot$s", 1),
    ):
        riferimento = valori[0] if abs(valori[0]) > 0 else 1.0
        ax.plot(
            ris.t, (valori - valori[0]) / abs(riferimento),
            color=SERIE[k], linestyle=TRATTI[k],
        )
        ax.set_ylabel(f"deriva relativa di {nome}")
        ax.set_title(
            f"{nome} = {valori[0]:.6g} {unita} all'istante iniziale", loc="left"
        )
    ax2.set_xlabel("tempo [s]")
    fig.suptitle(
        "S4 - verifica conservativa: b = 0, tau = 0", y=0.995, fontsize=10.5,
        color=INCHIOSTRO,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    _nota(
        fig,
        "La specifica richiede una deriva relativa sotto 1e-8: e' il controllo "
        "piu' severo sulla correttezza delle equazioni di Lagrange.",
    )
    return fig


def convergenza_linearizzazione(convs: Sequence[ConvergenzaLineare]) -> Figure:
    """Errore relativo lineare/non lineare in funzione dell'ampiezza.

    In scala doppio-logaritmica una linearizzazione corretta da' pendenza 1.
    La retta di riferimento a pendenza 1 e' tracciata in tono tenue.
    """
    fig, assi = plt.subplots(1, len(convs), figsize=(4.6 * len(convs), 4.0), squeeze=False)
    for ax, conv in zip(assi[0], convs):
        for k, (errori, nome) in enumerate(
            ((conv.errore_guidata, "coordinata guidata"),
             (conv.errore_incrociata, "coordinata incrociata")),
        ):
            stile = _stile_serie(k)
            ax.loglog(conv.bersagli_deg, errori, marker="o", **stile, label=nome)
        # Retta di riferimento a pendenza 1.
        x = np.asarray(conv.bersagli_deg, dtype=float)
        y0 = float(conv.errore_incrociata[-1])
        ax.loglog(
            x, y0 * x / x[-1], color=INCHIOSTRO_TENUE, linewidth=0.8,
            linestyle=(0, (4, 3)), label="pendenza 1 (riferimento)",
        )
        ax.axhline(0.01, color=INCHIOSTRO_TENUE, linewidth=0.7)
        ax.annotate(
            "soglia 1 %", xy=(x[0], 0.01), xytext=(2, 4), textcoords="offset points",
            fontsize=7.5, color=INCHIOSTRO_SECONDARIO,
        )
        ax.set_xlabel("ampiezza della perturbazione [deg]")
        ax.set_ylabel("errore relativo lineare/non lineare")
        ax.set_title(
            f"{conv.scenario}: pendenza {conv.pendenza_guidata:.2f} (guidata), "
            f"{conv.pendenza_incrociata:.2f} (incrociata)",
            loc="left",
        )
        ax.legend(loc="best")
    fig.tight_layout()
    _nota(
        fig,
        "L'errore relativo di una linearizzazione corretta e' proporzionale "
        "all'ampiezza della perturbazione: pendenza 1 in scala logaritmica.",
    )
    return fig


# ---------------------------------------------------------------------------
# Catalogo delle figure prodotte
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Figura:
    """Voce del catalogo delle figure, per l'elenco nel report."""

    nome: str
    descrizione: str
    percorsi: tuple[Path, ...]


def registra(
    catalogo: list[Figura], fig: Figure, cartella: str | Path, nome: str, descrizione: str
) -> Figura:
    """Salva una figura e la registra nel catalogo."""
    percorsi = salva(fig, cartella, nome)
    voce = Figura(nome=nome, descrizione=descrizione, percorsi=tuple(percorsi))
    catalogo.append(voce)
    return voce
