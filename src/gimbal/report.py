"""Generazione di ``results/report.md`` (sezione 11 della specifica).

Il report raccoglie, nell'ordine richiesto:

* la **nota di realismo** della sezione 0, in testa e non in fondo;
* i parametri usati, con i segnaposto evidenziati;
* le incoerenze rilevate nei dati;
* il riepilogo inerziale;
* l'esito dei controlli sui valori di riferimento della sezione 5;
* le tabelle di poli e zeri, con le cancellazioni documentate;
* l'esito delle simulazioni;
* l'elenco delle figure e dei file prodotti.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np

from .analysis import classifica
from .inertia import ModelloInerziale
from .linearize import ModelloLineare
from .params import Avviso, Parametri
from .plotting import Figura
from .simulate import ConvergenzaLineare, RisultatoScenario

NOTA_REALISMO: str = """\
> **Che cosa questo modello puo' e non puo' dire.**
>
> Questo e' un modello a **due corpi rigidi**. Non contiene alcuna
> flessibilita', quindi **non contiene modi strutturali**: nessuna delle
> frequenze riportate in questo documento e' una frequenza propria della
> struttura, e nessuna va citata come tale.
>
> Le uniche radici che il modello puo' produrre sono:
>
> 1. i **poli nulli** del moto rigido, dovuti al fatto che `psi` e' una
>    coordinata ciclica (con `b_psi = 0` l'asse di azimut e' un doppio
>    integratore e il polo nullo e' doppio);
> 2. una **dinamica a bassa frequenza**, tipicamente sotto 1 Hz, dovuta a
>    gravita' (sbilanciamento del telescopio), accoppiamento giroscopico e
>    rigidezza centrifuga.
>
> Le risonanze e antirisonanze strutturali che interessano il controllo di
> puntamento compaiono solo introducendo la cedevolezza dei giunti: e' quel
> che fa l'estensione a 4 GdL della sezione 10, che pero' richiede rigidezze
> e inerzie dei rotori misurate, non stimate.
"""


@dataclass
class ContenutoReport:
    """Tutto cio' che il report deve pubblicare."""

    par: Parametri
    modello: ModelloInerziale
    controlli_riferimento: Mapping[str, tuple[float, float, bool]]
    punti_di_lavoro: Sequence[ModelloLineare]
    scenari: Mapping[str, RisultatoScenario]
    convergenze: Sequence[ConvergenzaLineare] = ()
    figure: Sequence[Figura] = ()
    tabelle: Sequence[Path] = ()
    latex: Sequence[Path] = ()
    coerenza: Mapping[str, float] = field(default_factory=dict)
    flex_attivo: bool = False
    flex_messaggio: str = ""
    comando: str = "python run_all.py"


def _si_no(valore: bool) -> str:
    return "OK" if valore else "**FALLITO**"


def _cella(valore: object) -> str:
    """Contenuto di una cella, con le barre verticali neutralizzate.

    Una barra non protetta spezzerebbe la riga della tabella: capita per
    esempio nella descrizione dello scenario S1, che contiene ``|delta psi|``.
    """
    return str(valore).replace("|", r"\|")


def _tabella(intestazione: Sequence[str], righe: Sequence[Sequence[object]]) -> list[str]:
    """Tabella Markdown."""
    out = ["| " + " | ".join(_cella(c) for c in intestazione) + " |"]
    out.append("|" + "|".join("---" for _ in intestazione) + "|")
    for riga in righe:
        out.append("| " + " | ".join(_cella(c) for c in riga) + " |")
    out.append("")
    return out


def _sezione_parametri(par: Parametri) -> list[str]:
    """Parametri usati, con i segnaposto evidenziati."""
    segnaposto = {a.codice for a in par.avvisi_per_severita("segnaposto")}

    def marca(condizione: bool) -> str:
        return " **[SEGNAPOSTO]**" if condizione else ""

    righe: list[Sequence[object]] = [
        (
            "`az_axis_xy_cm`",
            f"[{par.r_O_W[0] * 100:.4g}, {par.r_O_W[1] * 100:.4g}]",
            "cm",
            "posizione dell'asse di azimut nel piano XY di W"
            + marca("AZ_AXIS_XY" in segnaposto),
        ),
        (
            "`z_O_cm`",
            f"{par.r_O_W[2] * 100:.4g}",
            "cm",
            ("quota del punto O, **misurata**" if par.z_O_misurato
             else "quota del punto O, ricavata da z_G_B")
            + marca("Z_O_NON_MISURATO" in segnaposto),
        ),
        ("`x_A0_W`", f"{par.x_A0_W.tolist()}", "-", "direzione orizzontale di x_A nella posa CAD"),
        ("`elev_axis_W`", f"{par.elev_axis_W.tolist()}", "-", "direzione dell'asse di elevazione"),
        ("`theta_cad_deg`", f"{np.rad2deg(par.theta_cad):.4g}", "deg", "elevazione nella posa CAD"),
        (
            "`onshape_products_sign`",
            f"`{par.onshape_products_sign}`",
            "-",
            "convenzione di segno dei prodotti d'inerzia"
            + marca("ONSHAPE_SEGNO_PRODOTTI" in segnaposto),
        ),
        ("`g`", f"{par.g:.6g}", "m/s^2", "accelerazione di gravita'"),
        (
            "`b_psi`", f"{par.b_psi:.6g}", "N m s/rad",
            "smorzamento viscoso in azimut" + marca("SMORZAMENTI_NULLI" in segnaposto),
        ),
        (
            "`b_theta`", f"{par.b_theta:.6g}", "N m s/rad",
            "smorzamento viscoso in elevazione" + marca("SMORZAMENTI_NULLI" in segnaposto),
        ),
        (
            "`omega0_deg_s`",
            f"{[round(float(np.rad2deg(o)), 4) for o in par.omega0]}",
            "deg/s",
            "velocita' di inseguimento in azimut" + marca("OMEGA0_SEGNAPOSTO" in segnaposto),
        ),
        (
            "`theta0_grid_deg`",
            f"da {np.rad2deg(par.theta0_grid[0]):.4g} a "
            f"{np.rad2deg(par.theta0_grid[-1]):.4g}, {par.theta0_grid.size} punti",
            "deg",
            "griglia dei punti di lavoro",
        ),
        (
            "`dz_sensitivity_cm`",
            f"{[round(v * 100, 4) for v in par.dz_sensitivity]}",
            "cm",
            "sbilanciamenti per l'analisi di sensibilita'",
        ),
        ("`flex_model`", f"{par.flex_model}", "-", "estensione flessibile a 4 GdL"),
    ]

    out = ["## 2. Parametri usati", ""]
    if par.sorgente is not None:
        out += [f"Sorgente: `{par.sorgente}`.", ""]
    out += _tabella(("parametro", "valore", "unita", "significato"), righe)
    out += [
        "I parametri marcati **[SEGNAPOSTO]** non provengono da una misura. "
        "Ogni risultato che ne dipende e' condizionato a essi e non va "
        "presentato come proprieta' della montatura reale.",
        "",
    ]
    return out


def _sezione_avvisi(avvisi: Sequence[Avviso]) -> list[str]:
    out = ["## 3. Avvisi e incoerenze rilevate", ""]
    incoerenze = [a for a in avvisi if a.severita == "incoerenza"]
    if incoerenze:
        out += [
            "### 3.1 Incoerenze (da risolvere prima di usare i numeri in tesi)",
            "",
        ]
        for a in incoerenze:
            out += [f"- **`{a.codice}`** - {a.messaggio}", ""]
    altri = [a for a in avvisi if a.severita != "incoerenza"]
    if altri:
        out += ["### 3.2 Segnaposto e note", ""]
        for a in altri:
            etichetta = "SEGNAPOSTO" if a.severita == "segnaposto" else "NOTA"
            out += [f"- *{etichetta}* `{a.codice}` - {a.messaggio}", ""]
    if not avvisi:
        out += ["Nessun avviso.", ""]
    return out


def _sezione_inerzia(modello: ModelloInerziale) -> list[str]:
    r = modello.riepilogo()
    righe = [
        ("m_A", f"{modello.m_A:.6f}", "kg", "massa della forcella e dell'albero di azimut"),
        ("m_B", f"{modello.m_B:.6f}", "kg", "massa del telescopio"),
        ("J_A", f"{modello.J_A:.6f}", "kg m^2", "inerzia di A attorno all'asse di azimut"),
        ("I_xx", f"{modello.I_xx:.6f}", "kg m^2", "asse di puntamento x_B"),
        ("I_yy", f"{modello.I_yy:.6f}", "kg m^2", "asse di elevazione y_B"),
        ("I_zz", f"{modello.I_zz:.6f}", "kg m^2", "asse z_B"),
        ("I_xy", f"{modello.I_xy:+.6e}", "kg m^2", "prodotto d'inerzia"),
        ("I_xz", f"{modello.I_xz:+.6e}", "kg m^2", "prodotto d'inerzia"),
        ("I_yz", f"{modello.I_yz:+.6e}", "kg m^2", "prodotto d'inerzia"),
        ("d_x", f"{modello.d_x:+.6f}", "m",
         "sbilanciamento lungo il puntamento: pesa su V come -m_B g d_x sin(theta)"),
        ("d_y", f"{modello.d_y:+.6f}", "m",
         "offset lungo l'asse di elevazione: non pesa su V, perche' y_B e' sempre orizzontale"),
        ("d_z", f"{modello.d_z:+.6f}", "m",
         "sbilanciamento perpendicolare: pesa su V come +m_B g d_z cos(theta)"),
        ("J_psi(0)", f"{r['J_psi_0_kg_m2']:.6f}", "kg m^2", "inerzia di azimut all'orizzonte"),
        (
            f"J_psi({r['theta_min_deg']:.0f} deg)",
            f"{r['J_psi_theta_min_kg_m2']:.6f}", "kg m^2",
            "inerzia di azimut all'estremo inferiore del campo operativo",
        ),
        (
            f"J_psi({r['theta_max_deg']:.0f} deg)",
            f"{r['J_psi_theta_max_kg_m2']:.6f}", "kg m^2",
            "inerzia di azimut all'estremo superiore del campo operativo",
        ),
        ("k_g(0)", f"{r['k_g_0_N_m_rad']:+.4f}", "N m/rad", "rigidezza gravitazionale (negativa = instabile)"),
        (
            "max |tau_theta0|",
            f"{r['tau_theta_mantenimento_max_N_m']:.4f}",
            "N m",
            f"coppia di mantenimento massima su theta in "
            f"[{r['theta_min_deg']:.0f}, {r['theta_max_deg']:.0f}] deg "
            "(campo operativo configurato)",
        ),
    ]
    return ["## 4. Riepilogo inerziale", ""] + _tabella(
        ("grandezza", "valore", "unita", "significato"), righe
    )


def _sezione_controlli(controlli: Mapping[str, tuple[float, float, bool]]) -> list[str]:
    righe = [
        (nome, f"{atteso:.6g}", f"{ottenuto:.6g}", f"{abs(ottenuto - atteso):.2e}", _si_no(esito))
        for nome, (atteso, ottenuto, esito) in controlli.items()
    ]
    out = ["## 5. Controlli sui valori di riferimento (sezione 5 della specifica)", ""]
    out += [
        "I valori attesi della specifica si riferiscono alla **configurazione di "
        "riferimento** (`x_A0_W = \"-Y\"`, `elev_axis_W = \"X\"`, `theta_cad_deg = 0`, "
        "`z_O_cm = null`). I controlli qui sotto sono eseguiti su quella "
        "configurazione, cosi' da verificare il **codice** indipendentemente dalle "
        "scelte fatte in `config.yaml`.",
        "",
    ]
    out += _tabella(("grandezza", "atteso", "ottenuto", "scarto", "esito"), righe)
    return out


def _sezione_punti_di_lavoro(punti: Sequence[ModelloLineare]) -> list[str]:
    righe = []
    for ml in punti:
        radici = classifica(ml.poli)
        instabili = [r for r in radici if r.instabile]
        righe.append(
            (
                f"{np.rad2deg(ml.theta0):+.4g}",
                f"{-np.rad2deg(ml.theta0) + 0.0:.4g}",
                f"{np.rad2deg(ml.Omega0):.4g}",
                f"{ml.M0[0, 0]:.4f}",
                f"{ml.M0[1, 1]:.4f}",
                f"{ml.gamma:+.4g}",
                f"{ml.k_theta:+.4g}",
                f"{ml.tau0[1]:+.4g}",
                sum(1 for r in radici if r.tipo == "nullo"),
                len(instabili),
                f"{max((float(np.real(r.valore)) for r in instabili), default=0.0):.4g}",
            )
        )
    out = ["## 6. Punti di lavoro e linearizzazione", ""]
    out += _tabella(
        (
            "theta_0 [deg]", "eps_0 [deg]", "Omega_0 [deg/s]",
            "J_psi [kg m^2]", "I_yy [kg m^2]", "gamma [N m s/rad]",
            "k_theta [N m/rad]", "tau_theta0 [N m]",
            "poli nulli", "poli instabili", "max Re(lambda) [rad/s]",
        ),
        righe,
    )
    return out


def _sezione_poli_zeri(punti: Sequence[ModelloLineare]) -> list[str]:
    out = ["## 7. Poli, zeri e cancellazioni", ""]
    for ml in punti:
        out += [
            f"### theta_0 = {np.rad2deg(ml.theta0):+.4g} deg, "
            f"Omega_0 = {np.rad2deg(ml.Omega0):.4g} deg/s",
            "",
        ]
        righe = []
        for r in classifica(ml.poli):
            righe.append(
                (
                    "polo",
                    f"{np.real(r.valore):+.6g}",
                    f"{np.imag(r.valore):+.6g}",
                    f"{r.f_hz:.6g}",
                    "-" if r.zeta is None else f"{r.zeta:.4g}",
                    r.tipo,
                )
            )
        for etichetta, radici in ml.zeri_di_blocco().items():
            for r in classifica(radici):
                righe.append(
                    (
                        f"zero {etichetta}",
                        f"{np.real(r.valore):+.6g}",
                        f"{np.imag(r.valore):+.6g}",
                        f"{r.f_hz:.6g}",
                        "-" if r.zeta is None else f"{r.zeta:.4g}",
                        r.tipo,
                    )
                )
        out += _tabella(
            ("grandezza", "Re [rad/s]", "Im [rad/s]", "|lambda|/2pi [Hz]", "zeta", "tipo"),
            righe,
        )
        if ml.cancellazioni:
            out += ["Cancellazioni di radici comuni numeratore/denominatore:", ""]
            for c in ml.cancellazioni:
                out += [f"- {c}"]
            out += [
                "",
                "La cancellazione del fattore `s` e' attesa: `psi` e' ciclica, quindi "
                "`Delta(s)` ha una radice nell'origine che i numeratori di `G_12`, "
                "`G_21` e `G_22` condividono.",
                "",
            ]
        else:
            out += ["Nessuna cancellazione.", ""]
    return out


def _sezione_simulazioni(
    scenari: Mapping[str, RisultatoScenario], convergenze: Sequence[ConvergenzaLineare]
) -> list[str]:
    out = ["## 8. Simulazioni (sezione 8.4)", ""]
    righe = []
    for nome, r in scenari.items():
        righe.append(
            (
                nome,
                r.descrizione,
                f"{r.diagnostica.get('t_finale_s', float('nan')):.4g}",
                "-" if "guidata" not in r.errori else f"{r.errori['guidata'] * 100:.4g}",
                "-" if "incrociata" not in r.errori else f"{r.errori['incrociata'] * 100:.4g}",
                r.esito,
            )
        )
    out += _tabella(
        (
            "scenario", "descrizione", "orizzonte [s]",
            "err. coord. guidata [%]", "err. coord. incrociata [%]", "esito",
        ),
        righe,
    )

    for nome, r in scenari.items():
        if not r.note:
            continue
        out += [f"**{nome}.**", ""]
        for nota in r.note:
            out += [f"- {nota}"]
        out += [""]

    if "S4" in scenari:
        d = scenari["S4"].diagnostica
        out += [
            "### Verifica conservativa S4",
            "",
            f"- deriva relativa dell'energia `E = T + V`: `{d['deriva_relativa_energia']:.3e}`",
            f"- deriva relativa del momento `p_psi`: `{d['deriva_relativa_momento']:.3e}`",
            "- soglia richiesta dalla specifica: `1e-8`",
            "",
        ]

    if convergenze:
        out += [
            "### Convergenza della linearizzazione",
            "",
            "Il criterio dell'1 % della specifica si applica alla coordinata "
            "**guidata** dalla perturbazione, che risponde al primo ordine. La "
            "coordinata **incrociata** puo' invece rispondere prevalentemente al "
            "secondo ordine (il termine centrifugo `-1/2 J_psi'(theta) dpsi^2` "
            "sparisce dalla linearizzazione quando `Omega_0 = 0`), quindi il suo "
            "errore relativo va verificato per **convergenza**: se le matrici "
            "`M_0`, `C_0`, `K_0` sono corrette, l'errore e' proporzionale "
            "all'ampiezza, cioe' ha pendenza 1 in scala logaritmica.",
            "",
        ]
        out += _tabella(
            ("scenario", "pendenza coord. guidata", "pendenza coord. incrociata",
             "err. max all'ampiezza minima [%]"),
            [
                (
                    c.scenario,
                    f"{c.pendenza_guidata:.3f}",
                    f"{c.pendenza_incrociata:.3f}",
                    f"{c.errore_minimo_totale * 100:.4g}",
                )
                for c in convergenze
            ],
        )
    return out


def genera(contenuto: ContenutoReport, percorso: str | Path) -> Path:
    """Scrive ``report.md`` e restituisce il percorso."""
    c = contenuto
    percorso = Path(percorso)
    percorso.parent.mkdir(parents=True, exist_ok=True)

    righe: list[str] = [
        "# Modello dinamico rigido a 2 GdL di un telescopio alt-azimutale",
        "",
        f"Generato il {date.today().isoformat()} da `{c.comando}`.",
        "",
        "## 1. Nota di realismo",
        "",
        NOTA_REALISMO,
        "",
    ]
    righe += _sezione_parametri(c.par)
    righe += _sezione_avvisi(c.modello.avvisi)
    righe += _sezione_inerzia(c.modello)
    righe += _sezione_controlli(c.controlli_riferimento)
    righe += _sezione_punti_di_lavoro(c.punti_di_lavoro)
    righe += _sezione_poli_zeri(c.punti_di_lavoro)
    righe += _sezione_simulazioni(c.scenari, c.convergenze)

    if c.coerenza:
        righe += [
            "## 9. Coerenza fra le rappresentazioni",
            "",
            "Controllo incrociato fra spazio di stato, polinomi e FRF "
            "(sezione 9, test 10):",
            "",
        ]
        righe += [
            f"- scarto relativo fra autovalori di `A` e radici di `Delta(s)`: "
            f"`{c.coerenza.get('scarto_poli', float('nan')):.3e}`",
            f"- scarto relativo fra FRF da spazio di stato e da `P(j omega)^-1`: "
            f"`{c.coerenza.get('scarto_frf', float('nan')):.3e}`",
            "",
        ]

    righe += ["## 10. Estensione flessibile a 4 GdL (sezione 10)", ""]
    if c.flex_attivo:
        righe += [c.flex_messaggio, ""]
    else:
        righe += [
            "Disattivata (`flex_model: false`). E' l'unico modello di questa "
            "catena che possa produrre **modi strutturali**; richiede pero' "
            "inerzie dei rotori (datasheet dei motori) e rigidezze dei giunti "
            "(FEM o identificazione sperimentale). Finche' quei valori non sono "
            "reali, `flex4dof.py` rifiuta di produrre frequenze.",
            "",
        ]

    if c.figure:
        righe += ["## 11. Figure", ""]
        righe += _tabella(
            ("file", "contenuto"),
            [
                (", ".join(f"`figures/{p.name}`" for p in f.percorsi), f.descrizione)
                for f in c.figure
            ],
        )

    if c.tabelle or c.latex:
        righe += ["## 12. Altri file prodotti", ""]
        for p in list(c.tabelle) + list(c.latex):
            righe.append(f"- `{p.relative_to(percorso.parent) if percorso.parent in p.parents else p}`")
        righe.append("")

    percorso.write_text("\n".join(righe), encoding="utf-8")
    return percorso
