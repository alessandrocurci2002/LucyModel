#!/usr/bin/env python3
"""Esecuzione completa dell'analisi del modello rigido a 2 GdL.

Uso tipico::

    python run_all.py                       # analisi completa con config.yaml
    python run_all.py --quick               # griglie ridotte, per una prova rapida
    python run_all.py --theta0 -45 --omega0 0 5
    python run_all.py --config altro.yaml --output risultati_alternativi

Tutti i risultati finiscono in ``results/``: figure in ``figures/`` (PDF
vettoriale e PNG a 300 dpi), tabelle in ``tables/`` (CSV), equazioni e funzioni
di trasferimento in ``latex/``, e il documento di sintesi in ``report.md``.
"""

from __future__ import annotations

import argparse
import shlex
import sys
import time
from pathlib import Path
from typing import Sequence

import numpy as np

RADICE = Path(__file__).resolve().parent
sys.path.insert(0, str(RADICE / "src"))

from gimbal import analysis as an  # noqa: E402
from gimbal import cad_data  # noqa: E402
from gimbal import flex4dof as fx  # noqa: E402
from gimbal import inertia as inz  # noqa: E402
from gimbal import lagrange as lg  # noqa: E402
from gimbal import linearize as lz  # noqa: E402
from gimbal import params as pr  # noqa: E402
from gimbal import plotting as pl  # noqa: E402
from gimbal import report as rp  # noqa: E402
from gimbal import simulate as sim  # noqa: E402


def _titolo(testo: str) -> None:
    print()
    print("=" * 78)
    print(testo.center(78))
    print("=" * 78)


def controlli_riferimento() -> dict[str, tuple[float, float, bool]]:
    """Verifica i valori attesi della sezione 5 sulla configurazione di riferimento.

    I valori della specifica valgono per i default della sezione 4: si usa
    quindi :func:`gimbal.params.parametri_di_riferimento`, non ``config.yaml``.
    Cosi' il controllo verifica il **codice** e resta valido qualunque posa CAD
    si sia scelta per l'analisi.
    """
    par = pr.parametri_di_riferimento()
    mod = inz.costruisci_modello_inerziale(par)
    rif = cad_data.VALORI_RIFERIMENTO

    coppie: dict[str, tuple[float, float]] = {
        "J_A [kg m^2]": (rif["J_A"], mod.J_A),
        "I_B,G,ZZ + Steiner sull'asse di azimut [kg m^2]": (
            rif["I_B_G_ZZ_piu_steiner_azimut"], mod.I_zz,
        ),
        "J_psi(theta_cad = 0) [kg m^2]": (rif["J_psi_theta_cad_0"], float(mod.J_psi(0.0))),
        "I_xx,O con d_z = 0 [kg m^2]": (rif["I_xx_O_dz0"], mod.I_xx),
        "I_yy,O con d_z = 0 [kg m^2]": (rif["I_yy_O_dz0"], mod.I_yy),
        "I_zz,O con d_z = 0 [kg m^2]": (rif["I_zz_O_dz0"], mod.I_zz),
    }
    esiti = {
        nome: (atteso, ottenuto, bool(abs(ottenuto - atteso) <= 5e-5 * max(1.0, abs(atteso))))
        for nome, (atteso, ottenuto) in coppie.items()
    }

    # Autovalori dei tensori baricentrici, come visualizzati da Onshape.
    principali = inz.tabella_momenti_principali(par)
    for corpo, attesi in cad_data.AUTOVALORI_RIFERIMENTO.items():
        ottenuti = principali[corpo]
        for k, (atteso, ottenuto) in enumerate(zip(attesi, ottenuti)):
            esiti[f"autovalore {k + 1} di I_{corpo},G [kg cm^2]"] = (
                float(atteso), float(ottenuto), bool(abs(ottenuto - atteso) <= 0.01 * max(1.0, atteso))
            )
    return esiti


def controllo_incrociato_azimut(par: pr.Parametri, mod: inz.ModelloInerziale) -> tuple[float, float, bool]:
    """``J_psi(theta_cad)`` deve coincidere con ``I_ZZ`` di A+B dai dati CAD.

    E' il test 5 della sezione 9, valido per qualunque configurazione: il
    trasporto sull'asse verticale non dipende da ``z_O``.
    """
    dal_modello = float(mod.J_psi(par.theta_cad))
    dal_cad = inz.I_zz_assieme_asse_azimut(par)
    return dal_cad, dal_modello, bool(abs(dal_modello - dal_cad) <= 1e-9 * max(1.0, abs(dal_cad)))


def main(argv: Sequence[str] | None = None) -> int:
    analizzatore = argparse.ArgumentParser(
        description="Analisi completa del modello rigido a 2 GdL del gimbal alt-azimutale.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    analizzatore.add_argument(
        "--config", type=Path, default=RADICE / "config.yaml",
        help="file di configurazione YAML (default: config.yaml)",
    )
    analizzatore.add_argument(
        "--theta0", type=float, nargs="+", default=None, metavar="DEG",
        help="punti di lavoro in elevazione, in gradi (default: theta0_bode_deg del config)",
    )
    analizzatore.add_argument(
        "--omega0", type=float, nargs="+", default=None, metavar="DEG_S",
        help="velocita' di inseguimento in azimut, in deg/s (default: omega0_deg_s del config)",
    )
    analizzatore.add_argument(
        "--output", type=Path, default=RADICE / "results",
        help="cartella dei risultati (default: results/)",
    )
    analizzatore.add_argument(
        "--quick", action="store_true",
        help="griglie e simulazioni ridotte: prova rapida, non risultati definitivi",
    )
    analizzatore.add_argument(
        "--no-figure", action="store_true", help="salta la produzione delle figure",
    )
    argomenti = analizzatore.parse_args(argv)

    avvio = time.perf_counter()
    comando = "python " + " ".join(shlex.quote(a) for a in [Path(sys.argv[0]).name, *(argv or sys.argv[1:])])

    figure_dir = argomenti.output / "figures"
    tabelle_dir = argomenti.output / "tables"
    latex_dir = argomenti.output / "latex"
    for d in (figure_dir, tabelle_dir, latex_dir):
        d.mkdir(parents=True, exist_ok=True)

    # --- 1. parametri ------------------------------------------------------
    _titolo("1. PARAMETRI")
    par = pr.parametri_da_config(argomenti.config)
    par.stampa_avvisi()
    if argomenti.quick:
        print("\n  [--quick] griglie ridotte: i risultati sono indicativi, non definitivi.\n")

    theta0_punti = (
        tuple(float(np.deg2rad(v)) for v in argomenti.theta0)
        if argomenti.theta0 is not None
        else par.theta0_bode
    )
    omega0_vals = (
        tuple(float(np.deg2rad(v)) for v in argomenti.omega0)
        if argomenti.omega0 is not None
        else par.omega0
    )
    theta0_grid = par.theta0_grid[::10] if argomenti.quick else par.theta0_grid

    # --- 2. preprocessing inerziale ---------------------------------------
    _titolo("2. PREPROCESSING INERZIALE (sezione 5)")
    modello = inz.costruisci_modello_inerziale(par)
    modello.stampa_riepilogo()
    tabelle = [modello.salva_riepilogo(tabelle_dir / "inertia_summary.csv")]

    # --- 3. controlli di riferimento --------------------------------------
    _titolo("3. CONTROLLI SUI VALORI DI RIFERIMENTO (sezione 5)")
    controlli = controlli_riferimento()
    for nome, (atteso, ottenuto, esito) in controlli.items():
        stato = "OK " if esito else "FAIL"
        print(f"  [{stato}] {nome:52s} atteso {atteso:12.6g}  ottenuto {ottenuto:12.6g}")
    dal_cad, dal_modello, esito_incrociato = controllo_incrociato_azimut(par, modello)
    controlli["J_psi(theta_cad) vs I_ZZ di A+B dai dati CAD [kg m^2]"] = (
        dal_cad, dal_modello, esito_incrociato
    )
    print(
        f"  [{'OK ' if esito_incrociato else 'FAIL'}] "
        f"{'J_psi(theta_cad) vs I_ZZ di A+B dai dati CAD':52s} "
        f"atteso {dal_cad:12.6g}  ottenuto {dal_modello:12.6g}"
    )
    if not all(e for _, _, e in controlli.values()):
        print("\n  ATTENZIONE: almeno un controllo di riferimento e' fallito.\n")

    # --- 4. modello simbolico ---------------------------------------------
    _titolo("4. EQUAZIONI DI MOTO (sezione 6)")
    simbolico = lg.costruisci_modello_simbolico()
    lg.verifica_struttura_M(simbolico)
    lg.verifica_forme_chiuse(simbolico)
    lg.verifica_antisimmetria(simbolico)
    print("  struttura di M(theta)                          OK")
    print("  riduzione alle forme chiuse con prodotti nulli OK")
    print("  antisimmetria di dM/dt - 2C                    OK")
    latex = [lg.esporta_latex(latex_dir / "eom.tex", simbolico)]

    lz.verifica_target_regressione()
    print("  target di regressione della linearizzazione    OK")
    latex.append(lz.esporta_latex(latex_dir / "tf.tex"))

    # --- 5. linearizzazione e poli/zeri -----------------------------------
    _titolo("5. LINEARIZZAZIONE, POLI E ZERI (sezioni 7 e 8.1)")
    punti = [lz.linearizza(modello, th, om) for th in theta0_punti for om in omega0_vals]
    for ml in punti:
        r = an.riepilogo_punto_di_lavoro(ml)
        print(
            f"  theta0 = {r['theta0_deg']:+6.1f} deg  Omega0 = {r['omega0_deg_s']:5.2f} deg/s  "
            f"J_psi = {r['J_psi']:8.4f}  gamma = {r['gamma']:+9.4g}  "
            f"k_theta = {r['k_theta']:+10.4g}  tau_theta0 = {r['tau_theta0']:+9.4g} N m  "
            f"poli instabili: {r['n_poli_instabili']}"
        )
    coerenza = an.verifica_coerenza(punti[0])
    print(
        f"  coerenza fra rappresentazioni: scarto poli {coerenza['scarto_poli']:.2e}, "
        f"scarto FRF {coerenza['scarto_frf']:.2e}"
    )

    dz_tabelle = None if argomenti.quick else par.dz_sensitivity
    consolidato, singoli = an.tabelle_poli_zeri(
        modello, tabelle_dir,
        theta0_punti=theta0_punti, omega0_vals=omega0_vals, dz_vals=dz_tabelle,
    )
    tabelle.append(consolidato)
    print(f"  tabelle di poli e zeri: 1 consolidata + {len(singoli)} per combinazione")

    # --- 6. analisi parametriche ------------------------------------------
    _titolo("6. ANALISI PARAMETRICHE (sezione 8.2)")
    sweeps_theta = [an.sweep_theta0(modello, theta0_grid, om) for om in omega0_vals]
    for sw, om in zip(sweeps_theta, omega0_vals):
        tabelle.append(
            an.salva_sweep(tabelle_dir / f"sweep_theta0_omega{np.rad2deg(om):05.2f}.csv", sw)
        )
    sweep_dz = an.sweep_dz(
        modello, par.dz_sensitivity, float(np.deg2rad(-45.0)), omega0_vals[0]
    )
    tabelle.append(an.salva_sweep(tabelle_dir / "sweep_dz.csv", sweep_dz))
    print(f"  scansioni in theta_0: {len(sweeps_theta)}  ({theta0_grid.size} punti ciascuna)")
    print(f"  scansione in d_z: {len(par.dz_sensitivity)} punti")

    # --- 7. simulazioni ----------------------------------------------------
    _titolo("7. SIMULAZIONI NON LINEARI (sezione 8.4)")
    scenari = sim.esegui_tutti(modello, rapido=argomenti.quick)
    for nome, ris in scenari.items():
        tabelle.append(ris.salva_csv(tabelle_dir / f"scenario_{nome}.csv"))
        guidata = ris.errori.get("guidata")
        incrociata = ris.errori.get("incrociata")
        print(
            f"  {nome} [{ris.esito}] orizzonte {ris.diagnostica.get('t_finale_s', float('nan')):.3g} s"
            + (f"  err. guidata {guidata * 100:.3g} %" if guidata is not None else "")
            + (f"  err. incrociata {incrociata * 100:.3g} %" if incrociata is not None else "")
        )
        for nota in ris.note:
            print(f"      NOTA: {nota}")

    convergenze = []
    if not argomenti.quick:
        for nome, om in (("S1", 0.0), ("S3", par.omega0_max)):
            convergenze.append(
                sim.convergenza_lineare(modello, Omega0=om, nome=nome, n_punti=1501)
            )
            c = convergenze[-1]
            print(
                f"  convergenza {nome}: pendenza {c.pendenza_guidata:.3f} (guidata), "
                f"{c.pendenza_incrociata:.3f} (incrociata) - attesa 1"
            )

    # --- 8. figure ---------------------------------------------------------
    figure: list[pl.Figura] = []
    if not argomenti.no_figure:
        _titolo("8. FIGURE (sezioni 8.2 e 8.3)")
        pl.applica_stile()

        for om in (omega0_vals[0], omega0_vals[-1]) if len(omega0_vals) > 1 else omega0_vals:
            modelli = [lz.linearizza(modello, th, om) for th in theta0_punti]
            fig = pl.bode_2x2(
                modelli,
                etichette=[rf"$\theta_0$ = {np.rad2deg(th):g} deg" for th in theta0_punti],
                titolo=rf"Diagrammi di Bode, $\Omega_0$ = {np.rad2deg(om):g} deg/s",
            )
            pl.registra(
                figure, fig, figure_dir, f"bode_omega{np.rad2deg(om):05.2f}",
                f"Bode 2x2 delle quattro FdT a Omega_0 = {np.rad2deg(om):g} deg/s, "
                f"per theta_0 in {[round(float(np.rad2deg(t)), 1) for t in theta0_punti]} deg",
            )

        pl.registra(
            figure,
            pl.frequenze_vs_parametro(
                sweeps_theta,
                titolo=r"Poli e zeri in funzione del punto di lavoro $\theta_0$",
            ),
            figure_dir, "frequenze_vs_theta0",
            "Frequenze di poli e zeri in funzione di theta_0, una curva per ogni Omega_0",
        )
        pl.registra(
            figure,
            pl.frequenze_vs_parametro(
                [sweep_dz], titolo=r"Poli e zeri in funzione dello sbilanciamento $d_z$"
            ),
            figure_dir, "frequenze_vs_dz",
            "Frequenze di poli e zeri in funzione dello sbilanciamento d_z",
        )
        pl.registra(
            figure, pl.sensibilita_dz(sweep_dz), figure_dir, "sensibilita_dz",
            "Rigidezza equivalente e coppia di mantenimento in funzione di d_z",
        )
        pl.registra(
            figure,
            pl.J_psi_vs_theta(par.theta0_grid, an.curva_J_psi(modello, par.theta0_grid)),
            figure_dir, "J_psi_vs_theta",
            "Inerzia efficace di azimut J_psi(theta) sul campo operativo "
            f"[{np.rad2deg(par.theta0_grid.min()):.0f}, "
            f"{np.rad2deg(par.theta0_grid.max()):.0f}] deg",
        )

        for nome, ris in scenari.items():
            pl.registra(
                figure, pl.storie_temporali(ris), figure_dir, f"scenario_{nome}",
                f"{nome}: {ris.descrizione}",
            )
        if "S4" in scenari:
            ris = scenari["S4"]
            mod0, mn0 = sim.modello_conservativo(modello)
            E = np.array([mn0.energia(ris.q[k], ris.dq[k]) for k in range(ris.t.size)])
            p_psi = np.array([mn0.momento_psi(ris.q[k], ris.dq[k]) for k in range(ris.t.size)])
            pl.registra(
                figure, pl.conservazione(ris, E, p_psi), figure_dir, "S4_conservazione",
                "S4: deriva relativa di energia e momento coniugato",
            )
        if convergenze:
            pl.registra(
                figure, pl.convergenza_linearizzazione(convergenze), figure_dir,
                "convergenza_linearizzazione",
                "Errore lineare/non lineare in funzione dell'ampiezza: pendenza attesa 1",
            )
        print(f"  {len(figure)} figure salvate in PDF e PNG a 300 dpi")

    # --- 9. modello flessibile --------------------------------------------
    flex_attivo = False
    flex_messaggio = ""
    if par.flex_model:
        _titolo("9. MODELLO FLESSIBILE A 4 GdL (sezione 10)")
        mf = fx.esegui(modello, theta0_punti[0])
        if mf is not None:
            flex_attivo = True
            frequenze = mf.frequenze_proprie()
            flex_messaggio = (
                "Modello flessibile attivo. Frequenze proprie [Hz]: "
                + ", ".join(f"{f:.4g}" for f in frequenze)
                + ". Antirisonanze collocate [Hz]: "
                + ", ".join(
                    f"{nome}: {v[0]:.4g}" for nome, v in mf.zeri_collocati().items()
                )
                + "."
            )
        else:
            flex_messaggio = (
                "`flex_model: true` ma i parametri non sono disponibili: nessun "
                "risultato prodotto, come previsto dalla sezione 10."
            )

    # --- 10. report --------------------------------------------------------
    _titolo("10. REPORT")
    contenuto = rp.ContenutoReport(
        par=par,
        modello=modello,
        controlli_riferimento=controlli,
        punti_di_lavoro=punti,
        scenari=scenari,
        convergenze=convergenze,
        figure=figure,
        tabelle=tabelle,
        latex=latex,
        coerenza=coerenza,
        flex_attivo=flex_attivo,
        flex_messaggio=flex_messaggio,
        comando=comando,
    )
    percorso_report = rp.genera(contenuto, argomenti.output / "report.md")
    print(f"  {percorso_report}")
    print(f"\nCompletato in {time.perf_counter() - avvio:.1f} s.")

    incoerenze = [a for a in modello.avvisi if a.severita == "incoerenza"]
    if incoerenze:
        print(
            f"\n  {len(incoerenze)} incoerenza/e nei dati di ingresso: "
            "vedi la sezione 3 del report prima di citare questi numeri."
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
