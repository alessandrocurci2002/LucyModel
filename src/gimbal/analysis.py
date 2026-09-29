"""Poli, zeri, frequenze e analisi parametriche (sezione 8.1 e 8.2).

**Nota di realismo (sezione 0 della specifica).** Un modello a corpi rigidi
non contiene modi strutturali. Le sole radici che questo modulo puo'
produrre sono:

* i **poli nulli** del moto rigido (``psi`` e' ciclica; con ``b_psi = 0``
  l'asse di azimut e' un doppio integratore e il polo nullo e' doppio);
* una **dinamica a bassa frequenza** dovuta a gravita' (sbilanciamento),
  accoppiamento giroscopico e rigidezza centrifuga, tipicamente sotto 1 Hz.

Le frequenze riportate non sono e non vanno presentate come frequenze
proprie strutturali: la classificazione in :class:`Radice` lo rende esplicito
attraverso il campo ``origine``.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Literal, Sequence

import numpy as np

from . import inertia as inz
from . import linearize as lz
from .inertia import ModelloInerziale
from .linearize import ModelloLineare

TipoRadice = Literal["nullo", "reale_stabile", "reale_instabile", "complesso"]

TOLLERANZA_NULLO: float = 1e-9
"""Soglia in rad/s sotto la quale una radice e' considerata nulla."""


# ---------------------------------------------------------------------------
# Classificazione delle radici
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Radice:
    """Una radice (polo o zero) classificata secondo la sezione 8.1.

    Attributes
    ----------
    valore
        La radice nel piano complesso, in rad/s.
    tipo
        ``"nullo"`` (moto rigido), ``"reale_stabile"``, ``"reale_instabile"``
        (per esempio pendolo inverso da sbilanciamento) oppure
        ``"complesso"`` (coppia coniugata).
    f_hz
        ``|lambda| / 2 pi``. Per una coppia complessa e' la frequenza
        naturale; per una radice reale e' solo il modulo riscalato, non una
        frequenza di oscillazione.
    zeta
        Smorzamento ``-Re(lambda)/|lambda|``, definito per le radici non nulle.
    tau_s
        Costante di tempo ``1/|Re(lambda)|``, in s.
    origine
        Interpretazione fisica, in italiano: serve a impedire che una radice
        del modello rigido venga presentata come modo strutturale.
    """

    valore: complex
    tipo: TipoRadice
    f_hz: float
    zeta: float | None
    tau_s: float | None
    origine: str

    @property
    def instabile(self) -> bool:
        return float(np.real(self.valore)) > TOLLERANZA_NULLO


def classifica_radice(valore: complex, *, tol: float = TOLLERANZA_NULLO) -> Radice:
    """Classifica una singola radice secondo la sezione 8.1."""
    valore = complex(valore)
    modulo = float(abs(valore))
    reale, immaginaria = float(np.real(valore)), float(np.imag(valore))

    if modulo <= tol:
        return Radice(
            valore=valore,
            tipo="nullo",
            f_hz=0.0,
            zeta=None,
            tau_s=None,
            origine="moto rigido (coordinata ciclica / integratore)",
        )

    if abs(immaginaria) <= tol * max(1.0, modulo):
        tipo: TipoRadice = "reale_instabile" if reale > 0.0 else "reale_stabile"
        origine = (
            "radice reale positiva: instabilita' statica da sbilanciamento "
            "gravitazionale (pendolo inverso), non un modo di vibrare"
            if reale > 0.0
            else "radice reale negativa: decadimento aperiodico "
            "(gravita' stabilizzante e/o smorzamento viscoso)"
        )
        return Radice(
            valore=valore,
            tipo=tipo,
            f_hz=modulo / (2.0 * np.pi),
            zeta=-reale / modulo,
            tau_s=1.0 / abs(reale) if reale != 0.0 else None,
            origine=origine,
        )

    return Radice(
        valore=valore,
        tipo="complesso",
        f_hz=modulo / (2.0 * np.pi),
        zeta=-reale / modulo,
        tau_s=1.0 / abs(reale) if reale != 0.0 else None,
        origine=(
            "coppia complessa coniugata di origine gravitazionale, giroscopica "
            "o centrifuga: dinamica rigida a bassa frequenza, NON un modo "
            "strutturale (il modello non contiene flessibilita')"
        ),
    )


def classifica(radici: Iterable[complex], *, tol: float = TOLLERANZA_NULLO) -> list[Radice]:
    """Classifica un insieme di radici, ordinandole per modulo crescente."""
    elencate = [classifica_radice(r, tol=tol) for r in radici]
    return sorted(elencate, key=lambda r: (abs(r.valore), float(np.imag(r.valore))))


# ---------------------------------------------------------------------------
# Tabelle di poli e zeri
# ---------------------------------------------------------------------------

INTESTAZIONE_RADICI: tuple[str, ...] = (
    "grandezza",
    "theta0_deg",
    "omega0_deg_s",
    "dz_cm",
    "re_rad_s",
    "im_rad_s",
    "modulo_rad_s",
    "f_Hz",
    "zeta",
    "tau_s",
    "tipo",
    "origine",
)


def righe_radici(ml: ModelloLineare) -> list[dict[str, object]]:
    """Righe di tabella con poli e zeri di un punto di lavoro."""
    theta0_deg = float(np.rad2deg(ml.theta0))
    omega0_deg_s = float(np.rad2deg(ml.Omega0))
    dz_cm = ml.modello_inerziale.d_z * 100.0

    righe: list[dict[str, object]] = []

    def aggiungi(grandezza: str, radici: Iterable[complex]) -> None:
        for r in classifica(radici):
            righe.append(
                {
                    "grandezza": grandezza,
                    "theta0_deg": theta0_deg,
                    "omega0_deg_s": omega0_deg_s,
                    "dz_cm": dz_cm,
                    "re_rad_s": float(np.real(r.valore)),
                    "im_rad_s": float(np.imag(r.valore)),
                    "modulo_rad_s": float(abs(r.valore)),
                    "f_Hz": r.f_hz,
                    "zeta": "" if r.zeta is None else r.zeta,
                    "tau_s": "" if r.tau_s is None else r.tau_s,
                    "tipo": r.tipo,
                    "origine": r.origine,
                }
            )

    aggiungi("polo (autovalore di A)", ml.poli)
    for etichetta, radici in ml.zeri_di_blocco().items():
        aggiungi(f"zero: {etichetta}", radici)
    return righe


def salva_righe(percorso: str | Path, righe: Sequence[dict[str, object]]) -> Path:
    """Scrive righe di radici in un CSV con intestazione fissa."""
    percorso = Path(percorso)
    percorso.parent.mkdir(parents=True, exist_ok=True)
    with percorso.open("w", encoding="utf-8", newline="") as f:
        scrittore = csv.DictWriter(f, fieldnames=list(INTESTAZIONE_RADICI))
        scrittore.writeheader()
        for riga in righe:
            scrittore.writerow(riga)
    return percorso


def tabelle_poli_zeri(
    modello: ModelloInerziale,
    cartella: str | Path,
    *,
    theta0_punti: Sequence[float],
    omega0_vals: Sequence[float],
    dz_vals: Sequence[float] | None = None,
) -> tuple[Path, list[Path]]:
    """Tabelle di poli e zeri per le combinazioni (theta_0, Omega_0, d_z).

    La specifica chiede una tabella per ogni combinazione. La griglia
    completa (91 valori di theta_0 x 4 di Omega_0 x 7 di d_z) produrrebbe
    migliaia di file, quindi:

    * un CSV **consolidato** ``poles_zeros_all.csv`` con una riga per ogni
      radice di ogni combinazione;
    * un CSV **per combinazione** nella sottocartella ``poles_zeros/``,
      limitato ai punti di lavoro di interesse passati in ``theta0_punti``.

    Returns
    -------
    (consolidato, elenco_per_combinazione)
    """
    cartella = Path(cartella)
    sotto = cartella / "poles_zeros"
    sotto.mkdir(parents=True, exist_ok=True)

    # Si svuota la cartella prima di riempirla: se la griglia cambia fra due
    # esecuzioni, i file della precedente resterebbero indistinguibili da
    # quelli nuovi e potrebbero finire in tesi come risultati correnti.
    for vecchio_csv in sotto.glob("*.csv"):
        vecchio_csv.unlink()

    dz_vals = list(dz_vals) if dz_vals is not None else [modello.d_z]
    tutte: list[dict[str, object]] = []
    file_singoli: list[Path] = []

    for dz in dz_vals:
        mod_dz = (
            modello
            if abs(dz - modello.d_z) < 1e-15
            else inz.modello_con_dz(modello.par, dz, verifica=False)
        )
        for th in theta0_punti:
            for om in omega0_vals:
                ml = lz.linearizza(mod_dz, float(th), float(om))
                righe = righe_radici(ml)
                tutte.extend(righe)
                nome = (
                    f"theta{np.rad2deg(th):+06.1f}_"
                    f"omega{np.rad2deg(om):05.2f}_"
                    f"dz{dz * 100:+06.2f}.csv"
                ).replace("+", "p").replace("-", "m")
                file_singoli.append(salva_righe(sotto / nome, righe))

    consolidato = salva_righe(cartella / "poles_zeros_all.csv", tutte)
    return consolidato, file_singoli


# ---------------------------------------------------------------------------
# Analisi parametriche (sezione 8.2)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Sweep:
    """Risultato di una scansione parametrica.

    Attributes
    ----------
    variabile
        ``"theta0"`` oppure ``"dz"``.
    ascissa
        Valori della variabile nelle unita' di presentazione (deg oppure cm).
    poli
        Array complesso ``(n, 4)`` con gli autovalori di ``A``.
    zeri_G11, zeri_G22
        Array complessi ``(n, 2)`` con gli zeri delle FdT collocate, cioe' le
        radici di ``P_22`` e di ``P_11``.
    J_psi
        Inerzia efficace di azimut lungo la scansione, in kg m^2.
    k_theta
        Rigidezza equivalente dell'asse di elevazione, in N m/rad.
    tau_theta0
        Coppia di mantenimento, in N m.
    """

    variabile: str
    etichetta_ascissa: str
    ascissa: np.ndarray
    omega0: float
    poli: np.ndarray
    zeri_G11: np.ndarray
    zeri_G22: np.ndarray
    J_psi: np.ndarray
    k_theta: np.ndarray
    tau_theta0: np.ndarray


def _riempi(radici: np.ndarray, n: int) -> np.ndarray:
    """Porta un vettore di radici a lunghezza ``n`` con NaN, per l'impilamento.

    L'ordinamento e' per **modulo** crescente, non per parte reale: nei
    grafici parametrici si traccia proprio ``|lambda|/2pi``, e il modulo varia
    con continuita' lungo la scansione, mentre la parte reale di una radice
    quasi immaginaria pura cambia segno in modo numericamente casuale e
    farebbe saltare le curve da un ramo all'altro.
    """
    out = np.full(n, np.nan + 0j, dtype=complex)
    radici = np.atleast_1d(np.asarray(radici, dtype=complex))
    m = min(n, radici.size)
    if m:
        ordinate = sorted(radici, key=lambda z: (abs(z), z.real, z.imag))
        out[:m] = ordinate[:m]
    return out


def sweep_theta0(
    modello: ModelloInerziale, theta0_grid: Sequence[float], omega0_val: float
) -> Sweep:
    """Poli, zeri e grandezze derivate in funzione di theta_0 (sezione 8.2)."""
    theta0_grid = np.asarray(theta0_grid, dtype=float)
    n = theta0_grid.size
    poli = np.empty((n, 4), dtype=complex)
    z11 = np.empty((n, 2), dtype=complex)
    z22 = np.empty((n, 2), dtype=complex)
    J = np.empty(n)
    kt = np.empty(n)
    tau = np.empty(n)

    for i, th in enumerate(theta0_grid):
        ml = lz.linearizza(modello, float(th), float(omega0_val))
        poli[i] = _riempi(ml.poli, 4)
        blocchi = ml.zeri_di_blocco()
        z11[i] = _riempi(blocchi["P22 (zeri di G11)"], 2)
        z22[i] = _riempi(blocchi["P11 (zeri di G22)"], 2)
        J[i] = float(modello.J_psi(float(th)))
        kt[i] = ml.k_theta
        tau[i] = float(ml.tau0[1])

    return Sweep(
        variabile="theta0",
        etichetta_ascissa=r"$\theta_0$ [deg]",
        ascissa=np.rad2deg(theta0_grid),
        omega0=float(omega0_val),
        poli=poli,
        zeri_G11=z11,
        zeri_G22=z22,
        J_psi=J,
        k_theta=kt,
        tau_theta0=tau,
    )


def sweep_dz(
    modello: ModelloInerziale,
    dz_vals: Sequence[float],
    theta0_val: float,
    omega0_val: float,
) -> Sweep:
    """Poli, zeri e grandezze derivate in funzione di d_z (sezione 8.2).

    Lo sbilanciamento non viene imposto a mano sul solo ``d``: per ogni
    valore si rifa' il preprocessing inerziale spostando il punto O, cosi'
    che ``I_B,O`` resti coerente con il teorema di Steiner.
    """
    dz_vals = np.asarray(dz_vals, dtype=float)
    n = dz_vals.size
    poli = np.empty((n, 4), dtype=complex)
    z11 = np.empty((n, 2), dtype=complex)
    z22 = np.empty((n, 2), dtype=complex)
    J = np.empty(n)
    kt = np.empty(n)
    tau = np.empty(n)

    for i, dz in enumerate(dz_vals):
        mod = inz.modello_con_dz(modello.par, float(dz), verifica=False)
        ml = lz.linearizza(mod, float(theta0_val), float(omega0_val))
        poli[i] = _riempi(ml.poli, 4)
        blocchi = ml.zeri_di_blocco()
        z11[i] = _riempi(blocchi["P22 (zeri di G11)"], 2)
        z22[i] = _riempi(blocchi["P11 (zeri di G22)"], 2)
        J[i] = float(mod.J_psi(float(theta0_val)))
        kt[i] = ml.k_theta
        tau[i] = float(ml.tau0[1])

    return Sweep(
        variabile="dz",
        etichetta_ascissa=r"$d_z$ [cm]",
        ascissa=dz_vals * 100.0,
        omega0=float(omega0_val),
        poli=poli,
        zeri_G11=z11,
        zeri_G22=z22,
        J_psi=J,
        k_theta=kt,
        tau_theta0=tau,
    )


def salva_sweep(percorso: str | Path, sweep: Sweep) -> Path:
    """Esporta una scansione parametrica in CSV."""
    percorso = Path(percorso)
    percorso.parent.mkdir(parents=True, exist_ok=True)
    with percorso.open("w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(
            [
                sweep.variabile,
                "omega0_deg_s",
                "J_psi_kg_m2",
                "k_theta_N_m_rad",
                "tau_theta0_N_m",
                *[f"polo{i + 1}_re" for i in range(sweep.poli.shape[1])],
                *[f"polo{i + 1}_im" for i in range(sweep.poli.shape[1])],
                "zero_G11_1_re", "zero_G11_1_im", "zero_G11_2_re", "zero_G11_2_im",
                "zero_G22_1_re", "zero_G22_1_im", "zero_G22_2_re", "zero_G22_2_im",
            ]
        )
        om = np.rad2deg(sweep.omega0)
        for i, x in enumerate(sweep.ascissa):
            w.writerow(
                [
                    f"{x:.6g}", f"{om:.6g}",
                    f"{sweep.J_psi[i]:.10g}", f"{sweep.k_theta[i]:.10g}",
                    f"{sweep.tau_theta0[i]:.10g}",
                    *[f"{np.real(v):.10g}" for v in sweep.poli[i]],
                    *[f"{np.imag(v):.10g}" for v in sweep.poli[i]],
                    *[f"{fn(v):.10g}" for v in sweep.zeri_G11[i] for fn in (np.real, np.imag)],
                    *[f"{fn(v):.10g}" for v in sweep.zeri_G22[i] for fn in (np.real, np.imag)],
                ]
            )
    return percorso


def curva_J_psi(modello: ModelloInerziale, theta_grid: Sequence[float]) -> np.ndarray:
    """``J_psi(theta)`` sulla griglia richiesta, in kg m^2 (sezione 8.2)."""
    return np.asarray(modello.J_psi(np.asarray(theta_grid, dtype=float)), dtype=float)


# ---------------------------------------------------------------------------
# Coerenza fra le rappresentazioni (sezione 9, test 10)
# ---------------------------------------------------------------------------


def scarto_fra_insiemi(a: np.ndarray, b: np.ndarray) -> float:
    """Distanza massima fra due insiemi di radici, con accoppiamento ottimo.

    Confrontare due liste di autovalori ordinandole e sottraendole termine a
    termine e' fragile: un ordinamento per parte reale su radici quasi
    immaginarie pure, le cui parti reali valgono ``1e-20`` con segni casuali,
    puo' permutare i due elenchi e far apparire enorme uno scarto nullo.
    Qui si cerca invece l'**abbinamento ottimo** fra le due liste e si
    restituisce la distanza massima fra elementi abbinati: il risultato non
    dipende dall'ordine.
    """
    from scipy.optimize import linear_sum_assignment

    a = np.atleast_1d(np.asarray(a, dtype=complex))
    b = np.atleast_1d(np.asarray(b, dtype=complex))
    if a.size != b.size:
        raise ValueError(f"insiemi di dimensione diversa: {a.size} e {b.size}")
    distanze = np.abs(a[:, None] - b[None, :])
    righe, colonne = linear_sum_assignment(distanze)
    return float(np.max(distanze[righe, colonne]))


def verifica_coerenza(
    ml: ModelloLineare, *, omega: Sequence[float] | None = None, tol: float = 1e-8
) -> dict[str, float]:
    """Confronta spazio di stato, polinomi e FRF.

    Verifica che gli autovalori di ``A`` coincidano con le radici di
    ``Delta(s)`` e che la FRF ottenuta da ``P(j omega)^{-1}`` coincida con
    quella ottenuta dallo spazio di stato.

    Returns
    -------
    dict
        ``{"scarto_poli": ..., "scarto_frf": ...}``, scarti massimi.

    Raises
    ------
    AssertionError
        Se uno dei due scarti supera ``tol`` (in senso relativo).
    """
    autovalori = np.asarray(ml.poli, dtype=complex)
    radici = np.asarray(ml.radici_delta, dtype=complex)
    if autovalori.size != radici.size:
        raise AssertionError(
            f"numero di autovalori ({autovalori.size}) diverso dal grado di "
            f"Delta ({radici.size})"
        )
    scala_poli = max(1.0, float(np.max(np.abs(autovalori))))
    scarto_poli = scarto_fra_insiemi(autovalori, radici) / scala_poli

    omega = np.asarray(
        omega if omega is not None else np.logspace(-3, 2, 60) * 2.0 * np.pi, dtype=float
    )
    G1 = ml.frf(omega)
    G2 = ml.frf_da_spazio_di_stato(omega)
    scala_frf = float(np.max(np.abs(G1)))
    scarto_frf = float(np.max(np.abs(G1 - G2))) / max(scala_frf, 1e-30)

    if scarto_poli > tol:
        raise AssertionError(
            f"autovalori di A e radici di Delta differiscono: {scarto_poli:.3e} > {tol:.1e}"
        )
    if scarto_frf > tol:
        raise AssertionError(
            f"FRF da spazio di stato e da P(jw)^-1 differiscono: {scarto_frf:.3e} > {tol:.1e}"
        )
    return {"scarto_poli": scarto_poli, "scarto_frf": scarto_frf}


def riepilogo_punto_di_lavoro(ml: ModelloLineare) -> dict[str, object]:
    """Riepilogo compatto di un punto di lavoro, per il report."""
    radici = classifica(ml.poli)
    instabili = [r for r in radici if r.instabile]
    complessi = [r for r in radici if r.tipo == "complesso"]
    return {
        "theta0_deg": float(np.rad2deg(ml.theta0)),
        "epsilon0_deg": float(-np.rad2deg(ml.theta0)),
        "omega0_deg_s": float(np.rad2deg(ml.Omega0)),
        "J_psi": float(ml.M0[0, 0]),
        "I_yy": float(ml.M0[1, 1]),
        "M_12": float(ml.M0[0, 1]),
        "gamma": ml.gamma,
        "k_theta": ml.k_theta,
        "tau_theta0": float(ml.tau0[1]),
        "n_poli_nulli": sum(1 for r in radici if r.tipo == "nullo"),
        "n_poli_instabili": len(instabili),
        "polo_instabile_max_rad_s": max((float(np.real(r.valore)) for r in instabili), default=0.0),
        "f_complesse_Hz": [r.f_hz for r in complessi],
        "zeta_complessi": [r.zeta for r in complessi],
    }
