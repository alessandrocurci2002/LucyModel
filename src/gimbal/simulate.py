"""Simulazioni non lineari e confronto con il modello lineare (sezione 8.4).

Integrazione con :func:`scipy.integrate.solve_ivp`, metodo ``DOP853``
(``Radau`` se il problema risulta rigido), ``rtol = 1e-10``, ``atol = 1e-12``
e uscita densa. Il modello lineare e' integrato con
:func:`scipy.signal.lsim`.

In tutti gli scenari con punto di lavoro assegnato le **coppie nominali**
``tau_0`` della sezione 7 sono applicate in avanti (feedforward): senza di
esse ``(theta_0, dpsi = Omega_0)`` non e' un equilibrio del modello non
lineare e il confronto con il modello linearizzato non avrebbe senso. La
perturbazione e' il solo doppietto, e il confronto avviene fra ``q - q_0``
del modello non lineare e ``delta q`` del modello lineare.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Callable, Literal, Sequence

import numpy as np
from scipy.integrate import solve_ivp
from scipy.signal import lsim

from . import lagrange as lg
from . import linearize as lz
from .inertia import ModelloInerziale
from .lagrange import ModelloNumerico
from .linearize import ModelloLineare

RTOL: float = 1e-10
ATOL: float = 1e-12
METODO_DEFAULT: str = "DOP853"


# ---------------------------------------------------------------------------
# Forzanti
# ---------------------------------------------------------------------------


def doppietto(
    t: float | np.ndarray, *, t_inizio: float, durata: float, ampiezza: float
) -> float | np.ndarray:
    """Doppietto di coppia: ``+ampiezza`` poi ``-ampiezza``, a media nulla.

    A media nulla per non introdurre deriva del momento: e' la forzante
    naturale per eccitare la dinamica intorno a un punto di lavoro senza
    spostarlo.
    """
    t = np.asarray(t, dtype=float)
    prima = (t >= t_inizio) & (t < t_inizio + durata)
    seconda = (t >= t_inizio + durata) & (t < t_inizio + 2.0 * durata)
    risultato = ampiezza * prima.astype(float) - ampiezza * seconda.astype(float)
    return float(risultato) if risultato.ndim == 0 else risultato


def bang_bang(
    t: float | np.ndarray, *, t_inizio: float, durata: float, ampiezza: float
) -> float | np.ndarray:
    """Profilo bang-bang: accelerazione per meta' tempo, decelerazione per l'altra."""
    return doppietto(t, t_inizio=t_inizio, durata=durata / 2.0, ampiezza=ampiezza)


# ---------------------------------------------------------------------------
# Integrazione
# ---------------------------------------------------------------------------


def integra_nonlineare(
    mn: ModelloNumerico,
    t_eval: np.ndarray,
    q0: np.ndarray,
    dq0: np.ndarray,
    tau: Callable[[float], np.ndarray],
    *,
    metodo: str = METODO_DEFAULT,
) -> tuple[np.ndarray, np.ndarray]:
    """Integra il modello non lineare a 2 GdL.

    Returns
    -------
    (q, dq)
        Array di forma ``(len(t_eval), 2)``.
    """

    def campo(t: float, x: np.ndarray) -> np.ndarray:
        q, dq = x[:2], x[2:]
        return np.concatenate([dq, mn.accelerazioni(q, dq, tau(t))])

    x0 = np.concatenate([np.asarray(q0, dtype=float), np.asarray(dq0, dtype=float)])
    sol = solve_ivp(
        campo,
        (float(t_eval[0]), float(t_eval[-1])),
        x0,
        method=metodo,
        t_eval=t_eval,
        rtol=RTOL,
        atol=ATOL,
        dense_output=True,
    )
    if not sol.success:
        raise RuntimeError(f"integrazione non riuscita ({metodo}): {sol.message}")
    return sol.y[:2].T.copy(), sol.y[2:].T.copy()


def risposta_lineare(
    ml: ModelloLineare, t: np.ndarray, dtau: np.ndarray, x0: np.ndarray | None = None
) -> tuple[np.ndarray, np.ndarray]:
    """Risposta del modello linearizzato con :func:`scipy.signal.lsim`.

    Parameters
    ----------
    dtau
        Ingresso di forma ``(len(t), 2)``: perturbazione di coppia.
    x0
        Stato iniziale ``[dpsi, dtheta, d(dpsi), d(dtheta)]``.

    Returns
    -------
    (dq, ddq)
        Perturbazioni di posizione e velocita', ``(len(t), 2)``.
    """
    sistema = (ml.A, ml.Bu, np.eye(4), np.zeros((4, 2)))
    _, y, _ = lsim(sistema, U=np.asarray(dtau, dtype=float), T=np.asarray(t, dtype=float), X0=x0)
    y = np.asarray(y)
    return y[:, :2].copy(), y[:, 2:].copy()


# ---------------------------------------------------------------------------
# Risultato di uno scenario
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class RisultatoScenario:
    """Storie temporali e diagnostica di uno scenario.

    Attributes
    ----------
    q, dq
        Traiettoria del modello non lineare, ``(n, 2)``.
    dq_lin, ddq_lin
        Perturbazioni del modello lineare, ``(n, 2)``, oppure ``None`` se lo
        scenario non prevede il confronto.
    q0, dq0_riferimento
        Punto di lavoro sottratto per ottenere le perturbazioni.
    errori
        Errori relativi lineare/non lineare per coordinata.
    diagnostica
        Grandezze scalari di interesse (energia, momento, deriva, ...).
    note
        Avvertenze in italiano, riportate nel report.
    """

    nome: str
    descrizione: str
    t: np.ndarray
    q: np.ndarray
    dq: np.ndarray
    tau: np.ndarray
    q0: np.ndarray
    dq0_riferimento: np.ndarray
    dq_lin: np.ndarray | None = None
    ddq_lin: np.ndarray | None = None
    errori: dict[str, float] = field(default_factory=dict)
    diagnostica: dict[str, float] = field(default_factory=dict)
    note: tuple[str, ...] = ()
    esito: Literal["ok", "attenzione", "non applicabile"] = "ok"

    @property
    def dq_nonlineare(self) -> np.ndarray:
        """Perturbazione di posizione del modello non lineare, ``q - q_0(t)``."""
        # psi cresce linearmente nel punto di lavoro rotante: va sottratto Omega_0 t.
        riferimento = self.q0[None, :] + np.outer(self.t - self.t[0], self.dq0_riferimento)
        return self.q - riferimento

    def salva_csv(self, percorso: str | Path) -> Path:
        """Esporta le storie temporali in CSV."""
        percorso = Path(percorso)
        percorso.parent.mkdir(parents=True, exist_ok=True)
        dqnl = self.dq_nonlineare
        with percorso.open("w", encoding="utf-8", newline="") as f:
            w = csv.writer(f)
            intestazione = [
                "t_s", "psi_deg", "theta_deg", "dpsi_deg_s", "dtheta_deg_s",
                "tau_psi_N_m", "tau_theta_N_m",
                "dpsi_nl_deg", "dtheta_nl_deg",
            ]
            if self.dq_lin is not None:
                intestazione += ["dpsi_lin_deg", "dtheta_lin_deg"]
            w.writerow(intestazione)
            for k, tk in enumerate(self.t):
                riga = [
                    f"{tk:.10g}",
                    f"{np.rad2deg(self.q[k, 0]):.10g}", f"{np.rad2deg(self.q[k, 1]):.10g}",
                    f"{np.rad2deg(self.dq[k, 0]):.10g}", f"{np.rad2deg(self.dq[k, 1]):.10g}",
                    f"{self.tau[k, 0]:.10g}", f"{self.tau[k, 1]:.10g}",
                    f"{np.rad2deg(dqnl[k, 0]):.10g}", f"{np.rad2deg(dqnl[k, 1]):.10g}",
                ]
                if self.dq_lin is not None:
                    riga += [
                        f"{np.rad2deg(self.dq_lin[k, 0]):.10g}",
                        f"{np.rad2deg(self.dq_lin[k, 1]):.10g}",
                    ]
                w.writerow(riga)
        return percorso


# ---------------------------------------------------------------------------
# Strumenti comuni agli scenari
# ---------------------------------------------------------------------------


def orizzonte_sicuro(
    ml: ModelloLineare, t_nominale: float, *, crescita_massima: float = 100.0
) -> tuple[float, tuple[str, ...]]:
    """Accorcia l'orizzonte se il punto di lavoro e' instabile ad anello aperto.

    Con un polo reale positivo ``lambda_u`` la perturbazione cresce come
    ``exp(lambda_u t)``: oltre un certo tempo il modello non lineare esce dal
    regime di validita' della linearizzazione e il confronto perde senso. Si
    limita l'orizzonte al tempo in cui la crescita raggiunge
    ``crescita_massima``.
    """
    parti_reali = np.real(ml.poli)
    lam = float(np.max(parti_reali))
    if lam <= 1e-9:
        return t_nominale, ()
    t_limite = float(np.log(crescita_massima) / lam)
    if t_limite >= t_nominale:
        return t_nominale, ()
    nota = (
        f"punto di lavoro instabile ad anello aperto (polo reale +{lam:.4g} rad/s, "
        f"costante di tempo {1 / lam:.4g} s): orizzonte ridotto da {t_nominale:g} s "
        f"a {t_limite:.4g} s, il tempo in cui una perturbazione cresce di un "
        f"fattore {crescita_massima:g}."
    )
    return t_limite, (nota,)


def _ampiezza_calibrata(
    ml: ModelloLineare,
    t: np.ndarray,
    canale: int,
    coordinata: int,
    *,
    bersaglio_deg: float,
    t_inizio: float,
    durata: float,
) -> float:
    """Ampiezza del doppietto che porta la coordinata al bersaglio, sul lineare.

    Si sfrutta la linearita': si calcola la risposta a un doppietto unitario e
    si riscala. Cosi' il vincolo della specifica (|delta psi| < 1 deg) e'
    soddisfatto per costruzione e il confronto resta nel regime lineare.
    """
    dtau = np.zeros((t.size, 2))
    dtau[:, canale] = doppietto(t, t_inizio=t_inizio, durata=durata, ampiezza=1.0)
    dq_lin, _ = risposta_lineare(ml, t, dtau)
    picco = float(np.max(np.abs(dq_lin[:, coordinata])))
    if picco <= 0.0:
        raise ValueError(
            "la risposta a un doppietto unitario e' nulla: impossibile calibrare "
            "l'ampiezza (canale disaccoppiato in questo punto di lavoro)"
        )
    return float(np.deg2rad(bersaglio_deg) / picco)


def _errori_relativi(
    dq_nl: np.ndarray, dq_lin: np.ndarray, *, coordinata_guidata: int | None = None
) -> dict[str, float]:
    """Errore relativo fra perturbazione non lineare e lineare, per coordinata.

    Normalizzato sul picco della risposta non lineare della coordinata stessa;
    se quella risposta e' trascurabile, sulla norma complessiva.

    Se ``coordinata_guidata`` e' data, si aggiungono le chiavi ``"guidata"`` e
    ``"incrociata"``. La distinzione e' essenziale: la coordinata **guidata**
    risponde al primo ordine nell'ampiezza della perturbazione, quindi il suo
    errore misura davvero la qualita' della linearizzazione. La coordinata
    **incrociata** puo' invece rispondere prevalentemente al **secondo**
    ordine (il termine centrifugo ``-1/2 J_psi' dpsi^2`` scompare nella
    linearizzazione quando ``Omega_0 = 0``), e allora il suo errore relativo
    resta finito per ampiezza finita e tende a zero solo linearmente con
    l'ampiezza: e' l'oggetto di :func:`convergenza_lineare`.
    """
    errori: dict[str, float] = {}
    scala_globale = max(float(np.max(np.abs(dq_nl))), 1e-30)
    for i, nome in enumerate(("psi", "theta")):
        scala = float(np.max(np.abs(dq_nl[:, i])))
        riferimento = scala if scala > 1e-3 * scala_globale else scala_globale
        errori[nome] = float(np.max(np.abs(dq_nl[:, i] - dq_lin[:, i]))) / riferimento
    errori["massimo"] = max(errori["psi"], errori["theta"])
    if coordinata_guidata is not None:
        nomi = ("psi", "theta")
        errori["guidata"] = errori[nomi[coordinata_guidata]]
        errori["incrociata"] = errori[nomi[1 - coordinata_guidata]]
    return errori


def modello_conservativo(modello: ModelloInerziale) -> tuple[ModelloInerziale, ModelloNumerico]:
    """Copia del modello con smorzamenti nulli, per la verifica conservativa."""
    par0 = replace(modello.par, b_psi=0.0, b_theta=0.0)
    mod0 = replace(modello, par=par0)
    return mod0, lg.costruisci_modello_numerico(mod0)


# ---------------------------------------------------------------------------
# Scenari
# ---------------------------------------------------------------------------


def scenario_doppietto(
    modello: ModelloInerziale,
    *,
    nome: str,
    descrizione: str,
    theta0: float,
    Omega0: float,
    canale: int,
    coordinata: int,
    bersaglio_deg: float,
    t_finale: float = 20.0,
    n_punti: int = 4001,
    durata_doppietto: float | None = None,
) -> RisultatoScenario:
    """Scenario generico a doppietto con confronto lineare/non lineare.

    Copre S1 (``canale = 0``), S2 (``canale = 1``) e, con ``Omega0 != 0``, S3.

    Parameters
    ----------
    canale
        Coppia su cui agisce il doppietto: 0 per ``tau_psi``, 1 per ``tau_theta``.
    coordinata
        Coordinata su cui si calibra l'ampiezza: 0 per ``psi``, 1 per ``theta``.
    bersaglio_deg
        Ampiezza di picco desiderata per quella coordinata, in gradi.
    """
    mn = lg.costruisci_modello_numerico(modello)
    ml = lz.linearizza(modello, theta0, Omega0)

    t_finale, note = orizzonte_sicuro(ml, t_finale)
    t = np.linspace(0.0, t_finale, n_punti)
    passo = float(t[1] - t[0])

    # I fronti del doppietto vengono allineati alla griglia temporale. Motivo:
    # scipy.signal.lsim assume l'ingresso **interpolato linearmente** fra i
    # campioni, quindi un fronte che cade fra due campioni verrebbe visto dai
    # due modelli in modo diverso e produrrebbe uno scarto O(passo)
    # indipendente dall'ampiezza, che maschererebbe l'errore di
    # linearizzazione che si vuole misurare.
    durata = durata_doppietto if durata_doppietto is not None else max(t_finale / 20.0, 1e-3)
    durata = max(round(durata / passo), 1) * passo
    t_inizio = max(round((t_finale / 10.0) / passo), 1) * passo

    ampiezza = _ampiezza_calibrata(
        ml, t, canale, coordinata,
        bersaglio_deg=bersaglio_deg, t_inizio=t_inizio, durata=durata,
    )

    # Perturbazione di coppia e coppia totale (feedforward + perturbazione).
    dtau = np.zeros((t.size, 2))
    dtau[:, canale] = doppietto(t, t_inizio=t_inizio, durata=durata, ampiezza=ampiezza)
    tau0 = mn.coppie_nominali(theta0, Omega0)
    tau_totale = tau0[None, :] + dtau

    # Il modello non lineare riceve esattamente lo stesso ingresso che vede
    # lsim, cioe' l'interpolazione lineare dei campioni di dtau: la differenza
    # fra le due risposte e' allora imputabile alla sola non linearita'.
    campioni_canale = dtau[:, canale]

    def tau_di_t(tempo: float) -> np.ndarray:
        d = np.zeros(2)
        d[canale] = float(np.interp(tempo, t, campioni_canale))
        return tau0 + d

    q0 = np.array([0.0, theta0])
    dq0 = np.array([Omega0, 0.0])
    q, dq = integra_nonlineare(mn, t, q0, dq0, tau_di_t)
    dq_lin, ddq_lin = risposta_lineare(ml, t, dtau)

    risultato = RisultatoScenario(
        nome=nome,
        descrizione=descrizione,
        t=t,
        q=q,
        dq=dq,
        tau=tau_totale,
        q0=q0,
        dq0_riferimento=dq0,
        dq_lin=dq_lin,
        ddq_lin=ddq_lin,
        note=note,
    )
    errori = _errori_relativi(
        risultato.dq_nonlineare, dq_lin, coordinata_guidata=coordinata
    )
    diagnostica = {
        "ampiezza_doppietto_N_m": ampiezza,
        "durata_doppietto_s": durata,
        "t_finale_s": t_finale,
        "picco_dpsi_deg": float(np.rad2deg(np.max(np.abs(risultato.dq_nonlineare[:, 0])))),
        "picco_dtheta_deg": float(np.rad2deg(np.max(np.abs(risultato.dq_nonlineare[:, 1])))),
        "tau_psi0_N_m": float(tau0[0]),
        "tau_theta0_N_m": float(tau0[1]),
        "gamma": ml.gamma,
        "k_theta": ml.k_theta,
    }
    # Il criterio dell'1 % della specifica si applica alla coordinata guidata,
    # che risponde al primo ordine nell'ampiezza. Sulla coordinata incrociata
    # l'errore relativo e' dominato dal secondo ordine e viene riportato a
    # parte, con la verifica di convergenza a fare da controllo quantitativo.
    esito: Literal["ok", "attenzione"] = "ok" if errori["guidata"] < 0.01 else "attenzione"
    note_finali = list(note)
    if errori["guidata"] >= 0.01:
        note_finali.append(
            f"errore relativo sulla coordinata guidata {errori['guidata'] * 100:.2f} %, "
            "oltre la soglia dell'1 % richiesta dalla specifica."
        )
    if errori["incrociata"] >= 0.01:
        note_finali.append(
            f"errore relativo sulla coordinata incrociata {errori['incrociata'] * 100:.1f} %: "
            "atteso, perche' con questa ampiezza la risposta incrociata e' dominata dal "
            "termine centrifugo -1/2 J_psi'(theta_0) dpsi^2, che e' quadratico nella "
            "perturbazione e quindi assente dal modello linearizzato. "
            "Vedi la verifica di convergenza: l'errore scala linearmente con l'ampiezza."
        )
    return replace(
        risultato, errori=errori, diagnostica=diagnostica, esito=esito, note=tuple(note_finali)
    )


def S1(modello: ModelloInerziale, **kwargs: object) -> RisultatoScenario:
    """S1: theta_0 = -45 deg, Omega_0 = 0, doppietto su ``tau_psi``.

    Ampiezza calibrata perche' ``|delta psi| < 1 deg``. Confronto con il
    modello lineare: errore relativo atteso sotto l'1 %.
    """
    return scenario_doppietto(
        modello,
        nome="S1",
        descrizione=(
            "theta_0 = -45 deg, Omega_0 = 0, doppietto su tau_psi con "
            "|delta psi| < 1 deg; confronto lineare/non lineare"
        ),
        theta0=float(np.deg2rad(-45.0)),
        Omega0=0.0,
        canale=0,
        coordinata=0,
        bersaglio_deg=0.5,
        **kwargs,  # type: ignore[arg-type]
    )


def S2(modello: ModelloInerziale, **kwargs: object) -> RisultatoScenario:
    """S2: come S1 ma con il doppietto su ``tau_theta``.

    Dove la rigidezza equivalente ``k_theta`` e' negativa l'asse di elevazione
    e' instabile ad anello aperto (pendolo inverso da sbilanciamento):
    l'orizzonte viene accorciato automaticamente e la cosa viene segnalata.
    Il segno di ``k_theta`` dipende da entrambi gli offset del baricentro,
    ``k_theta`` gravitazionale ``= m_B g (d_x sin(theta) - d_z cos(theta))``,
    non dal solo ``d_z``.
    """
    return scenario_doppietto(
        modello,
        nome="S2",
        descrizione=(
            "theta_0 = -45 deg, Omega_0 = 0, doppietto su tau_theta; "
            "confronto lineare/non lineare"
        ),
        theta0=float(np.deg2rad(-45.0)),
        Omega0=0.0,
        canale=1,
        coordinata=1,
        bersaglio_deg=0.5,
        **kwargs,  # type: ignore[arg-type]
    )


def S3(
    modello: ModelloInerziale, *, Omega0: float | None = None, **kwargs: object
) -> RisultatoScenario:
    """S3: inseguimento a Omega_0 massima con feedforward e piccolo doppietto.

    Mostra l'accoppiamento giroscopico su ``theta`` e lo confronta con il
    modello lineare nel punto ``(theta_0, Omega_0)``.
    """
    om = modello.par.omega0_max if Omega0 is None else float(Omega0)
    return scenario_doppietto(
        modello,
        nome="S3",
        descrizione=(
            f"inseguimento a Omega_0 = {np.rad2deg(om):g} deg/s con coppie nominali "
            "in feedforward e doppietto su tau_psi; accoppiamento su theta"
        ),
        theta0=float(np.deg2rad(-45.0)),
        Omega0=om,
        canale=0,
        coordinata=0,
        bersaglio_deg=0.5,
        **kwargs,  # type: ignore[arg-type]
    )


def S4(
    modello: ModelloInerziale,
    *,
    theta0: float = float(np.deg2rad(-60.0)),
    dpsi0: float = float(np.deg2rad(20.0)),
    dtheta0: float = float(np.deg2rad(10.0)),
    t_finale: float = 20.0,
    n_punti: int = 4001,
) -> RisultatoScenario:
    """S4: verifica conservativa con ``b = 0``, ``tau = 0``, condizioni ampie.

    Energia ``E = T + V`` e momento ``p_psi = dL/d(dpsi)`` devono conservarsi
    con deriva relativa sotto ``1e-8``. E' il controllo piu' severo sulla
    correttezza delle equazioni: qualunque errore di segno o di derivazione
    rompe la conservazione.
    """
    mod0, mn0 = modello_conservativo(modello)
    t = np.linspace(0.0, t_finale, n_punti)
    q0 = np.array([0.0, theta0])
    dq0 = np.array([dpsi0, dtheta0])

    def tau_nullo(_: float) -> np.ndarray:
        return np.zeros(2)

    q, dq = integra_nonlineare(mn0, t, q0, dq0, tau_nullo)

    E = np.array([mn0.energia(q[k], dq[k]) for k in range(t.size)])
    p = np.array([mn0.momento_psi(q[k], dq[k]) for k in range(t.size)])
    scala_E = max(float(np.max(np.abs(E))), 1e-30)
    scala_p = max(float(np.max(np.abs(p))), 1e-30)
    deriva_E = float(np.max(np.abs(E - E[0]))) / scala_E
    deriva_p = float(np.max(np.abs(p - p[0]))) / scala_p

    note: list[str] = []
    esito: Literal["ok", "attenzione"] = "ok"
    if max(deriva_E, deriva_p) > 1e-8:
        esito = "attenzione"
        note.append(
            f"deriva relativa oltre 1e-8: energia {deriva_E:.2e}, momento {deriva_p:.2e}."
        )
    if modello.par.b_psi != 0.0 or modello.par.b_theta != 0.0:
        note.append(
            "gli smorzamenti sono stati posti a zero solo per questo scenario: "
            "con b != 0 energia e momento non si conservano per definizione."
        )

    return RisultatoScenario(
        nome="S4",
        descrizione=(
            "verifica conservativa: b = 0, tau = 0, condizioni iniziali ampie; "
            "conservazione di E = T + V e di p_psi"
        ),
        t=t,
        q=q,
        dq=dq,
        tau=np.zeros((t.size, 2)),
        q0=q0,
        dq0_riferimento=np.zeros(2),
        diagnostica={
            "deriva_relativa_energia": deriva_E,
            "deriva_relativa_momento": deriva_p,
            "E_iniziale_J": float(E[0]),
            "p_psi_iniziale_N_m_s": float(p[0]),
            "theta_min_deg": float(np.rad2deg(np.min(q[:, 1]))),
            "theta_max_deg": float(np.rad2deg(np.max(q[:, 1]))),
            "t_finale_s": t_finale,
        },
        note=tuple(note),
        esito=esito,
    )


def S5(
    modello: ModelloInerziale,
    *,
    theta0: float = float(np.deg2rad(-45.0)),
    ampiezza_slew_deg: float = 90.0,
    t_slew: float = 10.0,
    n_punti: int = 4001,
) -> RisultatoScenario:
    """S5 (facoltativa): slew di 90 deg in azimut con coppia bang-bang.

    Serve a mostrare dove la predizione lineare smette di valere: la manovra
    e' ampia, quindi ``J_psi(theta)`` e l'accoppiamento variano lungo il moto.
    """
    mn = lg.costruisci_modello_numerico(modello)
    ml = lz.linearizza(modello, theta0, 0.0)

    t_finale = t_slew * 1.5
    t = np.linspace(0.0, t_finale, n_punti)

    # Coppia per uno slew rigido di ampiezza data in tempo t_slew, a inerzia
    # costante: Delta psi = (A / J) (t_slew / 2)^2.
    J = float(modello.J_psi(theta0))
    ampiezza = 4.0 * J * float(np.deg2rad(ampiezza_slew_deg)) / t_slew**2

    tau0 = mn.coppie_nominali(theta0, 0.0)

    dtau = np.zeros((t.size, 2))
    dtau[:, 0] = bang_bang(t, t_inizio=0.0, durata=t_slew, ampiezza=ampiezza)
    campioni = dtau[:, 0]

    def tau_di_t(tempo: float) -> np.ndarray:
        # Come negli altri scenari: stesso ingresso visto da lsim.
        d = np.zeros(2)
        d[0] = float(np.interp(tempo, t, campioni))
        return tau0 + d

    q0 = np.array([0.0, theta0])
    dq0 = np.zeros(2)
    q, dq = integra_nonlineare(mn, t, q0, dq0, tau_di_t)
    dq_lin, ddq_lin = risposta_lineare(ml, t, dtau)

    note_S5 = [
        "lo scarto rispetto al modello lineare e' atteso e voluto: la manovra "
        "e' ampia, quindi J_psi(theta) e l'accoppiamento variano lungo il moto."
    ]
    risultato = RisultatoScenario(
        nome="S5",
        descrizione=(
            f"slew di {ampiezza_slew_deg:g} deg in azimut con coppia bang-bang in "
            f"{t_slew:g} s: limite di validita' della predizione lineare"
        ),
        t=t,
        q=q,
        dq=dq,
        tau=tau0[None, :] + dtau,
        q0=q0,
        dq0_riferimento=np.zeros(2),
        dq_lin=dq_lin,
        ddq_lin=ddq_lin,
    )
    errori = _errori_relativi(risultato.dq_nonlineare, dq_lin, coordinata_guidata=0)
    escursione = float(np.rad2deg(np.ptp(q[:, 1])))
    theta_min_deg = float(np.rad2deg(modello.par.theta0_grid.min()))
    theta_max_deg = float(np.rad2deg(modello.par.theta0_grid.max()))
    larghezza_campo = theta_max_deg - theta_min_deg
    if escursione > larghezza_campo:
        note_S5.append(
            f"durante lo slew theta esce dal campo operativo "
            f"[{theta_min_deg:.0f}, {theta_max_deg:.0f}] deg configurato "
            f"(escursione {escursione:.0f} deg): con lo sbilanciamento corrente "
            "l'asse di elevazione, non controllato, si rovescia. Gli errori "
            "relativi rispetto al modello lineare non sono significativi in "
            "questo scenario e servono solo a documentare la rottura della "
            "predizione lineare."
        )
    return replace(
        risultato,
        errori=errori,
        diagnostica={
            "ampiezza_coppia_N_m": ampiezza,
            "psi_finale_deg": float(np.rad2deg(q[-1, 0])),
            "theta_escursione_deg": float(np.rad2deg(np.ptp(q[:, 1]))),
            "t_finale_s": t_finale,
            "J_psi_iniziale_kg_m2": J,
        },
        note=tuple(note_S5),
        esito="ok",
    )


# ---------------------------------------------------------------------------
# Verifica di convergenza della linearizzazione
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ConvergenzaLineare:
    """Convergenza del modello lineare a quello non lineare per ampiezza -> 0.

    E' la verifica quantitativa che la linearizzazione sia corretta, piu'
    informativa di un singolo confronto a tolleranza fissa. Se le matrici
    ``M_0``, ``C_0``, ``K_0`` sono giuste, il residuo del modello lineare e'
    ``O(ampiezza^2)``, quindi l'**errore relativo** e' ``O(ampiezza)``: in
    scala logaritmica la pendenza vale 1. Una pendenza prossima a 0 segnalerebbe
    un errore nelle matrici; una pendenza 2 un termine lineare mancante.

    Attributes
    ----------
    bersagli_deg
        Ampiezze di picco imposte alla coordinata guidata, in gradi.
    errore_guidata, errore_incrociata
        Errori relativi corrispondenti.
    pendenza_guidata, pendenza_incrociata
        Pendenze della regressione lineare di ``log(errore)`` su
        ``log(ampiezza)``, cioe' l'ordine di convergenza osservato.
    """

    scenario: str
    bersagli_deg: np.ndarray
    errore_guidata: np.ndarray
    errore_incrociata: np.ndarray
    pendenza_guidata: float
    pendenza_incrociata: float
    banda_guidata: np.ndarray
    banda_incrociata: np.ndarray

    @property
    def errore_minimo_totale(self) -> float:
        """Errore massimo fra le due coordinate all'ampiezza piu' piccola."""
        return float(max(self.errore_guidata[-1], self.errore_incrociata[-1]))


SATURAZIONE_CONVERGENZA: float = 0.5
"""Errore relativo oltre il quale la stima dell'ordine e' inaffidabile.

Sopra questa soglia la perturbazione non e' piu' piccola e l'errore relativo
satura verso l'unita': includere quei punti nella regressione abbasserebbe
artificialmente la pendenza.
"""

FATTORE_MINIMO_CONVERGENZA: float = 4.0
"""Riduzione minima dell'errore attesa a ogni raffinamento dell'ampiezza.

Serve a riconoscere il **pavimento numerico**. Con ampiezze che scalano di un
fattore 10 e convergenza del primo ordine l'errore deve calare di circa 10 a
ogni passo; quando invece smette di calare (o risale), si e' raggiunta la
tolleranza dell'integratore e i punti successivi non dicono piu' nulla
sull'ordine di convergenza.
"""


def _ordine_convergenza(
    ampiezze: np.ndarray,
    errori: np.ndarray,
    *,
    saturazione: float = SATURAZIONE_CONVERGENZA,
    fattore_minimo: float = FATTORE_MINIMO_CONVERGENZA,
) -> tuple[float, np.ndarray]:
    """Ordine di convergenza stimato sulla sola banda utile.

    La serie di errori ha, ai due estremi, due regimi in cui la pendenza non
    significa nulla:

    * alle **ampiezze grandi** l'errore relativo satura verso l'unita', perche'
      la risposta non lineare non somiglia piu' a quella lineare;
    * alle **ampiezze piccole** l'errore si appiattisce sul pavimento numerico
      dell'integratore (``rtol = 1e-10`` su risposte di frazioni di millesimo
      di grado).

    Si tiene quindi il tratto centrale: si parte dalla prima ampiezza non
    satura e si prosegue finche' l'errore continua a calare di almeno
    ``fattore_minimo`` a ogni raffinamento.

    Returns
    -------
    (pendenza, banda)
        La pendenza e la maschera booleana dei punti usati.
    """
    ampiezze = np.asarray(ampiezze, dtype=float)
    errori = np.asarray(errori, dtype=float)
    banda = np.zeros_like(errori, dtype=bool)

    utilizzabili = np.nonzero((errori > 0.0) & (errori < saturazione))[0]
    if utilizzabili.size >= 2:
        indici = [int(utilizzabili[0])]
        for i in range(indici[0] + 1, errori.size):
            if errori[i] > 0.0 and errori[i] <= errori[i - 1] / fattore_minimo:
                indici.append(i)
            else:
                break
        if len(indici) < 2:
            # Il tratto pulito e' troppo corto: si ripiega sulle due ampiezze
            # piu' grandi fra quelle non sature, le piu' lontane dal pavimento.
            indici = [int(k) for k in utilizzabili[:2]]
        banda[indici] = True
    else:
        non_nulli = np.nonzero(errori > 0.0)[0]
        if non_nulli.size < 2:
            return float("nan"), banda
        banda[non_nulli[-2:]] = True

    pendenza = float(
        np.polyfit(np.log(ampiezze[banda]), np.log(errori[banda]), 1)[0]
    )
    return pendenza, banda


def convergenza_lineare(
    modello: ModelloInerziale,
    *,
    theta0: float = float(np.deg2rad(-45.0)),
    Omega0: float = 0.0,
    canale: int = 0,
    coordinata: int = 0,
    bersagli_deg: Sequence[float] = (0.5, 0.05, 0.005, 0.0005),
    n_punti: int = 2001,
    nome: str = "S1",
) -> ConvergenzaLineare:
    """Ripete lo scenario a doppietto ad ampiezze decrescenti.

    Restituisce l'ordine di convergenza osservato: atteso 1 per entrambe le
    coordinate, perche' l'errore relativo di una linearizzazione corretta e'
    proporzionale all'ampiezza della perturbazione.
    """
    bersagli = np.asarray(bersagli_deg, dtype=float)
    e_guidata = np.empty(bersagli.size)
    e_incrociata = np.empty(bersagli.size)

    for i, bersaglio in enumerate(bersagli):
        r = scenario_doppietto(
            modello,
            nome=f"{nome}-conv",
            descrizione=f"convergenza, bersaglio {bersaglio:g} deg",
            theta0=theta0,
            Omega0=Omega0,
            canale=canale,
            coordinata=coordinata,
            bersaglio_deg=float(bersaglio),
            n_punti=n_punti,
        )
        e_guidata[i] = r.errori["guidata"]
        e_incrociata[i] = r.errori["incrociata"]

    pendenza_g, banda_g = _ordine_convergenza(bersagli, e_guidata)
    pendenza_i, banda_i = _ordine_convergenza(bersagli, e_incrociata)

    return ConvergenzaLineare(
        scenario=nome,
        bersagli_deg=bersagli,
        errore_guidata=e_guidata,
        errore_incrociata=e_incrociata,
        pendenza_guidata=pendenza_g,
        pendenza_incrociata=pendenza_i,
        banda_guidata=banda_g,
        banda_incrociata=banda_i,
    )


def esegui_tutti(
    modello: ModelloInerziale, *, rapido: bool = False
) -> dict[str, RisultatoScenario]:
    """Esegue gli scenari della sezione 8.4.

    Parameters
    ----------
    rapido
        Se ``True`` riduce il numero di punti temporali e salta S5.
    """
    n = 1001 if rapido else 4001
    scenari: dict[str, RisultatoScenario] = {
        "S1": S1(modello, n_punti=n),
        "S2": S2(modello, n_punti=n),
        "S3": S3(modello, n_punti=n),
        "S4": S4(modello, n_punti=n, t_finale=10.0 if rapido else 20.0),
    }
    if not rapido:
        scenari["S5"] = S5(modello, n_punti=n)
    return scenari
