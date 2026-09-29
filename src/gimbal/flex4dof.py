"""Estensione flessibile a 4 GdL (sezione 10), attivata con ``flex_model: true``.

Coordinate ``q = [psi_m, psi, theta_m, theta]``: i rotori dei motori
(``psi_m``, ``theta_m``) sono collegati alle rispettive strutture da molle
torsionali. E' il primo modello della catena che possa contenere modi
**strutturali**, cioe' risonanze e antirisonanze vere.

.. warning::
   I parametri ``J_m_psi``, ``J_m_theta`` (dai datasheet dei motori) e ``k``,
   ``c`` (da FEM o da identificazione sperimentale) sono **segnaposto**.
   Finche' non sono forniti, il modulo **rifiuta** di produrre risultati:
   presentare frequenze calcolate su rigidezze inventate come frequenze
   proprie della montatura sarebbe un errore di metodo, non un'imprecisione.

Le inerzie dei rotori vanno tolte da quelle dei corpi rigidi per non contarle
due volte:

.. math:: J_\\psi^{s}(\\theta) = J_\\psi(\\theta) - J_{m\\psi}, \\qquad
          I_{yy}^{s} = I_{yy} - J_{m\\theta}

Il contributo del rotore di elevazione all'inerzia di azimut resta, in
approssimazione, dentro ``J_psi^s``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final, Sequence

import numpy as np
import sympy as sp

from .inertia import ModelloInerziale
from .params import ParametriFlex

# Ordine delle coordinate generalizzate.
COORDINATE: Final[tuple[str, ...]] = ("psi_m", "psi", "theta_m", "theta")
INDICE_ROTORI: Final[tuple[int, int]] = (0, 2)
INDICE_STRUTTURE: Final[tuple[int, int]] = (1, 3)

MESSAGGIO_PARAMETRI_MANCANTI: Final[str] = (
    "Modello flessibile a 4 GdL non calcolabile: mancano i parametri {mancanti}.\n"
    "  J_m_psi, J_m_theta  -> dai datasheet dei motori frameless\n"
    "  k_psi, k_theta      -> da analisi FEM oppure da identificazione sperimentale\n"
    "  c_psi, c_theta      -> da identificazione sperimentale\n"
    "Compilare la sezione 'flex' di config.yaml. Finche' i valori non sono reali\n"
    "il modulo non produce frequenze: una risonanza calcolata su una rigidezza\n"
    "inventata non e' un risultato, e non va ne' pubblicata ne' citata."
)


class ParametriFlexMancanti(RuntimeError):
    """I parametri del modello flessibile non sono stati forniti."""


@dataclass(frozen=True)
class ModelloFlex:
    """Modello flessibile a 4 GdL linearizzato in un punto di lavoro.

    Attributes
    ----------
    M, C, K
        Matrici 4x4 nell'ordine ``[psi_m, psi, theta_m, theta]``.
    Bq
        Matrice 4x2 di distribuzione delle coppie: ``tau_psi`` agisce su
        ``psi_m``, ``tau_theta`` su ``theta_m``.
    segnaposto
        ``True`` se i parametri usati non provengono da misure: in tal caso
        ogni frequenza va etichettata come non fisica.
    """

    theta0: float
    M: np.ndarray
    C: np.ndarray
    K: np.ndarray
    Bq: np.ndarray
    flex: ParametriFlex
    modello_rigido: ModelloInerziale
    segnaposto: bool

    # -- funzioni di trasferimento -----------------------------------------

    def P(self, s_val: complex) -> np.ndarray:
        """``P(s) = M s^2 + C s + K``, 4x4."""
        return self.M * s_val**2 + self.C * s_val + self.K

    def _G(self, omega: np.ndarray, righe: Sequence[int]) -> np.ndarray:
        omega = np.atleast_1d(np.asarray(omega, dtype=float))
        jw = 1j * omega
        Pw = (
            self.M[None, :, :] * (jw**2)[:, None, None]
            + self.C[None, :, :] * jw[:, None, None]
            + self.K[None, :, :]
        )
        G = np.linalg.solve(Pw, np.broadcast_to(self.Bq, (omega.size, 4, 2)))
        return G[:, list(righe), :]

    def frf_rotori(self, omega: np.ndarray) -> np.ndarray:
        """FRF coppia -> angolo dei rotori: e' cio' che misura l'encoder.

        Matrice 2x2 **collocata**: ingresso e uscita stanno sullo stesso corpo.
        """
        return self._G(omega, INDICE_ROTORI)

    def frf_strutture(self, omega: np.ndarray) -> np.ndarray:
        """FRF coppia -> angolo delle strutture: il lato carico, il puntamento."""
        return self._G(omega, INDICE_STRUTTURE)

    # -- poli, zeri, frequenze ---------------------------------------------

    @property
    def poli(self) -> np.ndarray:
        """Autovalori del sistema del secondo ordine, via linearizzazione."""
        Minv = np.linalg.inv(self.M)
        A = np.block(
            [
                [np.zeros((4, 4)), np.eye(4)],
                [-Minv @ self.K, -Minv @ self.C],
            ]
        )
        return np.linalg.eigvals(A)

    def frequenze_proprie(self) -> np.ndarray:
        """Frequenze proprie non smorzate, in Hz, ordinate."""
        autovalori = np.linalg.eigvals(np.linalg.solve(self.M, self.K))
        omega2 = np.real(autovalori)
        omega = np.sqrt(np.clip(omega2, 0.0, None))
        return np.sort(omega) / (2.0 * np.pi)

    def frequenze_rotore_bloccato(self) -> np.ndarray:
        """Frequenze proprie con i rotori bloccati, in Hz.

        Si eliminano le righe e colonne dei rotori: restano le sole strutture,
        vincolate alle molle. Devono coincidere con le **antirisonanze** delle
        FRF collocate (sezione 10, terzo test).
        """
        liberi = list(INDICE_STRUTTURE)
        M = self.M[np.ix_(liberi, liberi)]
        K = self.K[np.ix_(liberi, liberi)]
        omega2 = np.real(np.linalg.eigvals(np.linalg.solve(M, K)))
        return np.sort(np.sqrt(np.clip(omega2, 0.0, None))) / (2.0 * np.pi)

    def zeri_collocati(self) -> dict[str, np.ndarray]:
        """Antirisonanze delle due FRF collocate, in Hz.

        Con il modello della sezione 10 e ``Omega_0 = 0`` i due assi sono
        disaccoppiati, quindi ciascuna FdT collocata e' quella di una catena a
        due inerzie e una molla: lo zero e' la frequenza della struttura con il
        rotore bloccato.
        """
        risultato: dict[str, np.ndarray] = {}
        for nome, (i_rotore, i_struttura) in (
            ("azimut", (0, 1)),
            ("elevazione", (2, 3)),
        ):
            # Zero della FdT collocata = radice del minore complementare.
            M = self.M[i_struttura, i_struttura]
            C = self.C[i_struttura, i_struttura]
            K = self.K[i_struttura, i_struttura]
            radici = np.roots([M, C, K])
            risultato[nome] = np.sort(np.abs(radici)) / (2.0 * np.pi)
        return risultato


# ---------------------------------------------------------------------------
# Costruzione
# ---------------------------------------------------------------------------


def costruisci(
    modello: ModelloInerziale,
    theta0: float,
    *,
    flex: ParametriFlex | None = None,
    consenti_segnaposto: bool = False,
) -> ModelloFlex:
    """Costruisce il modello flessibile linearizzato in ``theta0``.

    Parameters
    ----------
    flex
        Parametri del modello flessibile; se ``None`` si usano quelli della
        configurazione.
    consenti_segnaposto
        Deve essere ``True`` perche' il modulo accetti parametri non misurati.
        E' riservato ai **test strutturali** della sezione 10 (interlacciamento,
        limite ``k -> infinito``, antirisonanze a rotore bloccato), che
        verificano proprieta' qualitative indipendenti dai valori numerici.
        I risultati ottenuti cosi' non vanno presentati come frequenze della
        montatura.

    Raises
    ------
    ParametriFlexMancanti
        Se i parametri non sono completi e ``consenti_segnaposto`` e' ``False``.
    """
    flex = flex if flex is not None else modello.par.flex
    if not flex.completo:
        if not consenti_segnaposto:
            raise ParametriFlexMancanti(
                MESSAGGIO_PARAMETRI_MANCANTI.format(mancanti=list(flex.mancanti()))
            )
        raise ParametriFlexMancanti(
            "consenti_segnaposto = True ma i parametri sono comunque incompleti: "
            f"mancano {list(flex.mancanti())}. Passare un ParametriFlex completo."
        )

    J_m_psi = float(flex.J_m_psi)  # type: ignore[arg-type]
    J_m_theta = float(flex.J_m_theta)  # type: ignore[arg-type]
    k_psi = float(flex.k_psi)  # type: ignore[arg-type]
    k_theta = float(flex.k_theta)  # type: ignore[arg-type]
    c_psi = float(flex.c_psi)  # type: ignore[arg-type]
    c_theta = float(flex.c_theta)  # type: ignore[arg-type]

    # Inerzie strutturali: si tolgono i rotori per non contarli due volte.
    J_psi_s = float(modello.J_psi(theta0)) - J_m_psi
    I_yy_s = modello.I_yy - J_m_theta
    if J_psi_s <= 0.0 or I_yy_s <= 0.0:
        raise ValueError(
            "inerzia strutturale non positiva dopo la sottrazione dei rotori: "
            f"J_psi^s = {J_psi_s:.6g}, I_yy^s = {I_yy_s:.6g}. "
            "Verificare J_m_psi e J_m_theta."
        )

    # Rigidezza gravitazionale della struttura di elevazione.
    k_g = float(modello.k_g(theta0))

    # Ordine: [psi_m, psi, theta_m, theta]
    M = np.diag([J_m_psi, J_psi_s, J_m_theta, I_yy_s])
    K = np.array(
        [
            [k_psi, -k_psi, 0.0, 0.0],
            [-k_psi, k_psi, 0.0, 0.0],
            [0.0, 0.0, k_theta, -k_theta],
            [0.0, 0.0, -k_theta, k_theta + k_g],
        ]
    )
    b_psi, b_theta = modello.par.b_psi, modello.par.b_theta
    C = np.array(
        [
            [c_psi + b_psi, -c_psi, 0.0, 0.0],
            [-c_psi, c_psi, 0.0, 0.0],
            [0.0, 0.0, c_theta + b_theta, -c_theta],
            [0.0, 0.0, -c_theta, c_theta],
        ]
    )
    Bq = np.array([[1.0, 0.0], [0.0, 0.0], [0.0, 1.0], [0.0, 0.0]])

    return ModelloFlex(
        theta0=float(theta0),
        M=M,
        C=C,
        K=K,
        Bq=Bq,
        flex=flex,
        modello_rigido=modello,
        segnaposto=consenti_segnaposto,
    )


def equazioni_simboliche() -> dict[str, sp.Expr | sp.Matrix]:
    """Deriva simbolicamente le equazioni del modello flessibile.

    Serve a documentare in tesi la struttura del modello anche quando i
    parametri numerici non sono disponibili: la derivazione e' esatta, sono i
    valori a mancare.
    """
    psi_m, psi, theta_m, theta = sp.symbols("psi_m psi theta_m theta", real=True)
    dpsi_m, dpsi, dtheta_m, dtheta = sp.symbols(
        "dpsi_m dpsi dtheta_m dtheta", real=True
    )
    J_m_psi, J_m_theta = sp.symbols("J_mpsi J_mtheta", positive=True)
    J_psi_s = sp.Function("J_psi_s")(theta)
    I_yy_s = sp.Symbol("I_yy_s", positive=True)
    k_psi, k_theta = sp.symbols("k_psi k_theta", positive=True)
    c_psi, c_theta = sp.symbols("c_psi c_theta", nonnegative=True)
    b_psi, b_theta = sp.symbols("b_psi b_theta", nonnegative=True)
    V_g = sp.Function("V_g")(theta)
    tau_psi, tau_theta = sp.symbols("tau_psi tau_theta", real=True)

    T = (
        sp.Rational(1, 2) * J_m_psi * dpsi_m**2
        + sp.Rational(1, 2) * J_psi_s * dpsi**2
        + sp.Rational(1, 2) * I_yy_s * dtheta**2
        + sp.Rational(1, 2) * J_m_theta * dtheta_m**2
    )
    V = (
        sp.Rational(1, 2) * k_psi * (psi_m - psi) ** 2
        + sp.Rational(1, 2) * k_theta * (theta_m - theta) ** 2
        + V_g
    )
    D = (
        sp.Rational(1, 2) * c_psi * (dpsi_m - dpsi) ** 2
        + sp.Rational(1, 2) * c_theta * (dtheta_m - dtheta) ** 2
        + sp.Rational(1, 2) * b_psi * dpsi_m**2
        + sp.Rational(1, 2) * b_theta * dtheta_m**2
    )
    q = sp.Matrix([psi_m, psi, theta_m, theta])
    dq = sp.Matrix([dpsi_m, dpsi, dtheta_m, dtheta])
    Q = sp.Matrix([tau_psi, 0, tau_theta, 0])

    return {
        "T": T,
        "V": V,
        "D": D,
        "q": q,
        "dq": dq,
        "Q": Q,
        "M": sp.hessian(T, list(dq)),
        "K": sp.hessian(V, list(q)),
        "C": sp.hessian(D, list(dq)),
    }


# ---------------------------------------------------------------------------
# Test strutturali della sezione 10
# ---------------------------------------------------------------------------


def verifica_interlacciamento(mf: ModelloFlex, *, tol: float = 1e-9) -> dict[str, bool]:
    """Poli e zeri delle FRF collocate non smorzate si alternano.

    E' la proprieta' caratteristica di una funzione di trasferimento
    collocata in un sistema meccanico passivo: fra due risonanze cade sempre
    un'antirisonanza. Vale solo se il sistema e' stabile, quindi per
    l'elevazione si richiede ``k_theta + k_g > 0``.
    """
    risultato: dict[str, bool] = {}
    for nome, (i_rotore, i_struttura) in (("azimut", (0, 1)), ("elevazione", (2, 3))):
        liberi = [i_rotore, i_struttura]
        M = mf.M[np.ix_(liberi, liberi)]
        K = mf.K[np.ix_(liberi, liberi)]
        if K[1, 1] <= 0.0:
            raise ValueError(
                f"{nome}: rigidezza della struttura non positiva "
                f"(k + k_g = {K[1, 1]:.6g}): il sistema e' instabile e "
                "l'interlacciamento non e' definito."
            )
        poli = np.sort(np.sqrt(np.clip(np.real(np.linalg.eigvals(np.linalg.solve(M, K))), 0.0, None)))
        zeri = np.array([np.sqrt(K[1, 1] / M[1, 1])])
        # Catena a due inerzie: poli 0 <= zero <= polo superiore.
        ordinato = bool(poli[0] <= zeri[0] + tol and zeri[0] <= poli[1] + tol)
        if not ordinato:
            raise AssertionError(
                f"{nome}: poli {poli} e zeri {zeri} non sono alternati"
            )
        risultato[nome] = ordinato
    return risultato


def verifica_antirisonanze(mf: ModelloFlex, *, tol: float = 1e-9) -> float:
    """Le antirisonanze collocate coincidono con le frequenze a rotore bloccato.

    Returns
    -------
    float
        Scarto relativo massimo fra le due serie di frequenze.
    """
    bloccato = mf.frequenze_rotore_bloccato()
    zeri = np.sort(
        np.concatenate([np.array([v[0]]) for v in mf.zeri_collocati().values()])
    )
    if bloccato.size != zeri.size:
        raise AssertionError(
            f"numero di frequenze diverso: bloccato {bloccato.size}, zeri {zeri.size}"
        )
    scala = max(float(np.max(bloccato)), 1e-30)
    scarto = float(np.max(np.abs(np.sort(bloccato) - zeri))) / scala
    if scarto > tol:
        raise AssertionError(
            f"antirisonanze {zeri} e frequenze a rotore bloccato {bloccato} "
            f"differiscono: scarto relativo {scarto:.3e}"
        )
    return scarto


def verifica_limite_rigido(
    modello: ModelloInerziale,
    theta0: float,
    flex: ParametriFlex,
    *,
    fattore: float = 1e2,
    omega: Sequence[float] | None = None,
    tol_diagonale: float = 1e-6,
    tol_previsione: float = 1e-6,
) -> dict[str, float]:
    """Nel limite ``k -> infinito`` si ritrovano le FdT del modello rigido.

    Si confrontano a bassa frequenza le FRF coppia -> angolo dei rotori del
    modello flessibile irrigidito con quelle del modello rigido a 2 GdL.

    **I due modelli non possono coincidere del tutto, e la differenza e'
    nota in anticipo.** L'energia cinetica della sezione 10 e' una somma di
    termini disaccoppiati, quindi il modello a 4 GdL non contiene il termine
    di accoppiamento inerziale ``M_12 = -I_xy sin(theta) + I_yz cos(theta)``,
    che invece il modello rigido ha sui fuori diagonale. Di conseguenza:

    * i termini **diagonali** devono convergere a quelli rigidi, e il test lo
      verifica con ``tol_diagonale``;
    * i termini **fuori diagonale** del modello flessibile sono esattamente
      nulli (``M``, ``C`` e ``K`` sono diagonali a blocchi), quindi il residuo
      deve coincidere con il fuori diagonale del **modello rigido stesso**.
      Il test confronta il residuo osservato con questa previsione invece di
      usare una soglia tarata a mano: cosi' verifica che la discrepanza sia
      *tutta e sola* quella attesa, qualunque sia la configurazione.

    Il fattore di irrigidimento non va preso troppo grande: il numero di
    condizione di ``P(j omega)`` cresce come ``k / (omega^2 J)`` e oltre circa
    ``1e15`` l'inversione perde ogni cifra significativa, facendo **risalire**
    lo scarto. Con ``fattore = 1e2`` le frequenze delle molle sono gia' tre
    decadi sopra la banda di confronto.

    Returns
    -------
    dict
        ``scarto_diagonale``, ``scarto_fuori_diagonale``,
        ``previsione_fuori_diagonale`` (tutti relativi) e ``scarto_previsione``.
    """
    from dataclasses import replace as _replace

    from . import linearize as lz

    rigido = _replace(
        flex,
        k_psi=float(flex.k_psi) * fattore,  # type: ignore[arg-type]
        k_theta=float(flex.k_theta) * fattore,  # type: ignore[arg-type]
    )
    mf = costruisci(modello, theta0, flex=rigido, consenti_segnaposto=True)
    ml = lz.linearizza(modello, theta0, 0.0)

    w = np.asarray(omega if omega is not None else np.logspace(-2, 0, 25) * 2 * np.pi)
    G_flex = mf.frf_rotori(w)
    G_rigido = ml.frf(w)

    scala = max(float(np.max(np.abs(G_rigido))), 1e-30)
    scarti = np.abs(G_flex - G_rigido) / scala
    diagonale = [(0, 0), (1, 1)]
    fuori = [(0, 1), (1, 0)]

    scarto_diagonale = float(max(np.max(scarti[:, i, j]) for i, j in diagonale))
    scarto_fuori = float(max(np.max(scarti[:, i, j]) for i, j in fuori))
    # Previsione: il modello flessibile ha fuori diagonale identicamente nullo,
    # quindi il residuo deve valere quanto il fuori diagonale del modello rigido.
    previsione = float(
        max(np.max(np.abs(G_rigido[:, i, j])) for i, j in fuori)
    ) / scala

    if scarto_diagonale > tol_diagonale:
        raise AssertionError(
            "il limite k -> infinito non riproduce i termini diagonali del "
            f"modello rigido: scarto relativo {scarto_diagonale:.3e} > "
            f"{tol_diagonale:.1e}"
        )

    scarto_previsione = abs(scarto_fuori - previsione) / max(previsione, 1e-30)
    if scarto_previsione > tol_previsione:
        raise AssertionError(
            f"il residuo sui fuori diagonale vale {scarto_fuori:.3e} ma "
            f"l'accoppiamento M_12 del modello rigido ne prevede {previsione:.3e} "
            f"(scarto relativo {scarto_previsione:.3e}): la discrepanza non e' "
            "spiegata dal solo termine mancante nella sezione 10."
        )

    return {
        "scarto_diagonale": scarto_diagonale,
        "scarto_fuori_diagonale": scarto_fuori,
        "previsione_fuori_diagonale": previsione,
        "scarto_previsione": scarto_previsione,
    }


def esegui(modello: ModelloInerziale, theta0: float) -> ModelloFlex | None:
    """Punto di ingresso usato da ``run_all.py`` quando ``flex_model: true``.

    Se i parametri mancano stampa un messaggio chiaro e restituisce ``None``,
    senza produrre numeri.
    """
    try:
        return costruisci(modello, theta0)
    except ParametriFlexMancanti as errore:
        print("=" * 78)
        print("MODELLO FLESSIBILE A 4 GdL (sezione 10)".center(78))
        print("=" * 78)
        print(str(errore))
        print("=" * 78)
        return None
