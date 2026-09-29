# Modello dinamico rigido a 2 GdL di un telescopio alt-azimutale

Modello a due corpi rigidi e due gradi di liberta' (azimut `psi`, elevazione
`theta`) di una montatura alt-azimutale a presa diretta. Implementa la
specifica `SPEC_modello_gimbal_2dof.md`, che si trova nella cartella
superiore: ogni risultato prodotto qui e' tracciabile alle equazioni di quel
documento.

Il codice deriva le equazioni di moto **simbolicamente** con SymPy a partire
da energia cinetica, energia potenziale e funzione di dissipazione: nessuna
equazione e' scritta a mano.

---

## Che cosa questo modello puo' e non puo' dire

Questo e' un modello a **corpi rigidi**. Non contiene flessibilita', quindi
**non contiene modi strutturali**: nessuna frequenza prodotta da questo codice
e' una frequenza propria della struttura, e nessuna va citata come tale.

Le uniche radici che il modello puo' produrre sono:

1. i **poli nulli** del moto rigido (`psi` e' una coordinata ciclica; con
   `b_psi = 0` l'asse di azimut e' un doppio integratore e il polo nullo e'
   doppio);
2. una **dinamica a bassa frequenza**, tipicamente sotto 1 Hz, dovuta a
   gravita' (sbilanciamento), accoppiamento giroscopico e rigidezza
   centrifuga.

Le risonanze e antirisonanze che interessano il controllo di puntamento
compaiono solo introducendo la cedevolezza dei giunti: e' quel che fa
l'estensione a 4 GdL di `flex4dof.py`, che pero' richiede rigidezze e inerzie
dei rotori **misurate**. Finche' non lo sono, quel modulo si rifiuta di
produrre frequenze.

---

## Come si esegue

### Ambiente

Serve Python 3.10 o successivo. Le dipendenze sono in `requirements.txt`.

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
```

Un ambiente virtuale gia' pronto e' presente in `.venv/`.

### Analisi completa

```bash
.venv/bin/python run_all.py
```

Produce tutto quanto elencato sotto, in `results/`. Dura qualche minuto.

### Opzioni della riga di comando

| opzione | effetto |
|---|---|
| `--config FILE` | usa un file di configurazione diverso da `config.yaml` |
| `--theta0 DEG [DEG ...]` | punti di lavoro in elevazione (default: `theta0_bode_deg`) |
| `--omega0 DEG_S [...]` | velocita' di inseguimento in azimut (default: `omega0_deg_s`) |
| `--output DIR` | cartella dei risultati (default: `results/`) |
| `--quick` | griglie e simulazioni ridotte: prova rapida, **non** risultati definitivi |
| `--no-figure` | salta la produzione delle figure |

Esempi:

```bash
.venv/bin/python run_all.py --quick
.venv/bin/python run_all.py --theta0 -45 --omega0 0 10
.venv/bin/python run_all.py --config prova.yaml --output results_prova
```

### Test

```bash
.venv/bin/python -m pytest
```

I test coprono i dodici punti della sezione 9 della specifica piu' i tre
test strutturali della sezione 10.

---

## Dove sono i risultati

```
results/
├── report.md                 # documento di sintesi: leggere prima questo
├── figures/                  # PDF vettoriale + PNG a 300 dpi
├── tables/                   # CSV
│   ├── inertia_summary.csv   # riepilogo inerziale
│   ├── poles_zeros_all.csv   # poli e zeri, una riga per radice
│   ├── poles_zeros/          # un CSV per combinazione (theta_0, Omega_0, d_z)
│   ├── sweep_theta0_*.csv    # scansioni parametriche in theta_0
│   ├── sweep_dz.csv          # scansione in d_z
│   └── scenario_S*.csv       # storie temporali delle simulazioni
└── latex/
    ├── eom.tex               # equazioni di moto
    └── tf.tex                # funzioni di trasferimento G_ij(s)
```

`report.md` raccoglie parametri usati, segnaposto, incoerenze rilevate,
riepilogo inerziale, esito dei controlli di riferimento, tabelle di poli e
zeri, esito delle simulazioni ed elenco delle figure.

---

## Struttura del codice

| modulo | contenuto | sezione della specifica |
|---|---|---|
| `src/gimbal/cad_data.py` | dati grezzi Onshape, immutabili | 3 |
| `src/gimbal/params.py` | lettura di `config.yaml`, conversione SI, terne | 4 |
| `src/gimbal/inertia.py` | preprocessing inerziale, Steiner, controlli fisici | 5 |
| `src/gimbal/kinematics.py` | cinematica simbolica, formula di Poisson | 6 |
| `src/gimbal/lagrange.py` | equazioni di Lagrange, forma matriciale, LaTeX | 6 |
| `src/gimbal/linearize.py` | linearizzazione, spazio di stato, FdT | 7 |
| `src/gimbal/analysis.py` | poli, zeri, analisi parametriche | 8.1, 8.2 |
| `src/gimbal/simulate.py` | scenari S1..S5 e convergenza | 8.4 |
| `src/gimbal/plotting.py` | figure | 8.2, 8.3 |
| `src/gimbal/report.py` | generazione di `report.md` | 11 |
| `src/gimbal/flex4dof.py` | estensione flessibile a 4 GdL | 10 |

---

## Convenzioni (sezione 2 della specifica, vincolanti)

- **Terne.** `W` e' la terna dell'assieme Onshape, inerziale, con `Z_W`
  verticale verso l'alto. `N` ha origine in `O`, intersezione degli assi.
  `A` e' la forcella, `B` il telescopio con `x_B` asse di puntamento e
  `y_B` asse di elevazione. Vale `R_NB = R_z(psi) R_y(theta)`.
- **Angoli.** `theta > 0` **abbassa** il puntamento: `theta = 0` e'
  l'orizzonte, `theta = -90 deg` lo zenit (dominio naturale della
  cinematica). Le **configurazioni operative** della montatura reale sono
  pero' `theta in [-45, 45] deg` (confermato dall'utente il 2026-09-27):
  il pitch spazia da 45 deg sopra l'orizzonte a 45 deg sotto, non fino allo
  zenit. L'elevazione fisica `epsilon = -theta` compare solo nelle
  etichette dei grafici, mai nelle equazioni.
- **Unita'.** SI internamente, angoli in rad. Gradi e Hz solo in grafici e
  tabelle.
- **Tensori d'inerzia.** Convenzione tensoriale: `I_ij = -int(x_i x_j dm)`
  per `i != j`.

---

## Ipotesi e limiti

### Parametri segnaposto

I seguenti valori **non provengono da misure**. Ogni risultato che ne dipende
e' marcato con un avviso a video, nelle tabelle CSV e nel report.

| parametro | stato |
|---|---|
| `onshape_products_sign` | da verificare con un corpo di prova in Onshape |
| `b_psi`, `b_theta` | da identificare sperimentalmente; attualmente nulli |
| `omega0_deg_s` | serve la velocita' massima reale della montatura |
| parametri di `flex` | dai datasheet dei motori e da FEM o identificazione |

Non sono invece segnaposto, dopo la revisione dei dati CAD del 2026-09-27:

| parametro | stato |
|---|---|
| `az_axis_xy_cm` | esatto per costruzione: l'origine di `W` e' il centro della base dell'albero azimutale |
| `z_O_cm` | misurato nel CAD e **confermato in modo indipendente** (vedi sotto) |
| `x_A0_W`, `elev_axis_W` | il telescopio punta verso `-Y`, quindi l'asse di elevazione e' `X` |

### Verifica indipendente della quota dell'asse di elevazione

Il tubo ottico e' un corpo di rivoluzione attorno al proprio asse, quindi
rispetto al punto `O` il suo tensore d'inerzia dev'essere **principale**: in
particolare `I_xz,O = 0`. Questa condizione determina `z_O` dai soli dati CAD,
senza misurarlo:

```
z_O = z_G_B - I_YZ,G / (m_B * y_G_B)
```

Con i dati correnti la stima vale **54.9125 cm** contro i **54.9 cm** misurati
nel CAD: uno scarto di 0.13 mm. E' una conferma indipendente sia della quota
sia della posa CAD, ed e' verificata a ogni esecuzione
(`inertia.stima_z_O_da_prodotti`, avviso `Z_O_CONFERMATO`).

### Sbilanciamento del telescopio

Il baricentro di B dista `d_x = +7.97 cm` dall'asse di elevazione **lungo la
direzione di puntamento**, e solo `d_z = -0.44 cm` perpendicolarmente. E' la
sorgente dominante di coppia gravitazionale del modello:

- coppia di mantenimento fino a circa **18.8 N m** sul campo operativo
  confermato, `theta` in [-45, 45] deg;
- `k_theta` cambia segno a `theta0 ~= -3.16 deg`: **stabile e oscillatorio**
  per `theta0` da -3.16 a +45 deg (risonanza rigida a bassa frequenza, non
  strutturale), **instabile ad anello aperto** (pendolo invertito) da -45 a
  -3.16 deg, con polo reale fino a `+4.24 rad/s` (0.67 Hz) all'estremo -45 deg.

Non e' un difetto del modello ma una proprieta' del CAD: vale la pena
verificare se corrisponde alla macchina reale o se mancano contrappesi non
modellati.

### Doppio conteggio nei dati CAD

Nelle liste di istanze visibili non compare alcun componente condiviso fra le
due selezioni di Onshape, ma **le liste degli screenshot sono troncate**,
quindi la sovrapposizione non e' esclusa. Il campo `shared_component` di
`config.yaml` permette di sottrarre un componente condiviso da uno dei due
corpi, riportando i tensori allo stesso punto prima della sottrazione.

### Configurazione di riferimento e configurazione attiva

I valori attesi della tabella della sezione 5 della specifica valgono per i
**default della sezione 4**: `x_A0_W = "-Y"`, `elev_axis_W = "X"`,
`theta_cad_deg = 0`, `z_O_cm = null`. I test di regressione girano su quella
configurazione, cosi' da verificare il **codice** indipendentemente dalle
scelte fatte in `config.yaml`; `run_all.py` usa invece la configurazione
attiva e riporta entrambe nel report.

L'unica differenza fra le due e' `z_O_cm`: la configurazione di riferimento
usa `null` (cioe' `d_z = 0` esatto, telescopio bilanciato in verticale per
definizione), quella attiva usa la quota misurata, 54.9 cm, da cui
`d_z = -0.44 cm`.

Se `config.yaml` si discosta dai default, il report lo segnala nella sezione
"Avvisi e incoerenze rilevate". Vale la pena leggerla prima di citare i numeri.

### Controlli automatici di coerenza

A ogni esecuzione il codice verifica, e segnala se falliscono:

- che l'asse di puntamento sia quello a **inerzia minima**, come dev'essere
  per un tubo ottico (avviso `POSA_CAD_SOSPETTA`);
- che `z_O_cm` coincida con la quota stimata da `I_xz,O = 0`
  (`Z_O_CONFERMATO` / `Z_O_INCOERENTE`);
- che `J_psi(theta_cad)` coincida con `I_ZZ` dell'assieme A+B rispetto
  all'asse di azimut, calcolato direttamente dai tensori CAD;
- simmetria, definita positivita' e disuguaglianza triangolare di tutti i
  tensori, sia nel baricentro sia in `O`.
