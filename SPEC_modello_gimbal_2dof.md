# Specifica per Claude Code — Modello dinamico rigido a 2 GdL di un telescopio alt-azimutale

## 0. Contesto e obiettivo

Sto scrivendo una tesi magistrale sulla dinamica di un telescopio con montatura alt-azimutale: modi di vibrare, frequenze di risonanza e antirisonanza. La montatura ha due assi a presa diretta, azimut/yaw (ψ) ed elevazione/pitch (θ), azionati da motori frameless con statore e rotore montati separatamente.

Questo è il primo modello di una catena di complessità crescente: due corpi rigidi, due gradi di libertà. Ti chiedo un codice Python che:

1. simuli la dinamica non lineare e quella del modello linearizzato;
2. calcoli poli, zeri, frequenze naturali (risonanze) e antirisonanze;
3. fornisca le 4 funzioni di trasferimento coppia → angolo, $G_{ij}(s)$, in forma simbolica e numerica.

Il codice verrà citato in tesi. Deve quindi essere leggibile, rigoroso e riproducibile, e ogni risultato deve essere tracciabile alle equazioni di questa specifica. Commenti e docstring vanno in italiano, con nomi di variabili coerenti con la notazione qui sotto.

**Nota di realismo, da rispettare nei risultati e nel report.** Un modello a corpi rigidi non contiene modi strutturali. Le uniche "risonanze" che produrrà sono:

- i poli nulli del moto rigido;
- una dinamica a bassa frequenza dovuta a gravità (sbilanciamento), accoppiamento giroscopico e rigidezza centrifuga, tipicamente sotto 1 Hz.

Il report deve dirlo esplicitamente e non deve mai presentare queste frequenze come modi strutturali. L'architettura deve però essere pronta per l'estensione flessibile della sezione 10.

## 1. Modo di lavorare

- Leggi tutta la specifica prima di scrivere codice, poi proponimi un piano breve.
- La sezione 4 elenca parametri segnaposto. Prima di implementare, chiedimi di confermarli. Se non rispondo, usa i default indicati, ma ogni output che ne dipende deve riportare un avviso visibile.
- Le convenzioni della sezione 2 sono vincolanti e non vanno modificate.
- Se trovi incoerenze nei dati, segnalale invece di correggerle in silenzio.
- Non inventare dati (smorzamenti, rigidezze, inerzie dei rotori). Se servono e mancano, usa segnaposto dichiarati come tali.
- Implementa un modulo alla volta ed esegui i test della sezione 9 man mano.

## 2. Convenzioni (vincolanti)

**Terne**

- **W**: terna dell'assieme Onshape, inerziale, con $Z_W$ verticale verso l'alto e **origine nel centro della base dell'albero azimutale**. Tutti i dati CAD sono espressi in W. Di conseguenza l'asse di azimut passa per l'origine di W per costruzione, e le quote $z$ si leggono come altezze sopra la base dell'albero.
- **N**: terna inerziale del modello, con origine in O (intersezione degli assi di azimut ed elevazione) e $Z = Z_W$. N e W hanno **gli stessi assi**: N si ottiene da W con una pura traslazione verticale di $z_O$ (l'asse di azimut è verticale e passa per l'origine di entrambe). Non serve quindi nessuna matrice $\mathbf R_{WN}$: il codice lavora con le componenti in W trasportate nel punto O (sezione 5), che coincidono con le componenti in N proprio perché le terne sono parallele.
- **A**: forcella, ruotata di ψ attorno a Z.
- **B**: telescopio. $x_B$ è l'asse di puntamento, $y_B$ l'asse di elevazione e $z_B = x_B \times y_B$.

$$\mathbf R_{NB} = \mathbf R_z(\psi)\,\mathbf R_y(\theta), \qquad \mathbf R_y(\theta) = \begin{bmatrix}\cos\theta & 0 & \sin\theta\\ 0&1&0\\ -\sin\theta & 0 & \cos\theta\end{bmatrix}$$

**Angoli e coppie**

- θ è positivo secondo la regola della mano destra attorno a $y_B$, quindi **θ > 0 abbassa il puntamento**. θ = 0 corrisponde al puntamento orizzontale, θ = −90° allo zenit (il range θ ∈ [−90°, 0°], orizzonte-zenit, resta il dominio naturale della cinematica). **Le configurazioni operative della montatura reale sono però θ ∈ [−45°, 45°]** (confermato dall'utente il 2026-09-27): il pitch spazia da 45° sopra l'orizzonte a 45° sotto, non fino allo zenit.
- L'elevazione fisica è ε = −θ. Si usa solo nelle etichette dei grafici, mai nelle equazioni.
- ψ è positivo antiorario visto dall'alto (mano destra attorno a +Z).
- Le coppie τ_ψ e τ_θ sono positive nello stesso verso delle rispettive coordinate. τ_θ agisce tra statore (sulla forcella) e rotore (sul telescopio). Il lavoro virtuale è $\delta W = \tau_\psi\,\delta\psi + \tau_\theta\,\delta\theta$.
- Nella posa del modello CAD valgono ψ = 0 e θ = θ_CAD.

**Unità e tensori**

- Internamente si usa il SI, con angoli in rad. Grafici e tabelle usano gradi e Hz.
- Tensore d'inerzia in convenzione tensoriale: $I_{ij} = -\int x_i x_j\,dm$ per i ≠ j.

## 3. Dati CAD (Onshape, strumento *Mass properties*)

Nessun mate connector è stato selezionato come riferimento. Quindi le inerzie sono calcolate rispetto al baricentro delle parti selezionate, con assi paralleli a W (comportamento documentato da Onshape). Unità dei dati grezzi: cm, kg, kg·cm².

**Terna di riferimento del CAD (rilevazione del 2026-09-27).** L'origine di W coincide con il **centro della base dell'albero azimutale**. Nella posa del CAD il telescopio punta in direzione e verso **−Y**.

### Corpo A: forcella e albero di azimut (ruota solo in ψ)

- $m_A = 22.535$ kg
- $\mathbf r_{G_A}|_W = (-0.240,\ 0.001,\ 41.944)$ cm
- $\mathbf I_{A,G}|_W$ in kg·cm², come visualizzato da Onshape:

```
[[  6602.470,    -0.172,     6.604],
 [    -0.172, 15363.593,     0.031],
 [     6.604,     0.031, 10431.336]]
```

- Istanze selezionate (lista troncata nello screenshot): B-10 Albero di trasmissione <1>, B-04 Sede anello interno <1>, Part 1 <2>, INNENRING <1>, …

### Corpo B: telescopio e parti che ruotano in elevazione

- $m_B = 24.057$ kg
- $\mathbf r_{G_B}|_W = (1.369,\ -7.965,\ 54.460)$ cm
- $\mathbf I_{B,G}|_W$ in kg·cm², come visualizzato da Onshape:

```
[[  8932.534,  -263.414,   -10.194],
 [  -263.414,  5624.207,    86.696],
 [   -10.194,    86.696, 11222.053]]
```

- Istanze selezionate (lista troncata): Telescopio LAMERA <1>, T-02 Anello di registrazione del telescopio <1>, I04 - Albero condotto <1>, PH Countersunk flat head screw M3x0.5 x 12 <2>, …

### Avvisi sui dati

- **Possibile doppio conteggio.** Nelle liste di istanze visibili non compare alcun componente condiviso fra le due selezioni, ma **le liste sono troncate**, quindi la sovrapposizione non è esclusa. Prevedi in `config.yaml` un campo opzionale con massa, baricentro e tensore di un componente condiviso, e una funzione che lo sottragga da uno dei due corpi (tensori riportati allo stesso punto prima della sottrazione). Finché il campo è vuoto, stampa un avviso.
- **Segno dei prodotti d'inerzia.** La convenzione di Onshape va verificata: vedi `onshape_products_sign` nella sezione 4.
- **Sbilanciamento longitudinale del telescopio.** Il baricentro di B ha $y_{G_B} = -7.965$ cm: con il puntamento lungo −Y questo diventa $d_x = +7.965$ cm, cioè un offset di quasi 8 cm **lungo l'asse di puntamento**. È la sorgente dominante di coppia gravitazionale del modello: coppia di mantenimento fino a ≈ 18.8 N·m e asse di elevazione instabile ad anello aperto verso lo zenit. Va verificato che sia reale e non un residuo di contrappesi non modellati.

## 4. Parametri da confermare (`config.yaml`)

| Parametro | Default | Significato | Origine |
|---|---|---|---|
| `az_axis_xy_cm` | `[0.0, 0.0]` | Posizione dell'asse di azimut nel piano XY di W | **Esatta per costruzione**: l'origine di W è il centro della base dell'albero azimutale |
| `z_O_cm` | `54.9` | Quota dell'asse di elevazione, cioè del punto O | **Misurata nel CAD.** Confermata in modo indipendente: è la quota che rende principale il tensore di B rispetto a O (vedi sezione 5). Se `null`, usa $z_{G_B}$ (cioè $d_z = 0$) con avviso |
| `elev_axis_W` | `"X"` | Direzione dell'asse di elevazione in W | Conseguenza di $x_{A0} = -Y$: $y_{A0} = Z_W \times x_{A0} = +X$. Coerente con il momento minimo di A, che è $I_{xx}$: i bracci della forcella sono distanziati lungo X |
| `x_A0_W` | `"-Y"` | Direzione orizzontale di $x_A$ nella posa CAD, cioè la proiezione orizzontale del puntamento | **Confermata nel CAD**: il telescopio punta in direzione e verso −Y. Coerente con il momento minimo di B, che è $I_{yy}|_W$: l'asse di rotolamento del tubo è Y |
| `theta_cad_deg` | `0.0` | Elevazione del telescopio nella posa CAD | Dedotta dai prodotti d'inerzia di B rispetto a O, che con $z_O = 54.9$ cm sono tutti sotto $5\cdot 10^{-4}$ kg·m² contro momenti di $0.57$–$1.28$ kg·m² |
| `onshape_products_sign` | `"tensor"` | `"tensor"`: i fuori diagonale sono già $-\int x_i x_j\,dm$. `"positive"`: sono $+\int x_i x_j\,dm$ e vanno negati | Da verificare con un corpo di prova in Onshape |
| `b_psi`, `b_theta` | `0.0` | Smorzamenti viscosi in N·m·s/rad | Segnaposto, da identificare sperimentalmente; prevedi una lista di valori per la sensibilità |
| `g` | `9.80665` | Accelerazione di gravità in m/s² | |
| `omega0_deg_s` | `[0, 1, 5, 10]` | Velocità di inseguimento in azimut | Segnaposto: serve la velocità massima reale |
| `theta0_grid_deg` | da −45 a 45, passo 1 | Griglia dei punti di lavoro | **Confermato dall'utente (2026-09-27)**: range di pitch reale della montatura |
| `dz_sensitivity_cm` | `[-2, -1, -0.5, 0, 0.5, 1, 2]` | Sbilanciamenti perpendicolari al puntamento per l'analisi di sensibilità | |
| `flex_model` | `false` | Attiva l'estensione della sezione 10 | |

Costruzione della terna nella posa CAD:

- $y_{A0} = Z_W \times x_{A0}$; verifica con un `assert` che $y_{A0}$ sia parallelo a `elev_axis_W`;
- $\mathbf R_{WA0} = [x_{A0},\ y_{A0},\ Z_W]$, con i vettori come colonne;
- $\mathbf R_{WB0} = \mathbf R_{WA0}\,\mathbf R_y(\theta_{CAD})$.

## 5. Preprocessing inerziale (`inertia.py`)

1. **Conversione in SI.** Da cm a m, da kg·cm² a kg·m² (fattore $10^{-4}$).
2. **Convenzione dei prodotti.** Se `onshape_products_sign == "positive"`, nega i termini fuori diagonale.
3. **Controlli fisici nel baricentro, per A e B.** Il tensore deve essere simmetrico e definito positivo, e i momenti principali devono rispettare $I_1 + I_2 \ge I_3$. Se un controllo fallisce, solleva un errore.
4. **Punto O.** $\mathbf r_O|_W = (x_{az},\ y_{az},\ z_O)$.
5. **Trasporto di B in O (Huygens-Steiner tensoriale).**
   $$\mathbf I_{B,O}|_W = \mathbf I_{B,G}|_W + m_B\left(\|\mathbf r\|^2\,\mathbb 1 - \mathbf r\,\mathbf r^{\mathsf T}\right), \qquad \mathbf r = \mathbf r_{G_B} - \mathbf r_O$$
   Rotazione in terna B: $\mathbf I_{B,O}|_B = \mathbf R_{WB0}^{\mathsf T}\,\mathbf I_{B,O}|_W\,\mathbf R_{WB0}$. Da qui ricavi $I_{xx}, I_{yy}, I_{zz}, I_{xy}, I_{xz}, I_{yz}$ in convenzione tensoriale.
6. **Offset del baricentro in B.** $\mathbf d = (d_x, d_y, d_z) = \mathbf R_{WB0}^{\mathsf T}\,\mathbf r$.
7. **Inerzia di A attorno all'asse di azimut.**
   $$J_A = I_{A,G,ZZ} + m_A\left[(x_{G_A} - x_{az})^2 + (y_{G_A} - y_{az})^2\right]$$
   A ruota solo attorno a Z, quindi gli altri elementi del suo tensore generano solo carichi sui cuscinetti e non entrano nella dinamica.
8. **Riepilogo.** Stampa tutto e salvalo in `results/tables/inertia_summary.csv`.

9. **Stima indipendente di $z_O$.** La quota dell'asse di elevazione si può ricavare dai soli dati CAD, senza misurarla: il tubo è un corpo di rivoluzione attorno al proprio asse, quindi rispetto al punto O il suo tensore deve essere principale, cioè $I_{xz,O} = 0$. Imponendolo:

   $$I_{YZ,G} - m_B\,y_{G_B}\,(z_{G_B} - z_O) = 0 \quad\Longrightarrow\quad z_O = z_{G_B} - \frac{I_{YZ,G}}{m_B\,y_{G_B}}$$

   Confronta il risultato con `z_O_cm` e segnala uno scarto superiore a 1 cm. Con i dati correnti la stima vale **54.9125 cm**, contro i 54.9 cm misurati: uno scarto di 0.13 mm, che conferma sia la quota sia la posa CAD.

**Valori di riferimento con i default**, da usare nei test:

| Grandezza | Valore atteso |
|---|---|
| $J_A$ | 1.043263 kg·m² |
| $I_{B,G,ZZ} + m_B \cdot (\text{distanza orizzontale dall'asse di azimut})^2$ | 1.279335 kg·m² |
| $J_\psi(\theta_{CAD} = 0)$ | 2.322598 kg·m² (indipendente da $z_O$ e dalla convenzione dei prodotti) |
| $I_{xx,O}$ (puntamento), con $d_z = 0$ | 0.566929 kg·m² |
| $I_{yy,O}$ (elevazione), con $d_z = 0$ | 1.045874 kg·m² |
| $I_{zz,O}$, con $d_z = 0$ | 1.279335 kg·m² |
| Autovalori di $\mathbf I_{A,G}$ come visualizzato | ≈ 6602.46, 10431.35, 15363.59 kg·cm² |
| Autovalori di $\mathbf I_{B,G}$ come visualizzato | ≈ 5602.06, 8953.25, 11223.48 kg·cm² |
| $z_O$ stimato da $I_{xz,O} = 0$ | 54.9125 cm |

Nota: a differenza della revisione precedente dei dati CAD, ora $I_{xx,O} < I_{yy,O} < I_{zz,O}$, cioè l'asse di puntamento è quello a inerzia minima, come dev'essere per un tubo ottico. Di conseguenza $J_\psi$ **decresce** dall'orizzonte (2.3226 kg·m²) allo zenit (1.6102 kg·m²): valori della funzione sul suo intero dominio, utili come riferimento, anche se il campo operativo confermato ($\theta \in [-45°, 45°]$, sezione 4) non raggiunge lo zenit.

## 6. Modello non lineare (`kinematics.py`, `lagrange.py`)

Coordinate generalizzate: $\mathbf q = [\psi,\ \theta]^{\mathsf T}$.

### Cinematica

Ricava con SymPy la velocità angolare di B dalla formula di Poisson, $[\boldsymbol\omega_B|_B]_\times = \mathbf R_{NB}^{\mathsf T}\,\dot{\mathbf R}_{NB}$, e verifica che coincida con:

$$\boldsymbol\omega_B\big|_B = \mathbf R_y(\theta)^{\mathsf T}\begin{bmatrix}0\\0\\\dot\psi\end{bmatrix} + \begin{bmatrix}0\\\dot\theta\\0\end{bmatrix} = \begin{bmatrix}-\dot\psi\sin\theta\\ \dot\theta\\ \dot\psi\cos\theta\end{bmatrix}$$

### Energie, dissipazione e forze generalizzate

$$T = \tfrac12 J_A\dot\psi^2 + \tfrac12\,\boldsymbol\omega_B|_B^{\mathsf T}\;\mathbf I_{B,O}|_B\;\boldsymbol\omega_B|_B$$

Il tensore di B è completo, con tutti e sei gli elementi.

$$V = m_B\,g\,(\mathbf r_{OG}\cdot\mathbf Z) = m_B\,g\,(-d_x\sin\theta + d_z\cos\theta)$$

$d_y$ non contribuisce a $V$, perché $y_B$ è sempre orizzontale. A non ha variazioni di energia potenziale. $d_y$ non è però nullo in generale: entra in $I_{xy}$ e $I_{yz}$ (punto 5 della sezione 5) e quindi in $M_{12}(\theta)$ qui sotto — è un accoppiamento inerziale fra azimut ed elevazione, non gravitazionale.

$$\mathcal D = \tfrac12 b_\psi\dot\psi^2 + \tfrac12 b_\theta\dot\theta^2, \qquad \mathbf Q = [\tau_\psi,\ \tau_\theta]^{\mathsf T}$$

### Equazioni di Lagrange

$$\frac{d}{dt}\frac{\partial\mathcal L}{\partial\dot{\mathbf q}} - \frac{\partial\mathcal L}{\partial\mathbf q} + \frac{\partial\mathcal D}{\partial\dot{\mathbf q}} = \mathbf Q, \qquad \mathcal L = T - V$$

Derivale simbolicamente con SymPy. Non codificarle a mano.

**Struttura attesa (target di regressione).**

$$\mathbf M(\theta) = \begin{bmatrix}J_\psi(\theta) & M_{12}(\theta)\\ M_{12}(\theta) & I_{yy}\end{bmatrix}$$

$$J_\psi(\theta) = J_A + I_{xx}\sin^2\theta + I_{zz}\cos^2\theta - I_{xz}\sin 2\theta, \qquad M_{12}(\theta) = -I_{xy}\sin\theta + I_{yz}\cos\theta$$

Con $I_{xy} = I_{xz} = I_{yz} = 0$, le equazioni devono ridursi esattamente a:

$$J_\psi(\theta)\,\ddot\psi + J_\psi'(\theta)\,\dot\theta\,\dot\psi + b_\psi\dot\psi = \tau_\psi$$

$$I_{yy}\,\ddot\theta - \tfrac12 J_\psi'(\theta)\,\dot\psi^2 + \frac{\partial V}{\partial\theta} + b_\theta\dot\theta = \tau_\theta$$

$$J_\psi'(\theta) = (I_{xx} - I_{zz})\sin 2\theta, \qquad \frac{\partial V}{\partial\theta} = -m_B\,g\,(d_x\cos\theta + d_z\sin\theta)$$

### Forma matriciale

$$\mathbf M(\mathbf q)\ddot{\mathbf q} + \mathbf C(\mathbf q,\dot{\mathbf q})\dot{\mathbf q} + \mathbf B\dot{\mathbf q} + \frac{\partial V}{\partial\mathbf q} = \boldsymbol\tau$$

Costruisci $\mathbf C$ con i simboli di Christoffel e verifica che $\dot{\mathbf M} - 2\mathbf C$ sia antisimmetrica.

Output del modulo:

- funzioni numeriche ottenute con `lambdify` per $\mathbf M(\theta)$, $\mathbf C\dot{\mathbf q}$ e $\partial V/\partial\mathbf q$;
- esportazione LaTeX delle equazioni di moto in `results/latex/eom.tex`.

## 7. Linearizzazione e funzioni di trasferimento (`linearize.py`)

### Punto di lavoro

θ = θ₀, $\dot\psi = \Omega_0$, $\dot\theta = 0$, $\ddot{\mathbf q} = 0$. ψ è ciclica, quindi ψ₀ è irrilevante.

Le coppie nominali sono $\boldsymbol\tau_0 = \mathbf C\dot{\mathbf q}_0 + \mathbf B\dot{\mathbf q}_0 + \partial V/\partial\mathbf q|_0$. Riporta sempre $\tau_{\theta 0}$, che è la coppia di mantenimento.

### Matrici linearizzate

Ricavale come jacobiani simbolici del residuo $\mathbf r = \mathbf M\ddot{\mathbf q} + \mathbf C\dot{\mathbf q} + \mathbf B\dot{\mathbf q} + \partial V/\partial\mathbf q$, valutati nel punto di lavoro:

$$\mathbf M_0 = \frac{\partial\mathbf r}{\partial\ddot{\mathbf q}}\bigg|_0, \qquad \mathbf C_0 = \frac{\partial\mathbf r}{\partial\dot{\mathbf q}}\bigg|_0, \qquad \mathbf K_0 = \frac{\partial\mathbf r}{\partial\mathbf q}\bigg|_0$$

$$\mathbf M_0\,\delta\ddot{\mathbf q} + \mathbf C_0\,\delta\dot{\mathbf q} + \mathbf K_0\,\delta\mathbf q = \delta\boldsymbol\tau$$

La prima colonna di $\mathbf K_0$ è nulla, perché ψ è ciclica.

**Target di regressione con prodotti nulli.** Qui il coefficiente giroscopico si chiama γ, per non confonderlo con l'accelerazione di gravità g.

$$\mathbf C_0 = \begin{bmatrix}b_\psi & \gamma\\ -\gamma & b_\theta\end{bmatrix}, \qquad \mathbf K_0 = \begin{bmatrix}0 & 0\\ 0 & k_\theta\end{bmatrix}$$

$$\gamma = (I_{xx} - I_{zz})\,\Omega_0\sin 2\theta_0$$

$$k_\theta = \underbrace{-(I_{xx} - I_{zz})\,\Omega_0^2\cos 2\theta_0}_{k_c\ \text{(centrifuga)}} + \underbrace{m_B\,g\,(d_x\sin\theta_0 - d_z\cos\theta_0)}_{k_g = \partial^2 V/\partial\theta^2}$$

**Correzione (2026-09-28), verificata contro l'uscita simbolica di `linearize.py`.** La struttura $\mathbf C_0 = \begin{bmatrix}b_\psi & \gamma\\ -\gamma & b_\theta\end{bmatrix}$ e $\mathbf K_0 = \begin{bmatrix}0 & 0\\ 0 & k_\theta\end{bmatrix}$ **non è limitata al caso a prodotti nulli**: vale in generale, con $I_{xy}$ e $I_{yz}$ assenti da $\mathbf C_0$ e $\mathbf K_0$ in ogni caso (compaiono solo in $\mathbf M_0$, tramite $M_{12}$). Con prodotti non nulli basta sostituire

$$\gamma = \Omega_0\left[(I_{xx} - I_{zz})\sin 2\theta_0 - 2 I_{xz}\cos 2\theta_0\right], \qquad k_\theta = \Omega_0^2\left[-(I_{xx}-I_{zz})\cos2\theta_0 - 2I_{xz}\sin2\theta_0\right] + m_B g(d_x\sin\theta_0 - d_z\cos\theta_0)$$

La spiegazione è diretta: $\partial(\mathbf M\ddot{\mathbf q})/\partial\theta$ si annulla nel punto di lavoro perché è valutato a $\ddot{\mathbf q}=0$, e sia $C_{11}$ che $\partial V/\partial\psi$ sono identicamente nulli o proporzionali a $\dot\theta$ (nullo nel punto di lavoro) — non serve quindi ipotizzare prodotti d'inerzia nulli perché la prima riga/colonna di $\mathbf K_0$ e la diagonale di $\mathbf C_0$ si annullino.

### Spazio degli stati

$$\mathbf x = [\delta\psi,\ \delta\theta,\ \delta\dot\psi,\ \delta\dot\theta]^{\mathsf T}$$

$$\mathbf A = \begin{bmatrix}\mathbf 0 & \mathbb 1\\ -\mathbf M_0^{-1}\mathbf K_0 & -\mathbf M_0^{-1}\mathbf C_0\end{bmatrix}, \quad \mathbf B_u = \begin{bmatrix}\mathbf 0\\ \mathbf M_0^{-1}\end{bmatrix}, \quad \mathbf C_y = [\mathbb 1\ \ \mathbf 0], \quad \mathbf D = \mathbf 0$$

### Matrice di trasferimento

$$\mathbf P(s) = \mathbf M_0 s^2 + \mathbf C_0 s + \mathbf K_0, \qquad \mathbf G(s) = \mathbf P(s)^{-1} = \frac{\operatorname{adj}\mathbf P(s)}{\Delta(s)}, \qquad \Delta = \det\mathbf P$$

| FdT | Significato | Espressione |
|---|---|---|
| $G_{11} = \Psi/T_\psi$ | collocata, azimut | $P_{22}/\Delta$ |
| $G_{12} = \Psi/T_\theta$ | incrociata | $-P_{12}/\Delta$ |
| $G_{21} = \Theta/T_\psi$ | incrociata | $-P_{21}/\Delta$ |
| $G_{22} = \Theta/T_\theta$ | collocata, elevazione | $P_{11}/\Delta$ |

Consegne di questo modulo:

1. **FdT simboliche.** $G_{ij}(s;\theta_0,\Omega_0,\text{parametri})$ con SymPy, semplificate ed esportate in LaTeX (`results/latex/tf.tex`).
2. **FdT numeriche.** Coefficienti di numeratore e denominatore per ogni punto di lavoro di interesse, con cancellazione delle radici comuni entro una tolleranza. In particolare $\Delta$ ha un fattore s dovuto a ψ ciclica. Documenta ogni cancellazione nel report.
3. **FRF.** Valuta $\mathbf G(j\omega) = \mathbf P(j\omega)^{-1}$ direttamente, non tramite i polinomi, per avere una migliore accuratezza numerica.

**Target di regressione** con $\Omega_0 = 0$, prodotti nulli e $\mathbf d = 0$:

$$G_{11} = \frac{1}{s\,(J_\psi(\theta_0)\,s + b_\psi)}, \qquad G_{22} = \frac{1}{s\,(I_{yy}\,s + b_\theta)}, \qquad G_{12} = G_{21} = 0$$

Con $\mathbf d \neq 0$ e $\Omega_0 = 0$ deve valere $G_{22} = 1/(I_{yy}s^2 + b_\theta s + k_g)$.

## 8. Analisi e simulazioni (`analysis.py`, `simulate.py`)

### 8.1 Poli, zeri e frequenze

Calcola gli autovalori di $\mathbf A$ e classificali:

- **nulli**: moto rigido;
- **reali**: se positivi, instabilità (ad esempio un pendolo inverso per sbilanciamento);
- **complessi coniugati**: frequenza $f_n = |\lambda|/2\pi$ e smorzamento $\zeta = -\mathrm{Re}\,\lambda/|\lambda|$.

Gli zeri sono le radici di $P_{22}$ (antirisonanze di $G_{11}$), di $P_{11}$ (antirisonanze di $G_{22}$), di $P_{12}$ e di $P_{21}$. Salva una tabella CSV per ogni combinazione di (θ₀, Ω₀, $d_z$).

### 8.2 Analisi parametriche

- Frequenze naturali e zeri in funzione di θ₀, una curva per ogni Ω₀.
- Stesse grandezze in funzione di $d_z$.
- Grafico di $J_\psi(\theta)$ sul campo operativo confermato, θ ∈ [−45°, 45°].

### 8.3 Diagrammi di Bode

Griglia 2×2 con modulo (dB rispetto a 1 rad/(N·m)) e fase (gradi, con unwrap) delle quattro FdT. Punti di lavoro: θ₀ ∈ {−45°, −15°, 15°, 45°} (estremi e punti intermedi del campo operativo confermato, scelti per mostrare sia il regime instabile sia quello oscillatorio, separati da $k_\theta = 0$ a circa $\theta_0 = -3.16°$) e Ω₀ ∈ {0, massimo}. Frequenze: 2000 punti logaritmici tra $10^{-3}$ e $10^2$ Hz. Segna poli e zeri sui grafici.

### 8.4 Simulazioni non lineari

Usa `scipy.integrate.solve_ivp` con metodo `DOP853` (oppure `Radau` se il problema risulta rigido), `rtol=1e-10`, `atol=1e-12` e uscita densa.

- **S1.** θ₀ = −45°, Ω₀ = 0, doppietto di coppia su τ_ψ con ampiezza tale che |δψ| < 1°. Confronto con la risposta del modello lineare (`scipy.signal.lsim`): errore relativo sotto l'1%.
- **S2.** Come S1, ma su τ_θ. Se $d_z > 0$ l'asse è instabile ad anello aperto: usa un orizzonte breve e segnalalo nel report.
- **S3.** Inseguimento a Ω₀ massima, con coppie nominali $\boldsymbol\tau_0$ in feedforward più un piccolo doppietto su τ_ψ. Mostra l'accoppiamento su θ e confrontalo con il modello lineare nel punto (θ₀, Ω₀).
- **S4.** Verifica conservativa: $b = 0$, $\tau = 0$, condizioni iniziali ampie. L'energia $E = T + V$ e il momento $p_\psi = \partial\mathcal L/\partial\dot\psi$ devono conservarsi, con deriva relativa sotto $10^{-8}$.
- **S5 (facoltativa).** Slew di 90° in azimut con coppia bang-bang, per mostrare dove la predizione lineare smette di valere.

Per ogni scenario produci le storie temporali, con la sovrapposizione lineare/non lineare dove prevista.

## 9. Test (`pytest`)

1. Conversioni di unità e simmetria dei tensori.
2. Teorema di Steiner su un corpo noto: massa puntiforme e parallelepipedo omogeneo.
3. Controlli fisici (definita positività e disuguaglianza triangolare) su A e B, sia nel baricentro sia in O.
4. Valori di riferimento della sezione 5.
5. $J_\psi(\theta_{CAD})$ calcolato con il modello coincide con $I_{ZZ}$ dell'assieme A+B rispetto all'asse di azimut, calcolato direttamente dai dati CAD sommando i tensori.
5b. La stima di $z_O$ dalla condizione $I_{xz,O} = 0$ coincide con `z_O_cm` entro 1 cm.
6. La velocità angolare ricavata con Poisson coincide con quella della sezione 6.
7. Con prodotti nulli le equazioni simboliche coincidono con quelle in forma chiusa della sezione 6.
8. $\dot{\mathbf M} - 2\mathbf C$ è antisimmetrica.
9. Linearizzazione: i target di regressione della sezione 7.
10. Coerenza tra le rappresentazioni: autovalori di $\mathbf A$ = radici di $\Delta(s)$, e FRF dallo spazio degli stati = FRF da $\mathbf P(j\omega)^{-1}$.
11. Scenario S4: conservazione di energia e momento.
12. Scenari S1 e S3: accordo lineare/non lineare entro la tolleranza.

## 10. Estensione flessibile a 4 GdL (`flex4dof.py`, attivata con `flex_model: true`)

Implementa questa estensione come modulo separato, disattivato di default. Senza parametri reali deve rifiutarsi di produrre risultati e stampare un messaggio chiaro. Non presentare mai frequenze ottenute da segnaposto come risultati.

**Coordinate.** $\mathbf q = [\psi_m,\ \psi,\ \theta_m,\ \theta]$: i rotori dei motori ($\psi_m$, $\theta_m$) sono collegati alle rispettive strutture (ψ, θ) da molle torsionali.

$$T = \tfrac12 J_{m\psi}\dot\psi_m^2 + \tfrac12 J_\psi^{s}(\theta)\,\dot\psi^2 + \tfrac12 I_{yy}^{s}\,\dot\theta^2 + \tfrac12 J_{m\theta}\,\dot\theta_m^2$$

$$V = \tfrac12 k_\psi(\psi_m - \psi)^2 + \tfrac12 k_\theta(\theta_m - \theta)^2 + V_g(\theta)$$

$$\mathcal D = \tfrac12 c_\psi(\dot\psi_m - \dot\psi)^2 + \tfrac12 c_\theta(\dot\theta_m - \dot\theta)^2 + \tfrac12 b_\psi\dot\psi_m^2 + \tfrac12 b_\theta\dot\theta_m^2$$

Le coppie τ_ψ e τ_θ agiscono rispettivamente su $\psi_m$ e $\theta_m$.

Nel modello strutturale, le inerzie dei rotori vanno tolte da quelle dei corpi rigidi per non contarle due volte:

$$J_\psi^{s}(\theta) = J_\psi(\theta) - J_{m\psi}, \qquad I_{yy}^{s} = I_{yy} - J_{m\theta}$$

Il contributo del rotore di elevazione all'inerzia di azimut resta, in approssimazione, dentro $J_\psi^{s}$. I parametri $J_{m\psi}$, $J_{m\theta}$ (dai datasheet dei motori) e $k$, $c$ (da FEM o da identificazione) sono segnaposto.

**Uscite.**

- Matrice 2×2 collocata coppia → angolo dei rotori ($\psi_m$, $\theta_m$), cioè quello che misura l'encoder.
- Matrice 2×2 coppia → angolo delle strutture (ψ, θ), cioè il lato carico.

**Test.**

- Le FRF collocate non smorzate hanno poli e zeri alternati (interlacing).
- Nel limite $k \to \infty$ si ritrovano le FdT del modello rigido a bassa frequenza.
- Le antirisonanze delle FRF collocate coincidono con le frequenze proprie del sistema a rotore bloccato.

## 11. Struttura del progetto e requisiti tecnici

```
gimbal_2dof/
├── config.yaml
├── requirements.txt        # numpy, scipy, sympy, matplotlib, pyyaml, pytest
├── README.md               # come eseguire, dove sono i risultati, ipotesi
├── run_all.py              # CLI: --theta0, --omega0, --quick
├── src/gimbal/
│   ├── cad_data.py         # dati grezzi della sezione 3, immutabili
│   ├── params.py           # lettura config + conversione SI
│   ├── inertia.py
│   ├── kinematics.py
│   ├── lagrange.py
│   ├── linearize.py
│   ├── analysis.py
│   ├── simulate.py
│   ├── plotting.py
│   ├── report.py
│   └── flex4dof.py
├── tests/
└── results/
    ├── figures/            # PDF vettoriale + PNG a 300 dpi
    ├── tables/             # CSV
    ├── latex/              # equazioni e FdT in LaTeX
    └── report.md
```

**Requisiti tecnici.**

- Python 3.10 o successivo, type hints ovunque. `python-control` è ammesso solo come dipendenza facoltativa.
- Figure con matplotlib: etichette in italiano con unità, stile uniforme, nomi di file descrittivi.
- `report.md` deve contenere:
  - i parametri usati, con i segnaposto evidenziati;
  - il riepilogo inerziale;
  - l'esito dei controlli di riferimento;
  - le tabelle di poli e zeri;
  - l'elenco delle figure;
  - la nota di realismo della sezione 0.
