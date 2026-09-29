"""Test della sezione 9, punti 1-5: preprocessing inerziale."""

from __future__ import annotations

import numpy as np
import pytest

from gimbal import cad_data, inertia as inz, params as pr


# --- 1. conversioni di unita' e simmetria dei tensori -----------------------


def test_conversione_unita() -> None:
    """cm -> m e kg cm^2 -> kg m^2 con i fattori corretti."""
    assert cad_data.CM_TO_M == pytest.approx(1e-2)
    assert cad_data.KG_CM2_TO_KG_M2 == pytest.approx(1e-4)

    corpo = cad_data.CORPO_A
    assert corpo.r_G_W == pytest.approx(corpo.r_G_cm * 1e-2)
    assert corpo.I_G_W_grezzo == pytest.approx(corpo.I_G_kg_cm2 * 1e-4)
    # L'origine di W e' la base dell'albero azimutale: il baricentro della
    # forcella sta a qualche decimetro sopra, non a decine di metri ne' a
    # frazioni di centimetro.
    assert 0.1 < corpo.r_G_W[2] < 1.5
    assert 0.1 < cad_data.CORPO_B.r_G_W[2] < 1.5


def test_dati_cad_immutabili() -> None:
    """I dati grezzi non devono poter essere modificati per sbaglio."""
    with pytest.raises(ValueError):
        cad_data.CORPO_A.I_G_kg_cm2[0, 0] = 0.0


@pytest.mark.parametrize("corpo", [cad_data.CORPO_A, cad_data.CORPO_B])
def test_simmetria_tensori(corpo: cad_data.CorpoCAD) -> None:
    """I tensori CAD sono simmetrici, in entrambe le convenzioni di segno."""
    for convenzione in ("tensor", "positive"):
        I = inz.normalizza_prodotti(corpo.I_G_W_grezzo, convenzione)
        assert I == pytest.approx(I.T)


def test_normalizza_prodotti_nega_solo_fuori_diagonale() -> None:
    """La convenzione 'positive' nega i prodotti e lascia i momenti."""
    I = np.array([[1.0, 2.0, 3.0], [2.0, 4.0, 5.0], [3.0, 5.0, 6.0]])
    negato = inz.normalizza_prodotti(I, "positive")
    assert np.diag(negato) == pytest.approx(np.diag(I))
    assert negato[0, 1] == pytest.approx(-I[0, 1])
    assert negato[0, 2] == pytest.approx(-I[0, 2])
    assert negato[1, 2] == pytest.approx(-I[1, 2])
    assert inz.normalizza_prodotti(I, "tensor") == pytest.approx(I)


def test_convenzione_sconosciuta_solleva() -> None:
    with pytest.raises(ValueError, match="non riconosciuta"):
        inz.normalizza_prodotti(np.eye(3), "qualunque")


# --- 2. teorema di Steiner su corpi noti ------------------------------------


def test_steiner_massa_puntiforme() -> None:
    """Massa puntiforme: I_G = 0, quindi I_P = m(|r|^2 I - r r^T).

    Il momento attorno a un asse passante per P e parallelo a r deve essere
    nullo, e quello attorno a un asse perpendicolare deve valere m|r|^2.
    """
    m = 3.7
    r = np.array([0.3, -0.4, 1.2])
    I_P = inz.trasporta(np.zeros((3, 3)), m, r)

    versore = r / np.linalg.norm(r)
    assert float(versore @ I_P @ versore) == pytest.approx(0.0, abs=1e-12)

    perpendicolare = np.cross(versore, [0.0, 0.0, 1.0])
    perpendicolare /= np.linalg.norm(perpendicolare)
    assert float(perpendicolare @ I_P @ perpendicolare) == pytest.approx(m * float(r @ r))


def test_steiner_parallelepipedo_omogeneo() -> None:
    """Parallelepipedo omogeneo: trasporto dal baricentro a uno spigolo.

    Per un blocco a x b x c, il momento attorno a uno spigolo parallelo a z
    vale ``m(a^2 + b^2)/3``, contro ``m(a^2 + b^2)/12`` nel baricentro.
    """
    m, a, b, c = 2.5, 0.4, 0.6, 0.9
    I_G = m / 12.0 * np.diag([b**2 + c**2, a**2 + c**2, a**2 + b**2])
    r = np.array([a, b, c]) / 2.0  # dallo spigolo al baricentro
    I_spigolo = inz.trasporta(I_G, m, r)

    assert I_spigolo[2, 2] == pytest.approx(m * (a**2 + b**2) / 3.0)
    assert I_spigolo[0, 0] == pytest.approx(m * (b**2 + c**2) / 3.0)
    assert I_spigolo[1, 1] == pytest.approx(m * (a**2 + c**2) / 3.0)
    # In convenzione tensoriale il prodotto vale -m x y.
    assert I_spigolo[0, 1] == pytest.approx(-m * a * b / 4.0)


def test_steiner_andata_e_ritorno() -> None:
    """Trasportare e ritrasportare deve riportare al tensore di partenza."""
    m = 4.2
    r = np.array([0.1, 0.2, -0.3])
    I_G = np.diag([0.5, 0.7, 0.9])
    assert inz.trasporta(inz.trasporta(I_G, m, r), -m, r) == pytest.approx(I_G)


def test_rotazione_conserva_autovalori() -> None:
    """Il cambio di terna non cambia i momenti principali."""
    I = inz.normalizza_prodotti(cad_data.CORPO_B.I_G_W_grezzo, "tensor")
    theta = 0.7
    R = pr.R_y(theta) @ pr.R_z(-0.3)
    assert inz.momenti_principali(inz.ruota(I, R)) == pytest.approx(
        inz.momenti_principali(I)
    )


# --- 3. controlli fisici su A e B, nel baricentro e in O --------------------


@pytest.mark.parametrize("corpo", [cad_data.CORPO_A, cad_data.CORPO_B])
def test_controlli_fisici_nel_baricentro(corpo: cad_data.CorpoCAD) -> None:
    """Simmetria, definita positivita' e disuguaglianza triangolare."""
    I = inz.normalizza_prodotti(corpo.I_G_W_grezzo, "tensor")
    principali = inz.controlli_fisici(I, f"I_{corpo.nome},G")
    assert principali[0] > 0.0
    assert principali[0] + principali[1] >= principali[2] * (1 - 1e-12)


def test_controlli_fisici_in_O(par_riferimento, modello_riferimento) -> None:
    """Gli stessi controlli devono valere sui tensori trasportati in O."""
    inz.controlli_fisici(modello_riferimento.I_B_O_W, "I_B,O|W")
    inz.controlli_fisici(modello_riferimento.I_B_O_B, "I_B,O|B")
    I_A_O = inz.trasporta(
        modello_riferimento.I_A_G_W,
        modello_riferimento.m_A,
        modello_riferimento.r_G_A_W - modello_riferimento.r_O_W,
    )
    inz.controlli_fisici(I_A_O, "I_A,O|W")


def test_controlli_fisici_rifiutano_tensori_non_fisici() -> None:
    """Un tensore non fisico deve far fallire il controllo, non passare."""
    with pytest.raises(inz.TensoreNonFisico, match="non simmetrico"):
        inz.controlli_fisici(np.array([[1.0, 0.5, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]), "prova")
    with pytest.raises(inz.TensoreNonFisico, match="non definito positivo"):
        inz.controlli_fisici(np.diag([1.0, 1.0, -1.0]), "prova")
    # I1 + I2 < I3: impossibile per un corpo reale.
    with pytest.raises(inz.TensoreNonFisico, match="triangolare"):
        inz.controlli_fisici(np.diag([1.0, 1.0, 5.0]), "prova")


# --- 4. valori di riferimento della sezione 5 -------------------------------


def test_valori_riferimento_sezione5(modello_riferimento) -> None:
    """I sei valori attesi della tabella della sezione 5."""
    rif = cad_data.VALORI_RIFERIMENTO
    m = modello_riferimento
    assert m.J_A == pytest.approx(rif["J_A"], abs=5e-6)
    assert m.I_zz == pytest.approx(rif["I_B_G_ZZ_piu_steiner_azimut"], abs=5e-6)
    assert float(m.J_psi(0.0)) == pytest.approx(rif["J_psi_theta_cad_0"], abs=5e-6)
    assert m.I_xx == pytest.approx(rif["I_xx_O_dz0"], abs=5e-6)
    assert m.I_yy == pytest.approx(rif["I_yy_O_dz0"], abs=5e-6)
    assert m.I_zz == pytest.approx(rif["I_zz_O_dz0"], abs=5e-6)
    # Nella configurazione di riferimento il telescopio e' assunto bilanciato.
    assert m.d_z == pytest.approx(0.0, abs=1e-15)


def test_autovalori_riferimento(par_riferimento) -> None:
    """Gli autovalori dei tensori baricentrici, in kg cm^2."""
    principali = inz.tabella_momenti_principali(par_riferimento)
    for corpo, attesi in cad_data.AUTOVALORI_RIFERIMENTO.items():
        assert principali[corpo] == pytest.approx(np.array(attesi), abs=0.01)


def test_J_psi_indipendente_da_z_O_a_theta_nullo() -> None:
    """``J_psi(0)`` non dipende da ``z_O``: il trasporto verticale non pesa su ZZ."""
    valori = [
        float(
            inz.costruisci_modello_inerziale(
                pr.parametri_di_riferimento(z_O_cm=z), verifica=False
            ).J_psi(0.0)
        )
        for z in (None, 0.0, 54.9, 148.524, 300.0)
    ]
    assert valori == pytest.approx([valori[0]] * len(valori), abs=1e-12)


def test_J_psi_indipendente_dalla_convenzione_dei_prodotti() -> None:
    """``J_psi(0)`` non dipende dal segno dei prodotti d'inerzia."""
    a = inz.costruisci_modello_inerziale(
        pr.parametri_di_riferimento(onshape_products_sign="tensor")
    )
    b = inz.costruisci_modello_inerziale(
        pr.parametri_di_riferimento(onshape_products_sign="positive")
    )
    assert float(a.J_psi(0.0)) == pytest.approx(float(b.J_psi(0.0)))


# --- 5. controllo incrociato con i dati CAD ---------------------------------


@pytest.mark.parametrize(
    "override",
    [
        {},
        {"z_O_cm": 54.9},
        {"theta_cad_deg": -30.0},
        {"x_A0_W": "+X", "elev_axis_W": "Y"},
        {"onshape_products_sign": "positive"},
    ],
)
def test_J_psi_theta_cad_coincide_con_I_ZZ_cad(override: dict) -> None:
    """``J_psi(theta_cad)`` = ``I_ZZ`` di A+B sull'asse di azimut, dai dati CAD.

    Deve valere per qualunque posa CAD e qualunque ``z_O``: nella posa del CAD
    il modello e l'assieme sono lo stesso oggetto.
    """
    par = pr.parametri_di_riferimento(**override)
    mod = inz.costruisci_modello_inerziale(par, verifica=False)
    assert float(mod.J_psi(par.theta_cad)) == pytest.approx(
        inz.I_zz_assieme_asse_azimut(par), rel=1e-12
    )


# --- componente condiviso e sbilanciamento imposto --------------------------


def test_sottrazione_componente_condiviso() -> None:
    """Sottrarre un componente e poi riaggiungerlo riporta al corpo iniziale."""
    corpo = inz.CorpoRidotto(
        nome="prova",
        massa=10.0,
        r_G_W=np.array([0.1, 0.2, 1.0]),
        I_G_W=np.diag([1.0, 1.2, 1.4]),
    )
    condiviso = {
        "massa_kg": 1.5,
        "r_G_cm": [5.0, 10.0, 110.0],
        "I_G_kg_cm2": np.diag([100.0, 120.0, 140.0]).tolist(),
    }
    ridotto = inz.sottrai_componente_condiviso(corpo, condiviso, "tensor")
    assert ridotto.massa == pytest.approx(corpo.massa - 1.5)

    # Momento di primo grado e tensore nell'origine restano additivi.
    m_c = 1.5
    r_c = np.array(condiviso["r_G_cm"]) * 1e-2
    assert ridotto.massa * ridotto.r_G_W + m_c * r_c == pytest.approx(
        corpo.massa * corpo.r_G_W
    )
    I_origine = inz.trasporta(ridotto.I_G_W, ridotto.massa, ridotto.r_G_W) + inz.trasporta(
        np.diag([100.0, 120.0, 140.0]) * 1e-4, m_c, r_c
    )
    assert I_origine == pytest.approx(inz.trasporta(corpo.I_G_W, corpo.massa, corpo.r_G_W))


def test_sottrazione_massa_eccessiva_solleva() -> None:
    corpo = inz.CorpoRidotto("x", 1.0, np.zeros(3), np.eye(3))
    with pytest.raises(ValueError, match="massa residua"):
        inz.sottrai_componente_condiviso(
            corpo, {"massa_kg": 2.0, "r_G_cm": [0, 0, 0], "I_G_kg_cm2": np.eye(3).tolist()},
            "tensor",
        )


@pytest.mark.parametrize("dz", [-0.02, -0.005, 0.0, 0.005, 0.02])
def test_modello_con_dz_e_coerente_con_steiner(par_attivo, dz: float) -> None:
    """Imporre ``d_z`` sposta O e aggiorna ``I_B,O`` in modo coerente.

    Il controllo e' che l'inerzia di elevazione soddisfi Steiner rispetto al
    baricentro: ``I_yy(d_z) - I_yy(0) = m_B (d_z^2 - d_z0^2)``.
    """
    mod = inz.modello_con_dz(par_attivo, dz, verifica=False)
    base = inz.modello_con_dz(par_attivo, 0.0, verifica=False)
    assert mod.d_z == pytest.approx(dz, abs=1e-12)
    assert mod.I_yy - base.I_yy == pytest.approx(mod.m_B * dz**2, rel=1e-9, abs=1e-12)


# --- coerenza della posa CAD ------------------------------------------------


def test_posa_cad_coerente_nella_configurazione_di_riferimento(modello_riferimento) -> None:
    """Con i default della specifica l'asse di puntamento e' a inerzia minima."""
    codici = {a.codice for a in modello_riferimento.avvisi}
    assert "POSA_CAD_SOSPETTA" not in codici
    diagonale = np.diag(
        inz.ruota(modello_riferimento.I_B_G_W, modello_riferimento.R_WB0)
    )
    assert np.argmin(diagonale) == 0  # x_B, il puntamento


def test_posa_cad_sospetta_viene_segnalata() -> None:
    """Una posa che non mette il puntamento a inerzia minima produce un avviso.

    Per un tubo ottico l'asse di puntamento e' l'asse di rotolamento: se il
    momento minimo cade altrove, ``x_A0_W`` e' quasi certamente orientato male.
    """
    par = pr.parametri_di_riferimento(x_A0_W="+X", elev_axis_W="Y")
    mod = inz.costruisci_modello_inerziale(par)
    sospetti = [a for a in mod.avvisi if a.codice == "POSA_CAD_SOSPETTA"]
    assert len(sospetti) == 1
    assert sospetti[0].severita == "incoerenza"
    # Con quella posa I_xx e I_yy risultano scambiati rispetto alla sezione 5.
    rif = cad_data.VALORI_RIFERIMENTO
    assert mod.I_xx == pytest.approx(rif["I_yy_O_dz0"], abs=5e-6)
    assert mod.I_yy == pytest.approx(rif["I_xx_O_dz0"], abs=5e-6)



# --- stima indipendente di z_O ---------------------------------------------


def test_stima_z_O_da_prodotti(par_attivo) -> None:
    """``z_O`` stimato annullando ``I_xz,O`` coincide con quello configurato.

    Il tubo e' un corpo di rivoluzione attorno al proprio asse, quindi rispetto
    al punto O il suo tensore dev'essere principale. La quota che lo rende tale
    e' determinata dai soli dati CAD ed e' una verifica indipendente sia di
    ``z_O_cm`` sia della posa CAD.
    """
    stimato = inz.stima_z_O_da_prodotti(par_attivo)
    assert stimato is not None
    assert stimato * 100 == pytest.approx(cad_data.Z_O_STIMATO_CM, abs=1e-3)
    # Scarto rispetto al valore misurato: sotto il centimetro.
    assert abs(stimato - float(par_attivo.r_O_W[2])) < 0.01


def test_stima_z_O_annulla_davvero_il_prodotto(par_attivo) -> None:
    """Costruendo il modello alla quota stimata, ``I_xz,O`` si annulla."""
    from dataclasses import replace

    stimato = inz.stima_z_O_da_prodotti(par_attivo)
    par = replace(
        par_attivo, r_O_W=np.array([par_attivo.r_O_W[0], par_attivo.r_O_W[1], stimato])
    )
    mod = inz.costruisci_modello_inerziale(par, verifica=False)
    assert mod.I_xz == pytest.approx(0.0, abs=1e-12)


def test_stima_z_O_indeterminata_senza_offset() -> None:
    """Se il baricentro di B e' sull'asse di elevazione la stima non esiste.

    Con ``d_x = 0`` il prodotto ``I_xz,O`` non dipende piu' da ``z_O``, quindi
    la condizione non determina nulla e la funzione deve dirlo restituendo
    ``None`` invece di un numero privo di significato.
    """
    from dataclasses import replace

    par = pr.parametri_di_riferimento()
    # Si sposta l'asse di azimut sul baricentro di B lungo il puntamento:
    # con x_A0 = -Y il puntamento e' Y, quindi si allinea y_az a y_G_B.
    y_G_B = float(cad_data.CORPO_B.r_G_W[1])
    par = replace(par, r_O_W=np.array([0.0, y_G_B, float(par.r_O_W[2])]))
    assert inz.stima_z_O_da_prodotti(par) is None


def test_conferma_z_O_negli_avvisi(modello_attivo) -> None:
    """Il modello registra la conferma indipendente di ``z_O``."""
    codici = {a.codice for a in modello_attivo.avvisi}
    assert "Z_O_CONFERMATO" in codici
    assert "Z_O_INCOERENTE" not in codici


def test_z_O_incoerente_viene_segnalato() -> None:
    """Una quota sbagliata di alcuni centimetri deve produrre un'incoerenza."""
    par = pr.parametri_di_riferimento(z_O_cm=cad_data.Z_O_STIMATO_CM + 5.0)
    mod = inz.costruisci_modello_inerziale(par, verifica=False)
    incoerenti = [a for a in mod.avvisi if a.codice == "Z_O_INCOERENTE"]
    assert len(incoerenti) == 1
    assert "I_xz,O = 0" in incoerenti[0].messaggio


def test_sbilanciamento_longitudinale_segnalato(modello_attivo) -> None:
    """Il modello quantifica lo sbilanciamento lungo il puntamento.

    Con i dati correnti ``d_x`` vale quasi 8 cm: e' la sorgente dominante di
    coppia gravitazionale e rende instabile l'asse di elevazione verso lo
    zenit. L'avviso deve riportare coppia di mantenimento e polo instabile.
    """
    assert modello_attivo.d_x > 0.05
    avvisi = [a for a in modello_attivo.avvisi if a.codice == "SBILANCIAMENTO"]
    assert len(avvisi) == 1
    assert "coppia di mantenimento" in avvisi[0].messaggio.lower()
    assert "instabile ad anello aperto" in avvisi[0].messaggio
