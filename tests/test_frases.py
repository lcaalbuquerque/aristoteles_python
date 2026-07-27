"""O acumulador de frases e o coracao da baixa latencia -- vale testar bem."""

import doctest

from aristoteles import frases
from aristoteles.frases import AcumuladorDeFrases


def test_doctests_do_modulo():
    """Os exemplos do docstring sao documentacao viva -- dois estavam errados."""
    resultado = doctest.testmod(frases, verbose=False)
    assert resultado.failed == 0, f"{resultado.failed} de {resultado.attempted} doctests falharam"


def test_frase_completa_sai_na_hora():
    a = AcumuladorDeFrases()
    assert a.alimentar("Bom dia. ") == ["Bom dia."]


def test_frase_incompleta_fica_no_buffer():
    a = AcumuladorDeFrases()
    assert a.alimentar("Bom ") == []
    assert a.alimentar("dia") == []
    assert a.finalizar() == ["Bom dia"]


def test_streaming_token_a_token():
    a = AcumuladorDeFrases()
    saida = []
    for token in ["Ola", "! ", "Como ", "vai", "? ", "Tudo ", "bem", "."]:
        saida.extend(a.alimentar(token))
    saida.extend(a.finalizar())
    assert saida == ["Ola!", "Como vai?", "Tudo bem."]


def test_nao_quebra_em_decimal():
    a = AcumuladorDeFrases()
    assert a.alimentar("O valor e 3.14 reais. ") == ["O valor e 3.14 reais."]


def test_nao_quebra_em_abreviacao():
    a = AcumuladorDeFrases()
    assert a.alimentar("O Sr. Silva chegou. ") == ["O Sr. Silva chegou."]


def test_nao_quebra_em_inicial():
    a = AcumuladorDeFrases()
    assert a.alimentar("Falei com J. Silva ontem. ") == ["Falei com J. Silva ontem."]


def test_reticencias_e_pontuacao_multipla():
    a = AcumuladorDeFrases()
    assert a.alimentar("Serio?! Nao acredito... ") == ["Serio?!", "Nao acredito..."]


def test_corta_na_virgula_se_muito_longo():
    a = AcumuladorDeFrases(max_chars=40)
    texto = "primeira parte bem longa da frase, segunda parte que continua sem fim"
    frases = a.alimentar(texto)
    assert len(frases) == 1
    assert frases[0].endswith(",")


def test_finalizar_esvazia_o_buffer():
    a = AcumuladorDeFrases()
    a.alimentar("resto sem ponto")
    assert a.finalizar() == ["resto sem ponto"]
    assert a.finalizar() == []
