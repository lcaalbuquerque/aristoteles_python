"""A janela de historico e o filtro de tags -- as duas partes puras do cerebro.

Nao instanciamos `Cerebro` aqui: o construtor exige credencial da API.
"""

from aristoteles.cerebro import aparar_historico
from aristoteles.frases import sem_tags


def _conversa(turnos: int) -> list[dict]:
    msgs = []
    for i in range(turnos):
        msgs.append({"role": "user", "content": f"pergunta {i}"})
        msgs.append({"role": "assistant", "content": f"resposta {i}"})
    return msgs


def test_janela_curta_passa_intacta():
    msgs = _conversa(3)
    assert aparar_historico(list(msgs), 20) == msgs


def test_primeira_mensagem_e_sempre_do_usuario():
    """A API devolve 400 se messages[0] for do assistente."""
    for turnos in range(1, 16):
        msgs = _conversa(turnos)
        msgs.append({"role": "user", "content": "pergunta nova"})
        janela = aparar_historico(msgs, 20)
        assert janela[0]["role"] == "user"


def test_descarta_em_pares_e_respeita_o_teto():
    # 10 pares + 1 pergunta = 21 mensagens, teto 20.
    msgs = _conversa(10) + [{"role": "user", "content": "pergunta nova"}]
    janela = aparar_historico(msgs, 20)
    # Corta a pergunta 0, depois a resposta 0 orfa: sobram 19 (9 pares + a nova).
    assert len(janela) == 19
    assert janela[0]["content"] == "pergunta 1"
    assert janela[-1]["content"] == "pergunta nova"


def test_papeis_continuam_alternando_depois_do_corte():
    msgs = _conversa(12) + [{"role": "user", "content": "pergunta nova"}]
    janela = aparar_historico(msgs, 20)
    papeis = [m["role"] for m in janela]
    assert all(a != b for a, b in zip(papeis, papeis[1:]))


def test_historico_vazio():
    assert aparar_historico([], 20) == []


def test_remove_tag_de_thinking():
    assert sem_tags("<thinking>Vou ver.</thinking> Sao dez graus.") == "Vou ver. Sao dez graus."


def test_remove_tag_partida_entre_frases():
    # O acumulador quebra por frase, entao a tag chega picada.
    assert sem_tags("<thinking>Preciso calcular.") == "Preciso calcular."
    assert sem_tags("</thinking> A resposta e quatro.") == "A resposta e quatro."


def test_remove_tag_com_atributos():
    assert sem_tags('<system-reminder foo="bar">Nada.') == "Nada."


def test_frase_que_sobra_vazia():
    """Quem consome deve poder descartar: uma frase so de tag nao vai para o TTS."""
    assert sem_tags("<thinking>") == ""


def test_nao_mexe_em_matematica():
    assert sem_tags("Sete e maior que 3 > 2.") == "Sete e maior que 3 > 2."
    assert sem_tags("Cinco < 7 e verdadeiro.") == "Cinco < 7 e verdadeiro."


def test_texto_normal_passa_intacto():
    frase = "Sao quinze graus em Sao Paulo."
    assert sem_tags(frase) is frase
