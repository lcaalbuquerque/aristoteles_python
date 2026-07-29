"""A janela de historico, o filtro de tags e o tratamento de erro da API.

A maior parte nao instancia `Cerebro`: o construtor exige credencial da API. Os
testes de erro injetam uma credencial falsa e um cliente dublê, sem tocar na rede.
"""

from pathlib import Path

import anthropic
import httpx
import pytest

from aristoteles.cerebro import Cerebro, aparar_historico
from aristoteles.config import LlmCfg
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


# --- credencial ----------------------------------------------------------------

def test_newline_no_fim_da_chave_e_aparado(monkeypatch, capsys):
    """A regressao relatada.

    Uma chave terminando em \\n nao da erro de autenticacao: da
    LocalProtocolError("Illegal header value ..."), que o SDK embrulha em
    APIConnectionError -- e o assistente passa a culpar a rede.
    """
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-api03-abc\n")
    c = Cerebro(LlmCfg())
    assert c._cliente.api_key == "sk-ant-api03-abc"
    assert "quebra de linha" in capsys.readouterr().out


def test_espacos_em_volta_da_chave_sao_aparados(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "  sk-ant-api03-abc\t\r\n ")
    assert Cerebro(LlmCfg())._cliente.api_key == "sk-ant-api03-abc"


def test_chave_limpa_nao_gera_aviso(monkeypatch, capsys):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-api03-abc")
    Cerebro(LlmCfg())
    assert capsys.readouterr().out == ""


def test_newline_no_meio_da_chave_falha_na_inicializacao(monkeypatch):
    """`strip()` nao resolve isso -- e melhor recusar do que culpar a rede depois."""
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-api03\nabc")
    with pytest.raises(RuntimeError, match="cabeçalho HTTP"):
        Cerebro(LlmCfg())


def test_credencial_ausente_falha_com_instrucao(monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.delenv("ANTHROPIC_AUTH_TOKEN", raising=False)
    # o SDK tambem procura o perfil do `ant auth login`; forca a ausencia
    monkeypatch.setattr(anthropic, "Anthropic",
                        lambda **kw: type("C", (), {"api_key": None, "auth_token": None,
                                                    "timeout": None, "max_retries": 0})())
    with pytest.raises(RuntimeError, match="Nenhuma credencial"):
        Cerebro(LlmCfg())


def test_le_a_chave_do_arquivo_quando_o_ambiente_nao_tem(monkeypatch, tmp_path):
    """A regressao relatada.

    O servico systemd nao herda o ambiente do login, entao lia uma copia da chave
    em ~/.config/aristoteles/env. Rotacionar a chave atualizou o original e deixou
    a copia com a revogada: funcionava no terminal e devolvia 401 so como servico.
    Com o app lendo o arquivo original, a chave tem um lugar so.
    """
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.delenv("ANTHROPIC_AUTH_TOKEN", raising=False)
    arq = tmp_path / "chave"
    arq.write_text("sk-ant-do-arquivo\n", encoding="utf-8")

    c = Cerebro(LlmCfg(arquivo_chave=arq))
    assert c._cliente.api_key == "sk-ant-do-arquivo"  # e sem o \n


def test_ambiente_tem_prioridade_sobre_o_arquivo(monkeypatch, tmp_path):
    """Quem exporta a variavel esta sendo explicito; nao atropelamos."""
    arq = tmp_path / "chave"
    arq.write_text("sk-ant-do-arquivo", encoding="utf-8")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-do-ambiente")
    assert Cerebro(LlmCfg(arquivo_chave=arq))._cliente.api_key == "sk-ant-do-ambiente"


def test_arquivo_de_chave_ausente_nao_explode(monkeypatch, tmp_path):
    from aristoteles.cerebro import _credencial_limpa

    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.delenv("ANTHROPIC_AUTH_TOKEN", raising=False)
    assert _credencial_limpa(tmp_path / "nao_existe") == {}


def test_arquivo_de_chave_vazio_conta_como_ausente(monkeypatch, tmp_path):
    from aristoteles.cerebro import _credencial_limpa

    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.delenv("ANTHROPIC_AUTH_TOKEN", raising=False)
    arq = tmp_path / "vazio"
    arq.write_text("   \n", encoding="utf-8")
    assert _credencial_limpa(arq) == {}


def test_til_no_caminho_do_arquivo_e_expandido(monkeypatch, tmp_path):
    from aristoteles.cerebro import _credencial_limpa

    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.delenv("ANTHROPIC_AUTH_TOKEN", raising=False)
    monkeypatch.setenv("HOME", str(tmp_path))
    (tmp_path / ".anthropic_api_key").write_text("sk-ant-via-til", encoding="utf-8")
    assert _credencial_limpa(Path("~/.anthropic_api_key")) == {"api_key": "sk-ant-via-til"}


def test_chave_so_com_espaco_conta_como_ausente(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "   \n")
    monkeypatch.delenv("ANTHROPIC_AUTH_TOKEN", raising=False)
    from aristoteles.cerebro import _credencial_limpa
    assert _credencial_limpa() == {}


# --- tratamento de erro da API -------------------------------------------------

def _cerebro(monkeypatch, erro: Exception, reconexoes: int = 0) -> Cerebro:
    """Cerebro cujo `messages.stream` levanta `erro`. Nao toca na rede.

    `reconexoes=0` por default: estes testes checam a mensagem falada, nao a
    reconexao, e esperar de verdade so deixaria a suite lenta.
    """
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-teste")
    c = Cerebro(LlmCfg(reconexoes=reconexoes, espera_reconexao_s=0.0))

    def estoura(*_a, **_k):
        raise erro

    monkeypatch.setattr(c._cliente.messages, "stream", estoura)
    return c


def test_timeout_nao_e_anunciado_como_falta_de_internet(monkeypatch, capsys):
    """A regressao relatada: rede perfeitamente sa, e o assistente afirmava estar
    sem internet.

    `APITimeoutError` e subclasse de `APIConnectionError`, entao o `except` da
    conexao capturava os timeouts e dava um diagnostico falso -- que manda o
    usuario depurar a rede errada.
    """
    assert issubclass(anthropic.APITimeoutError, anthropic.APIConnectionError)

    erro = anthropic.APITimeoutError(request=httpx.Request("POST", "http://x"))
    c = _cerebro(monkeypatch, erro)
    dito = " ".join(c.responder("oi"))

    assert "internet" not in dito.lower()
    assert "demorei" in dito.lower()
    assert "timeout" in capsys.readouterr().out.lower()  # deixa pista no log


def test_falha_de_conexao_registra_a_causa(monkeypatch, capsys):
    """Era o unico handler sem print: o usuario ficava sem nenhuma pista."""
    causa = OSError("Name or service not known")
    erro = anthropic.APIConnectionError(request=httpx.Request("POST", "http://x"))
    erro.__cause__ = causa

    c = _cerebro(monkeypatch, erro)
    dito = " ".join(c.responder("oi"))

    assert "servidor" in dito.lower()
    assert "Name or service not known" in capsys.readouterr().out


def test_reconecta_e_responde_se_nada_foi_falado(monkeypatch, capsys):
    """Uma queda antes da primeira frase nao deve custar a pergunta ao usuario."""
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-teste")
    c = Cerebro(LlmCfg(reconexoes=2, espera_reconexao_s=0.0))

    chamadas = {"n": 0}

    def transmitir():
        chamadas["n"] += 1
        if chamadas["n"] == 1:
            raise anthropic.APIConnectionError(request=httpx.Request("POST", "http://x"))
        yield "Pronto."

    monkeypatch.setattr(c, "_transmitir", transmitir)
    assert list(c.responder("oi")) == ["Pronto."]
    assert chamadas["n"] == 2
    assert "reconectando" in capsys.readouterr().out


def test_nao_reconecta_depois_de_ja_ter_falado(monkeypatch):
    """Repetir a resposta do zero e pior que admitir o erro: o alto-falante nao
    tem desfazer."""
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-teste")
    c = Cerebro(LlmCfg(reconexoes=2, espera_reconexao_s=0.0))

    chamadas = {"n": 0}

    def transmitir():
        chamadas["n"] += 1
        yield "Primeira frase."
        raise anthropic.APIConnectionError(request=httpx.Request("POST", "http://x"))

    monkeypatch.setattr(c, "_transmitir", transmitir)
    dito = list(c.responder("oi"))
    assert chamadas["n"] == 1, "retentou apesar de ja ter falado"
    assert dito[0] == "Primeira frase."
    assert "servidor" in dito[-1].lower()


def test_desiste_depois_das_reconexoes_configuradas(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-teste")
    c = Cerebro(LlmCfg(reconexoes=3, espera_reconexao_s=0.0))

    chamadas = {"n": 0}

    def transmitir():
        chamadas["n"] += 1
        raise anthropic.APITimeoutError(request=httpx.Request("POST", "http://x"))
        yield  # pragma: no cover

    monkeypatch.setattr(c, "_transmitir", transmitir)
    dito = list(c.responder("oi"))
    assert chamadas["n"] == 4          # 1 tentativa + 3 reconexoes
    assert "demorei" in dito[-1].lower()
    assert list(c._historico) == []    # nao deixa pergunta orfa


def test_espera_dobra_entre_reconexoes(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-teste")
    c = Cerebro(LlmCfg(reconexoes=3, espera_reconexao_s=0.5))
    esperas = []
    monkeypatch.setattr("aristoteles.cerebro.time.sleep", esperas.append)

    def transmitir():
        raise anthropic.APIConnectionError(request=httpx.Request("POST", "http://x"))
        yield  # pragma: no cover

    monkeypatch.setattr(c, "_transmitir", transmitir)
    list(c.responder("oi"))
    assert esperas == [0.5, 1.0, 2.0]


def test_erro_nao_de_rede_nao_reconecta(monkeypatch):
    """Recusa de credencial nao melhora esperando."""
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-teste")
    c = Cerebro(LlmCfg(reconexoes=3, espera_reconexao_s=0.0))

    chamadas = {"n": 0}

    def transmitir():
        chamadas["n"] += 1
        raise anthropic.AuthenticationError(
            "nao", response=httpx.Response(401, request=httpx.Request("GET", "http://x")),
            body=None)
        yield  # pragma: no cover

    monkeypatch.setattr(c, "_transmitir", transmitir)
    dito = list(c.responder("oi"))
    assert chamadas["n"] == 1
    assert "chave" in dito[-1].lower()


def test_erro_nao_deixa_pergunta_orfa_no_historico(monkeypatch):
    """Turno sem resposta na frente do historico faria a proxima chamada dar 400."""
    erro = anthropic.APITimeoutError(request=httpx.Request("POST", "http://x"))
    c = _cerebro(monkeypatch, erro)
    list(c.responder("pergunta que falhou"))
    assert list(c._historico) == []


def test_timeouts_vem_da_config(monkeypatch):
    """Default do SDK e read=600s: dez minutos de silencio num assistente de voz."""
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-teste")
    c = Cerebro(LlmCfg(timeout_conexao_s=3.0, timeout_leitura_s=7.0, tentativas=1))
    assert c._cliente.timeout.connect == 3.0
    assert c._cliente.timeout.read == 7.0
    assert c._cliente.max_retries == 1
    assert LlmCfg().timeout_leitura_s < 600
