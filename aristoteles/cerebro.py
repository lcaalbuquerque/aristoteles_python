"""O cerebro: Claude via API oficial, em streaming, entregando frase por frase."""

from __future__ import annotations

import os
import time
from collections import deque
from typing import Iterator

import anthropic
import httpx

from .config import LlmCfg
from .frases import AcumuladorDeFrases, sem_tags


def _credencial_limpa() -> dict[str, str]:
    """Le a credencial do ambiente e apara espaco em branco das pontas.

    Motivo: um `\\n` no fim da chave nao vira erro de autenticacao, vira

        LocalProtocolError("Illegal header value b'sk-ant-...\\n'")

    embrulhado num APIConnectionError -- ou seja, o assistente reclama de rede
    quando o problema e a variavel de ambiente. Cabecalho HTTP nao aceita quebra
    de linha, e o SDK repassa o valor cru.

    Acontece facil: `export ANTHROPIC_API_KEY=$(cat arquivo)` apara sozinho, mas
    colar a chave no terminal ou num EnvironmentFile do systemd nao.

    Devolve kwargs vazios se nada estiver no ambiente, para nao atropelar a
    resolucao do SDK pelo perfil do `ant auth login`.
    """
    for var, kwarg in (("ANTHROPIC_API_KEY", "api_key"),
                       ("ANTHROPIC_AUTH_TOKEN", "auth_token")):
        bruto = os.environ.get(var)
        if not bruto:
            continue
        limpo = bruto.strip()
        if not limpo:
            continue
        if limpo != bruto:
            print(f"[cerebro] {var} tinha espaço/quebra de linha nas pontas; "
                  "aparei. Corrija a origem para evitar erro noutro cliente.")
        return {kwarg: limpo}
    return {}


def aparar_historico(msgs: list[dict], max_msgs: int) -> list[dict]:
    """Corta a janela pela esquerda, sempre parando num turno do usuario.

    A API exige role "user" na primeira mensagem. Descartar mensagem a mensagem
    -- que e o que um `deque(maxlen=N)` faz sozinho -- deixa a *resposta* do
    turno mais antigo na primeira posicao e a API devolve 400. Com
    turnos_historico=10 isso acontecia a partir da 11a pergunta, e depois
    alternava: 11 falha, 12 funciona, 13 falha.
    """
    while len(msgs) > max_msgs:
        msgs.pop(0)
    while msgs and msgs[0]["role"] != "user":
        msgs.pop(0)
    return msgs


class Cerebro:
    def __init__(self, cfg: LlmCfg) -> None:
        self.cfg = cfg
        self._cliente = anthropic.Anthropic(
            **_credencial_limpa(),
            timeout=httpx.Timeout(
                connect=cfg.timeout_conexao_s,
                read=cfg.timeout_leitura_s,
                write=cfg.timeout_leitura_s,
                pool=cfg.timeout_leitura_s,
            ),
            max_retries=cfg.tentativas,
        )
        self._checar_credenciais()
        # Sem maxlen: quem apara e o aparar_historico(), que descarta turnos
        # inteiros em vez de mensagens soltas.
        self._historico: deque[dict] = deque()
        self._max_msgs = max(2, cfg.turnos_historico * 2)

    def _checar_credenciais(self) -> None:
        """Falha na inicializacao, nao na primeira pergunta.

        O SDK atual nao valida a chave ao construir o cliente, e AuthenticationError
        e subclasse de APIStatusError -- sem esta checagem o usuario ouviria um
        vago "tive um problema para pensar" em vez de saber que falta a chave.
        """
        credencial = (getattr(self._cliente, "api_key", None)
                      or getattr(self._cliente, "auth_token", None))
        if not credencial:
            raise RuntimeError(
                "Nenhuma credencial da API Anthropic encontrada.\n"
                "  export ANTHROPIC_API_KEY=sk-ant-...   (ou rode: ant auth login)"
            )
        # O `_credencial_limpa()` apara as pontas, mas um \n no MEIO da chave
        # continuaria passando e estourando como LocalProtocolError na primeira
        # pergunta -- que o SDK embrulha em APIConnectionError, fazendo o
        # assistente culpar a rede. Melhor recusar aqui, com o motivo certo.
        ruins = {c for c in credencial if not 0x20 <= ord(c) < 0x7F}
        if ruins:
            raise RuntimeError(
                "A credencial da API tem caractere que não cabe em cabeçalho HTTP: "
                f"{sorted(repr(c) for c in ruins)}\n"
                "  Reveja ANTHROPIC_API_KEY (quebra de linha ou espaço no meio?).\n"
                "  Se o valor vem de arquivo: export ANTHROPIC_API_KEY=\"$(cat arquivo)\""
            )

    def _params_thinking(self) -> dict:
        if self.cfg.pensar:
            # Necessario se voce adicionar ferramentas: com thinking desligado o
            # modelo as vezes escreve a chamada como texto em vez de emitir tool_use.
            return {"type": "adaptive"}
        return {"type": "disabled"}

    def _aparar(self) -> None:
        janela = aparar_historico(list(self._historico), self._max_msgs)
        self._historico.clear()
        self._historico.extend(janela)

    def responder(self, pergunta: str) -> Iterator[str]:
        """Gera as frases da resposta, retentando falhas de rede transitorias.

        O SDK ja retenta internamente (`llm.tentativas`), mas so ate o momento em
        que o stream comeca a chegar: se a conexao cai no meio, ele nao remonta a
        chamada. Aqui reemitimos o turno inteiro -- **mas somente enquanto nada
        tiver sido falado**. Depois da primeira frase no alto-falante nao ha como
        voltar atras, e repetir a resposta do zero seria pior que admitir o erro.
        """
        self._historico.append({"role": "user", "content": pergunta})
        self._aparar()

        for tentativa in range(self.cfg.reconexoes + 1):
            falou_algo = False
            try:
                for frase in self._transmitir():
                    falou_algo = True
                    yield frase
                return
            except (anthropic.APITimeoutError, anthropic.APIConnectionError) as e:
                ultima = tentativa >= self.cfg.reconexoes
                if falou_algo or ultima:
                    yield from self._explicar(e)
                    return
                espera = self.cfg.espera_reconexao_s * (2 ** tentativa)
                print(f"[cerebro] {type(e).__name__}: {e.__cause__!r}; "
                      f"reconectando em {espera:.1f}s "
                      f"({tentativa + 1}/{self.cfg.reconexoes})")
                time.sleep(espera)
            except anthropic.RateLimitError:
                self._historico.pop()
                yield "Atingi o limite de uso. Tente de novo em instantes."
                return
            except anthropic.AuthenticationError:
                self._historico.pop()
                print("[cerebro] credencial invalida ou revogada. Verifique ANTHROPIC_API_KEY.")
                yield "Minha chave de acesso foi recusada."
                return
            except anthropic.APIStatusError as e:
                self._historico.pop()
                print(f"[cerebro] erro da API ({e.status_code}): {e.message}")
                yield "Tive um problema para pensar agora."
                return

    def _explicar(self, e: Exception) -> Iterator[str]:
        """Mensagem falada para uma falha de rede que nao deu para contornar."""
        self._historico.pop()
        if isinstance(e, anthropic.APITimeoutError):
            print(f"[cerebro] timeout ({type(e).__name__}): "
                  f"connect={self.cfg.timeout_conexao_s}s "
                  f"read={self.cfg.timeout_leitura_s}s; causa: {e.__cause__!r}")
            yield "Demorei demais para responder. Pergunte de novo."
        else:
            print(f"[cerebro] falha de conexao ({type(e).__name__}): {e.__cause__!r}")
            yield "Nao consegui falar com o servidor agora."

    def _transmitir(self) -> Iterator[str]:
        """Uma tentativa: abre o stream e entrega as frases prontas."""
        acumulador = AcumuladorDeFrases()
        completa: list[str] = []

        with self._cliente.messages.stream(
            model=self.cfg.modelo,
            max_tokens=self.cfg.max_tokens,
            system=self.cfg.prompt_sistema,
            thinking=self._params_thinking(),
            output_config={"effort": self.cfg.effort},
            messages=list(self._historico),
        ) as stream:
            for pedaco in stream.text_stream:
                for frase in acumulador.alimentar(pedaco):
                    if frase := sem_tags(frase):
                        completa.append(frase)
                        yield frase
            for frase in acumulador.finalizar():
                if frase := sem_tags(frase):
                    completa.append(frase)
                    yield frase

            final = stream.get_final_message()

        if final.stop_reason == "refusal":
            self._historico.pop()  # nao guarda turno sem resposta
            yield "Desculpe, nao posso responder isso."
            return

        if completa:
            self._historico.append({"role": "assistant", "content": " ".join(completa)})
        else:
            self._historico.pop()

    def esquecer(self) -> None:
        self._historico.clear()
