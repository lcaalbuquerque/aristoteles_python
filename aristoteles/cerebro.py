"""O cerebro: Claude via API oficial, em streaming, entregando frase por frase."""

from __future__ import annotations

from collections import deque
from typing import Iterator

import anthropic

from .config import LlmCfg
from .frases import AcumuladorDeFrases, sem_tags


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
        # Sem api_key explicita: o SDK resolve ANTHROPIC_API_KEY, ANTHROPIC_AUTH_TOKEN
        # ou o perfil do `ant auth login`.
        self._cliente = anthropic.Anthropic()
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
        if getattr(self._cliente, "api_key", None) or getattr(self._cliente, "auth_token", None):
            return
        raise RuntimeError(
            "Nenhuma credencial da API Anthropic encontrada.\n"
            "  export ANTHROPIC_API_KEY=sk-ant-...   (ou rode: ant auth login)"
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
        """Gera as frases da resposta conforme elas ficam prontas."""
        self._historico.append({"role": "user", "content": pergunta})
        self._aparar()
        acumulador = AcumuladorDeFrases()
        completa: list[str] = []

        try:
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
                yield "Desculpe, nao posso responder isso."
                self._historico.pop()  # nao guarda turno sem resposta
                return

            if completa:
                self._historico.append({"role": "assistant", "content": " ".join(completa)})
            else:
                self._historico.pop()

        except anthropic.APIConnectionError:
            self._historico.pop()
            yield "Estou sem conexao com a internet no momento."
        except anthropic.RateLimitError:
            self._historico.pop()
            yield "Atingi o limite de uso. Tente de novo em instantes."
        except anthropic.AuthenticationError:
            self._historico.pop()
            print("[cerebro] credencial invalida ou revogada. Verifique ANTHROPIC_API_KEY.")
            yield "Minha chave de acesso foi recusada."
        except anthropic.APIStatusError as e:
            self._historico.pop()
            print(f"[cerebro] erro da API ({e.status_code}): {e.message}")
            yield "Tive um problema para pensar agora."

    def esquecer(self) -> None:
        self._historico.clear()
