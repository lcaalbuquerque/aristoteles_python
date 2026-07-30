"""O cerebro: Claude via API oficial, em streaming, entregando frase por frase."""

from __future__ import annotations

import os
import time
from collections import deque
from pathlib import Path
from typing import Iterator

import anthropic
import httpx

from .config import LlmCfg
from .frases import AcumuladorDeFrases, sem_tags


def _credencial_limpa(arquivo: Path | None = None) -> dict[str, str]:
    """Acha a credencial e apara espaco em branco das pontas.

    Ordem: `ANTHROPIC_API_KEY`, `ANTHROPIC_AUTH_TOKEN`, o arquivo `arquivo`, e por
    fim a resolucao do proprio SDK (perfil do `ant auth login`).

    O arquivo existe para a chave ter **um lugar so**. Antes, o servico systemd
    lia uma copia em `~/.config/aristoteles/env`; rotacionar a chave atualizou o
    original e deixou a copia com a revogada, e todo pedido virou 401 -- com o
    agravante de o app funcionar no terminal e falhar so como servico.

    Motivo: um `\\n` no fim da chave nao vira erro de autenticacao, vira

        LocalProtocolError("Illegal header value b'sk-ant-...\\n'")

    embrulhado num APIConnectionError -- ou seja, o assistente reclama de rede
    quando o problema e a variavel de ambiente. Cabecalho HTTP nao aceita quebra
    de linha, e o SDK repassa o valor cru.

    Acontece facil: `export ANTHROPIC_API_KEY=$(cat arquivo)` apara sozinho, mas
    colar a chave no terminal ou num EnvironmentFile do systemd nao.

    Devolve kwargs vazios se nada for encontrado, para nao atropelar a resolucao
    do SDK pelo perfil do `ant auth login`.
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

    if arquivo is not None:
        caminho = Path(arquivo).expanduser()
        try:
            limpo = caminho.read_text(encoding="utf-8").strip()
        except OSError:
            return {}
        if limpo:
            print(f"[cerebro] credencial lida de {caminho}")
            return {"api_key": limpo}
    return {}


TIPOS_TRANSITORIOS = frozenset({"overloaded_error", "api_error"})
STATUS_TRANSITORIOS = frozenset({500, 502, 503, 504, 529})


def _tipo_do_erro(e: anthropic.APIStatusError) -> str:
    """Le `error.type` do corpo da resposta, se houver."""
    corpo = getattr(e, "body", None)
    if isinstance(corpo, dict):
        erro = corpo.get("error")
        if isinstance(erro, dict):
            return str(erro.get("type") or "")
    return ""


def eh_transitorio(e: Exception) -> bool:
    """Vale a pena tentar de novo?

    O caso que motivou isto: `overloaded_error` da Anthropic **dentro do stream**.
    O SDK monta a excecao a partir do status HTTP, e num erro no meio do stream o
    status e 200 -- os cabecalhos ja foram enviados. Resultado: nao vira
    `OverloadedError`, vira um `APIStatusError` cru com status_code 200, e nem o
    `max_retries` do SDK ajuda, porque para ele a requisicao *funcionou*.

    Ou seja, falha transitoria de servidor no meio do stream so pode ser retentada
    aqui. Identificamos pelo `error.type` do corpo, nao pelo status.
    """
    if isinstance(e, (anthropic.APITimeoutError, anthropic.APIConnectionError)):
        return True
    if isinstance(e, (anthropic.AuthenticationError, anthropic.PermissionDeniedError,
                      anthropic.BadRequestError, anthropic.NotFoundError,
                      anthropic.RequestTooLargeError,
                      anthropic.UnprocessableEntityError)):
        return False  # nao melhora esperando
    if isinstance(e, anthropic.APIStatusError):
        return (_tipo_do_erro(e) in TIPOS_TRANSITORIOS
                or e.status_code in STATUS_TRANSITORIOS)
    return False


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
            **_credencial_limpa(cfg.arquivo_chave),
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
            # AuthenticationError e RateLimitError primeiro: sao subclasses de
            # APIStatusError e nao devem cair no ramo de retentativa.
            except anthropic.AuthenticationError:
                self._historico.pop()
                print("[cerebro] credencial invalida ou revogada. Verifique ANTHROPIC_API_KEY.")
                yield "Minha chave de acesso foi recusada."
                return
            except anthropic.RateLimitError:
                # O SDK ja retentou o 429 com backoff antes de levantar; insistir
                # aqui so somaria silencio.
                self._historico.pop()
                yield "Atingi o limite de uso. Tente de novo em instantes."
                return
            except (anthropic.APIConnectionError, anthropic.APIStatusError) as e:
                ultima = tentativa >= self.cfg.reconexoes
                if not eh_transitorio(e) or falou_algo or ultima:
                    yield from self._explicar(e)
                    return
                espera = self.cfg.espera_reconexao_s * (2 ** tentativa)
                detalhe = _tipo_do_erro(e) or repr(getattr(e, "__cause__", None))
                print(f"[cerebro] {type(e).__name__} ({detalhe}); "
                      f"reconectando em {espera:.1f}s "
                      f"({tentativa + 1}/{self.cfg.reconexoes})")
                time.sleep(espera)

    def _explicar(self, e: Exception) -> Iterator[str]:
        """Mensagem falada para uma falha que nao deu para contornar."""
        self._historico.pop()
        if isinstance(e, anthropic.APITimeoutError):
            print(f"[cerebro] timeout ({type(e).__name__}): "
                  f"connect={self.cfg.timeout_conexao_s}s "
                  f"read={self.cfg.timeout_leitura_s}s; causa: {e.__cause__!r}")
            yield "Demorei demais para responder. Pergunte de novo."
        elif isinstance(e, anthropic.APIStatusError):
            tipo = _tipo_do_erro(e)
            print(f"[cerebro] erro da API (status {e.status_code}, tipo "
                  f"{tipo or 'desconhecido'}): {e.message}")
            if tipo == "overloaded_error":
                # Distingue do erro genérico: o problema e do outro lado e passa.
                yield "O servidor está sobrecarregado. Tente de novo em instantes."
            else:
                yield "Tive um problema para pensar agora."
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
