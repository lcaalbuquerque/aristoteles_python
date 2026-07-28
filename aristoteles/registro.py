"""Copia para arquivo tudo que o app imprime no console.

Por que tee em `sys.stdout`/`sys.stderr` e nao converter os `print()` para
`logging`:

* O app usa `end=""` e `\\r` para reescrever a linha de status ao vivo
  ("  [ouvindo]" -> "\\r  voce (0.9s): ..."). O `logging` emite uma linha por
  chamada, entao converter quebraria a interface no terminal.
* O tee pega tambem o que **nao** e nosso: o `[audio]` do callback do PortAudio,
  avisos do onnxruntime, e tracebacks. Trocar os prints deixaria isso de fora,
  que e justamente o que se quer num log de diagnostico.

Limitacao honesta: bibliotecas que escrevem no descritor 2 em nivel C -- o
PortAudio faz isso em alguns erros de ALSA -- passam ao lado, porque a troca e no
objeto Python. Esses aparecem no terminal (e no `journalctl`, como servico) mas
nao no arquivo.
"""

from __future__ import annotations

import logging
import logging.handlers
import sys
import threading
from datetime import datetime
from pathlib import Path
from types import TracebackType

from .config import LogCfg

_FORMATO = "%(asctime)s %(fluxo)s %(message)s"


class _Fluxo:
    """Encaminha para o console e acumula linhas para o log.

    Precisa parecer um arquivo de texto o suficiente para as bibliotecas nao
    reclamarem, daí os delegates de `isatty`, `fileno` e afins. Em particular
    `isatty()` importa: quem escreve barra de progresso decide por ele, e mentir
    aqui mudaria o comportamento do que estamos observando.
    """

    def __init__(self, original, logger: logging.Logger, nome: str,
                 expandir_retorno: bool, trava: threading.Lock) -> None:
        self._original = original
        self._logger = logger
        self._nome = nome
        self._expandir = expandir_retorno
        self._trava = trava
        self._buffer = ""

    # --- escrita --------------------------------------------------------------

    def write(self, texto: str) -> int:
        n = self._original.write(texto)
        with self._trava:
            self._buffer += texto
            self._drenar()
        return n

    def _drenar(self) -> None:
        """Emite as linhas completas do buffer, guardando o resto."""
        if self._expandir:
            # Trata `\r` como fim de linha: cada reescrita da linha de status vira
            # sua propria entrada, em vez de um emaranhado com retornos no meio.
            self._buffer = self._buffer.replace("\r\n", "\n").replace("\r", "\n")
        while "\n" in self._buffer:
            linha, _, resto = self._buffer.partition("\n")
            self._buffer = resto
            linha = linha.rstrip()
            if linha:
                self._logger.info(linha, extra={"fluxo": self._nome})

    def fechar_linha_pendente(self) -> None:
        """Grava o que sobrou sem `\\n` -- tipico de `print(..., end="")`."""
        with self._trava:
            pendente = self._buffer.strip()
            self._buffer = ""
        if pendente:
            self._logger.info(pendente, extra={"fluxo": self._nome})

    def flush(self) -> None:
        self._original.flush()

    # --- fachada de arquivo ---------------------------------------------------

    def isatty(self) -> bool:
        return self._original.isatty()

    def fileno(self) -> int:
        return self._original.fileno()

    def writable(self) -> bool:
        return True

    def readable(self) -> bool:
        return False

    def seekable(self) -> bool:
        return False

    @property
    def encoding(self) -> str:
        return getattr(self._original, "encoding", "utf-8")

    @property
    def errors(self):
        return getattr(self._original, "errors", None)

    def writelines(self, linhas) -> None:
        for linha in linhas:
            self.write(linha)

    def __getattr__(self, nome):  # pragma: no cover - fachada para o resto
        return getattr(self._original, nome)


class Registro:
    """Ativa o tee enquanto viver. Use como context manager."""

    def __init__(self, cfg: LogCfg, raiz: Path) -> None:
        self.cfg = cfg
        self.caminho = self._resolver(cfg.arquivo, raiz)
        self._logger: logging.Logger | None = None
        self._handler: logging.Handler | None = None
        self._antigos: tuple | None = None
        self._fluxos: list[_Fluxo] = []

    @staticmethod
    def _resolver(arquivo: Path, raiz: Path) -> Path:
        p = Path(arquivo).expanduser()
        return p if p.is_absolute() else raiz / p

    def __enter__(self) -> "Registro":
        if not self.cfg.ativo:
            return self
        self.caminho.parent.mkdir(parents=True, exist_ok=True)
        self._handler = logging.handlers.RotatingFileHandler(
            self.caminho, maxBytes=self.cfg.max_bytes,
            backupCount=self.cfg.backups, encoding="utf-8")
        self._handler.setFormatter(logging.Formatter(_FORMATO, "%Y-%m-%d %H:%M:%S"))

        self._logger = logging.getLogger("aristoteles.console")
        self._logger.setLevel(logging.INFO)
        self._logger.propagate = False  # nao duplica no root, que voltaria ao console
        self._logger.handlers.clear()
        self._logger.addHandler(self._handler)

        # Uma trava para os dois fluxos: stdout e stderr do mesmo processo, e as
        # threads de barge-in e de reproducao tambem imprimem.
        trava = threading.Lock()
        self._antigos = (sys.stdout, sys.stderr)
        saida = _Fluxo(sys.stdout, self._logger, "out", self.cfg.expandir_retorno, trava)
        erro = _Fluxo(sys.stderr, self._logger, "ERR", self.cfg.expandir_retorno, trava)
        self._fluxos = [saida, erro]
        sys.stdout, sys.stderr = saida, erro

        self._logger.info("=" * 60, extra={"fluxo": "---"})
        self._logger.info(f"sessao iniciada em {datetime.now():%d/%m/%Y %H:%M:%S}",
                          extra={"fluxo": "---"})
        return self

    def __exit__(self, tipo: type[BaseException] | None, valor: BaseException | None,
                 tb: TracebackType | None) -> None:
        if not self.cfg.ativo or self._antigos is None:
            return
        for f in self._fluxos:
            f.fechar_linha_pendente()
        if self._logger is not None:
            self._logger.info("sessao encerrada", extra={"fluxo": "---"})
        sys.stdout, sys.stderr = self._antigos
        self._antigos = None
        if self._handler is not None:
            self._handler.close()
            if self._logger is not None:
                self._logger.removeHandler(self._handler)
            self._handler = None
        self._fluxos = []
