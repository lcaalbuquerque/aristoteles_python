"""Gatilho de ativacao.

Fases 0-4: push-to-talk (Enter). Fase 5: openWakeWord com modelo "Aristoteles"
treinado a partir de audio sintetico -- veja o README.
"""

from __future__ import annotations

from typing import Protocol

import numpy as np

from .audio.entrada import EntradaAudio
from .config import WakeCfg


class Gatilho(Protocol):
    def aguardar(self, entrada: EntradaAudio) -> bool:
        """Bloqueia ate o usuario chamar. False = encerrar o programa."""
        ...

    @property
    def usa_pre_roll(self) -> bool: ...


class PushToTalk:
    """Gatilho manual: Enter grava, Ctrl-D ou 'sair' encerra."""

    usa_pre_roll = False

    def aguardar(self, entrada: EntradaAudio) -> bool:
        try:
            resposta = input("\n[Enter] para falar, 'sair' para encerrar > ").strip().lower()
        except EOFError:
            return False
        if resposta in ("sair", "q", "quit", "exit"):
            return False
        entrada.limpar()  # descarta o que entrou enquanto esperava
        return True


class OpenWakeWord:
    """Escuta continua ate ouvir a palavra de ativacao."""

    usa_pre_roll = True

    def __init__(self, cfg: WakeCfg, raiz) -> None:
        from pathlib import Path

        import openwakeword
        from openwakeword.model import Model

        caminho = Path(cfg.modelo)
        if not caminho.is_absolute():
            caminho = raiz / caminho
        if not caminho.exists():
            raise FileNotFoundError(
                f"Modelo de wake word nao encontrado: {caminho}\n"
                "Treine o seu (veja README, fase 5) ou use wake.modo: push_to_talk"
            )
        openwakeword.utils.download_models()  # baixa o melspectrogram/embedding base
        self._modelo = Model(wakeword_models=[str(caminho)], inference_framework="onnx")
        self.cfg = cfg

    def aguardar(self, entrada: EntradaAudio) -> bool:
        self._modelo.reset()
        entrada.limpar()
        print("\nOuvindo... (diga 'Aristoteles')")
        while True:
            bloco = entrada.ler(timeout=1.0)
            if bloco is None:
                continue
            pontuacoes = self._modelo.predict(bloco)
            if max(pontuacoes.values(), default=0.0) >= self.cfg.limiar:
                return True


def criar(cfg: WakeCfg, raiz) -> Gatilho:
    if cfg.modo == "push_to_talk":
        return PushToTalk()
    if cfg.modo == "openwakeword":
        return OpenWakeWord(cfg, raiz)
    raise ValueError(f"wake.modo desconhecido: {cfg.modo!r}")
