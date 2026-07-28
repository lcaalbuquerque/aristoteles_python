"""Gatilho de ativacao.

Dois modos atras do mesmo `Protocol`: push-to-talk (Enter) e openWakeWord.

Alem de `aguardar()`, que bloqueia no ocioso, os gatilhos expoem `alimentar()`
para avaliar um bloco isolado. E o que permite vigiar o microfone *enquanto o
assistente fala* -- o barge-in da fase 6. Sem isso o loop principal teria de
escolher entre esperar a fala terminar e reimplementar a deteccao.
"""

from __future__ import annotations

import time
from typing import Protocol

import numpy as np

from .audio.entrada import EntradaAudio
from .config import WakeCfg


class Gatilho(Protocol):
    def aguardar(self, entrada: EntradaAudio) -> bool:
        """Bloqueia ate o usuario chamar. False = encerrar o programa."""
        ...

    def alimentar(self, bloco: np.ndarray) -> bool:
        """Avalia um bloco isolado. True = ouviu a palavra agora."""
        ...

    def reiniciar(self) -> None:
        """Zera o estado interno antes de comecar a escutar."""
        ...

    @property
    def usa_pre_roll(self) -> bool: ...

    @property
    def suporta_barge_in(self) -> bool: ...


class PushToTalk:
    """Gatilho manual: Enter grava, Ctrl-D ou 'sair' encerra."""

    usa_pre_roll = False
    # Sem barge-in: quem esta no teclado interrompe com Ctrl-C, e vigiar o
    # microfone durante a fala nao faria sentido num modo que ignora o microfone
    # no ocioso.
    suporta_barge_in = False

    def aguardar(self, entrada: EntradaAudio) -> bool:
        try:
            resposta = input("\n[Enter] para falar, 'sair' para encerrar > ").strip().lower()
        except EOFError:
            return False
        if resposta in ("sair", "q", "quit", "exit"):
            return False
        entrada.limpar()  # descarta o que entrou enquanto esperava
        return True

    def alimentar(self, bloco: np.ndarray) -> bool:
        return False

    def reiniciar(self) -> None:
        pass


class OpenWakeWord:
    """Escuta continua ate ouvir a palavra de ativacao."""

    usa_pre_roll = True
    suporta_barge_in = True

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
        self._mudo_ate = 0.0

    def reiniciar(self) -> None:
        self._modelo.reset()
        self._mudo_ate = 0.0

    def alimentar(self, bloco: np.ndarray) -> bool:
        """Avalia um bloco. Respeita o cooldown depois de um disparo.

        O cooldown existe porque uma unica pronuncia da palavra atravessa varias
        janelas do modelo e passa do limiar em mais de uma. Sem ele, interromper a
        fala com "Aristoteles" disparava de novo no bloco seguinte, e o assistente
        entrava e saia da gravacao.
        """
        agora = time.monotonic()
        if agora < self._mudo_ate:
            self._modelo.predict(bloco)  # mantem o buffer de features coerente
            return False
        pontuacoes = self._modelo.predict(bloco)
        if max(pontuacoes.values(), default=0.0) >= self.cfg.limiar:
            self._mudo_ate = agora + self.cfg.cooldown_s
            return True
        return False

    def aguardar(self, entrada: EntradaAudio) -> bool:
        self.reiniciar()
        entrada.limpar()
        print("\nOuvindo... (diga 'Aristoteles')")
        while True:
            bloco = entrada.ler(timeout=1.0)
            if bloco is None:
                continue
            if self.alimentar(bloco):
                return True


def criar(cfg: WakeCfg, raiz) -> Gatilho:
    if cfg.modo == "push_to_talk":
        return PushToTalk()
    if cfg.modo == "openwakeword":
        return OpenWakeWord(cfg, raiz)
    raise ValueError(f"wake.modo desconhecido: {cfg.modo!r}")
