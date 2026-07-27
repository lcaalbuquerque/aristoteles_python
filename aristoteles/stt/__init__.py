"""Reconhecimento de fala. Dois backends, uma interface.

    cpu     -> faster-whisper (CTranslate2). CUDA-only, entao aqui roda em CPU.
    vulkan  -> whisper.cpp compilado com Vulkan, servindo HTTP. Usa a Radeon.

Trocar de um para o outro e uma linha no config.yaml (stt.backend).
"""

from __future__ import annotations

from typing import Protocol

import numpy as np

from ..config import SttCfg


class STT(Protocol):
    def transcrever(self, audio: np.ndarray) -> str:
        """audio: float32 mono 16 kHz em [-1, 1]. Retorna o texto (pode ser vazio)."""
        ...

    def aquecer(self) -> None:
        """Carrega o modelo / faz uma inferencia boba para tirar o custo do primeiro uso."""
        ...


def criar(cfg: SttCfg, taxa_amostragem: int) -> STT:
    if cfg.backend == "cpu":
        from .cpu_faster_whisper import FasterWhisperCPU

        return FasterWhisperCPU(cfg)
    if cfg.backend == "vulkan":
        from .vulkan_whisper_cpp import WhisperCppVulkan

        return WhisperCppVulkan(cfg, taxa_amostragem)
    raise ValueError(f"backend de STT desconhecido: {cfg.backend!r}")
