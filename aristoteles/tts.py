"""Sintese de voz local com Piper (onnxruntime, CPU).

Piper roda bem mais rapido que tempo real em CPU, entao a GPU nao faz falta aqui.
A API do pacote piper-tts mudou entre versoes; o shim abaixo cobre as duas formas.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterator

import numpy as np

from .config import TtsCfg


class Voz:
    def __init__(self, cfg: TtsCfg, raiz: Path) -> None:
        caminho = Path(cfg.voz)
        if not caminho.is_absolute():
            caminho = raiz / caminho
        if not caminho.exists():
            raise FileNotFoundError(
                f"Voz do Piper nao encontrada: {caminho}\n"
                "Baixe com: ./scripts/02_baixar_modelos.sh"
            )

        from piper import PiperVoice

        self._voz = PiperVoice.load(str(caminho))
        self.cfg = cfg
        self.taxa_amostragem = self._resolver_taxa()
        self._syn_config = self._montar_syn_config()

    def _montar_syn_config(self):
        """length_scale controla a velocidade (>1 = mais lento). Só existe no piper >= 1.3."""
        if self.cfg.velocidade == 1.0:
            return None
        try:
            from piper import SynthesisConfig
        except ImportError:
            print("[tts] versao do piper nao suporta tts.velocidade; ignorando")
            return None
        return SynthesisConfig(length_scale=self.cfg.velocidade)

    def _resolver_taxa(self) -> int:
        config = getattr(self._voz, "config", None)
        taxa = getattr(config, "sample_rate", None)
        if taxa is None and isinstance(config, dict):
            taxa = config.get("audio", {}).get("sample_rate")
        return int(taxa or 22_050)

    def sintetizar(self, texto: str) -> Iterator[np.ndarray]:
        """Gera blocos int16 mono a `self.taxa_amostragem`."""
        texto = texto.strip()
        if not texto:
            return

        # piper-tts >= 1.3: synthesize() devolve AudioChunk
        if hasattr(self._voz, "synthesize"):
            for chunk in self._sintetizar_novo(texto):
                dados = getattr(chunk, "audio_int16_bytes", None)
                if dados is None:  # versao que ja devolve ndarray
                    yield np.asarray(chunk, dtype=np.int16)
                else:
                    yield np.frombuffer(dados, dtype=np.int16)
            return

        # piper-tts < 1.3
        for bruto in self._voz.synthesize_stream_raw(texto):
            yield np.frombuffer(bruto, dtype=np.int16)

    def _sintetizar_novo(self, texto: str):
        if self._syn_config is None:
            return self._voz.synthesize(texto)
        try:
            return self._voz.synthesize(texto, syn_config=self._syn_config)
        except TypeError:  # nome do parametro mudou entre versoes
            self._syn_config = None
            return self._voz.synthesize(texto)
