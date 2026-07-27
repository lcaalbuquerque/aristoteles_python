"""Backend de STT em CPU via faster-whisper (CTranslate2).

CTranslate2 nao tem backend AMD/ROCm -- e CUDA ou CPU. Com 16 nucleos e o modelo
`small` em int8, uma frase curta sai em ~0,5-1,5 s, o que ja e utilizavel.
Para usar a Radeon, veja o backend `vulkan`.
"""

from __future__ import annotations

import numpy as np

from ..config import SttCfg


class FasterWhisperCPU:
    def __init__(self, cfg: SttCfg) -> None:
        self.cfg = cfg
        self._modelo = None

    def _carregar(self):
        if self._modelo is None:
            from faster_whisper import WhisperModel  # import tardio: ~2 s

            self._modelo = WhisperModel(
                self.cfg.modelo_cpu,
                device="cpu",
                compute_type=self.cfg.compute_type,
                cpu_threads=self.cfg.threads,
            )
        return self._modelo

    def aquecer(self) -> None:
        modelo = self._carregar()
        silencio = np.zeros(16_000, dtype=np.float32)
        list(modelo.transcribe(silencio, language=self.cfg.idioma, beam_size=1)[0])

    def transcrever(self, audio: np.ndarray) -> str:
        modelo = self._carregar()
        segmentos, _info = modelo.transcribe(
            audio,
            language=self.cfg.idioma,
            beam_size=1,           # greedy: bem mais rapido, diferenca minima em frase curta
            vad_filter=False,      # ja fizemos VAD no endpointing
            condition_on_previous_text=False,  # evita alucinacao em audio curto
        )
        return " ".join(s.text.strip() for s in segmentos).strip()
