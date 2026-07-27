"""Backend de STT na GPU AMD via whisper.cpp + Vulkan.

Por que Vulkan e nao ROCm: a Polaris (gfx803, sua RX 470/480/570/580/590) saiu
do suporte do ROCm a partir da 5.x, entao PyTorch+ROCm nao e opcao nessa placa.
O backend Vulkan do ggml roda bem nela usando o driver Mesa RADV, que ja esta
instalado no sistema.

Falamos com o `whisper-server` por HTTP em vez de invocar o binario a cada frase:
assim o modelo fica residente e nao se paga o carregamento (1-2 s) toda vez.

Suba o servidor com: ./scripts/03_servidor_whisper.sh
"""

from __future__ import annotations

import io
import wave

import numpy as np
import requests

from ..config import SttCfg


class WhisperCppVulkan:
    def __init__(self, cfg: SttCfg, taxa_amostragem: int) -> None:
        self.cfg = cfg
        self.taxa = taxa_amostragem

    def aquecer(self) -> None:
        base = self.cfg.servidor_url.rsplit("/", 1)[0]
        try:
            requests.get(base + "/", timeout=3)
        except requests.RequestException as e:
            raise RuntimeError(
                f"whisper-server nao respondeu em {self.cfg.servidor_url}.\n"
                "Suba com: ./scripts/03_servidor_whisper.sh  "
                "(ou troque stt.backend para 'cpu' no config.yaml)"
            ) from e

    def transcrever(self, audio: np.ndarray) -> str:
        wav = _para_wav(audio, self.taxa)
        try:
            resposta = requests.post(
                self.cfg.servidor_url,
                files={"file": ("audio.wav", wav, "audio/wav")},
                data={
                    "temperature": "0.0",
                    "language": self.cfg.idioma,
                    "response_format": "json",
                },
                timeout=60,
            )
            resposta.raise_for_status()
        except requests.RequestException as e:
            print(f"[stt] falha no whisper-server: {e}")
            return ""

        try:
            return (resposta.json().get("text") or "").strip()
        except ValueError:
            return resposta.text.strip()


def _para_wav(audio: np.ndarray, taxa: int) -> bytes:
    pcm = np.clip(audio, -1.0, 1.0)
    pcm = (pcm * 32767).astype(np.int16)
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(taxa)
        w.writeframes(pcm.tobytes())
    return buf.getvalue()
