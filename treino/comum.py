"""Constantes e utilidades de audio compartilhadas pelo treino.

Os numeros nao sao arbitrarios -- vem do que o openWakeWord faz em
`openwakeword/train.py`, e mudar um deles quebra a compatibilidade com o
`openwakeword.model.Model` que o runtime usa para carregar o .onnx:

* `TAXA` 16 kHz: exigido pelo modelo de melspectrogram.
* `AMOSTRAS_CLIPE` 32000 (2,0 s): o train.py calcula
  `round(duracao_mediana/1000)*1000 + 12000` com piso de 32000. "Aristoteles"
  dura ~0,91 s -> 27000, que cai no piso. Mesmo valor dos modelos prontos.
* `FRAMES` 16 x `DIM` 96: e o que `AudioFeatures.get_embedding_shape(2)` devolve.
  Cuidado: 16 frames NAO sao 16 x 80 ms. O embedding usa janela de 76 frames de
  melspectrogram (760 ms) com passo de 8 (80 ms), entao ha um deslocamento fixo:
  frames = floor((duracao_ms/10 - 76)/8) + 1.
"""

from __future__ import annotations

import wave
from pathlib import Path

import numpy as np
import scipy.signal

TAXA = 16_000
AMOSTRAS_CLIPE = 32_000  # 2,0 s
FRAMES = 16
DIM = 96

PALAVRA = "Aristóteles"


def reamostrar(audio: np.ndarray, de: int, para: int = TAXA) -> np.ndarray:
    """Reamostra int16 -> int16. As vozes medium do Piper saem a 22050 Hz."""
    if de == para:
        return audio.astype(np.int16)
    n = int(round(len(audio) * para / de))
    return scipy.signal.resample(audio.astype(np.float32), n).astype(np.int16)


def _para_int16(bruto: bytes, largura: int) -> np.ndarray:
    """Converte PCM de 8/16/24/32 bits para int16.

    O 24 bits nao e gratuito: as respostas impulsivas do MIT vem em
    WAVE_FORMAT_EXTENSIBLE de 24 bits, e o numpy nao tem esse dtype -- monta-se
    o int32 a partir dos 3 bytes, com o byte baixo em zero para preservar o sinal.
    """
    if largura == 1:  # PCM de 8 bits e sem sinal
        return ((np.frombuffer(bruto, dtype=np.uint8).astype(np.int16) - 128) * 256)
    if largura == 2:
        return np.frombuffer(bruto, dtype=np.int16)
    if largura == 3:
        b = np.frombuffer(bruto, dtype=np.uint8).reshape(-1, 3).astype(np.uint32)
        v = (b[:, 0] << 8) | (b[:, 1] << 16) | (b[:, 2] << 24)
        return (v.astype(np.int32) >> 16).astype(np.int16)
    if largura == 4:
        return (np.frombuffer(bruto, dtype=np.int32) >> 16).astype(np.int16)
    raise ValueError(f"largura de amostra não suportada: {largura} bytes")


def ler_wav(caminho: Path) -> np.ndarray:
    """Le um wav mono PCM e reamostra para 16 kHz se preciso."""
    with wave.open(str(caminho), "rb") as f:
        canais = f.getnchannels()
        audio = _para_int16(f.readframes(f.getnframes()), f.getsampwidth())
        if canais > 1:
            audio = audio.reshape(-1, canais)[:, 0]
        return reamostrar(audio, f.getframerate())


def escrever_wav(caminho: Path, audio: np.ndarray, taxa: int = TAXA) -> None:
    caminho.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(caminho), "wb") as f:
        f.setnchannels(1)
        f.setsampwidth(2)
        f.setframerate(taxa)
        f.writeframes(audio.astype(np.int16).tobytes())


def aparar_silencio(audio: np.ndarray, limiar_rel: float = 0.02) -> np.ndarray:
    """Corta silencio das pontas, relativo ao pico do proprio clipe.

    O Piper e as gravacoes vem com folga nas bordas; sem aparar, o posicionamento
    aleatorio dentro da janela de 2 s fica preso no meio e o modelo aprende a
    esperar a palavra sempre na mesma posicao.
    """
    if audio.size == 0:
        return audio
    env = np.abs(audio.astype(np.float32))
    pico = env.max()
    if pico <= 0:
        return audio
    acima = np.flatnonzero(env > limiar_rel * pico)
    if acima.size == 0:
        return audio
    # 30 ms de folga para nao cortar plosiva inicial/final
    folga = int(0.03 * TAXA)
    ini = max(0, acima[0] - folga)
    fim = min(len(audio), acima[-1] + folga)
    return audio[ini:fim]


def encaixar(audio: np.ndarray, rng: np.random.Generator,
             total: int = AMOSTRAS_CLIPE) -> np.ndarray:
    """Posiciona o clipe em lugar aleatorio de uma janela de `total` amostras.

    Se o clipe nao cabe, mantem o FIM dele: o modelo dispara quando a palavra
    termina, entao o final e a parte que importa.
    """
    audio = audio[-total:] if len(audio) >= total else audio
    saida = np.zeros(total, dtype=np.int16)
    if len(audio) >= total:
        return audio.astype(np.int16)
    inicio = int(rng.integers(0, total - len(audio) + 1))
    saida[inicio:inicio + len(audio)] = audio
    return saida


def rms(audio: np.ndarray) -> float:
    if audio.size == 0:
        return 0.0
    return float(np.sqrt(np.mean(audio.astype(np.float64) ** 2)))
