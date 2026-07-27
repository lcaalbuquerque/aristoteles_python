"""Deteccao de fala (VAD) e endpointing: saber quando o usuario parou de falar.

Duas camadas, porque uma nao basta:

1. **Gate de energia auto-calibrado.** O webrtcvad classifica ruido estacionario
   de banda larga (ventoinha, ar-condicionado, ganho alto de mic USB) como fala --
   medido nesta maquina: 98/100 blocos de ambiente viravam "fala" mesmo na
   agressividade 3. Sem o gate, o endpointing nunca ve silencio e grava ate o
   timeout. O piso de ruido e medido na inicializacao, entao adapta a cada sala.
2. **webrtcvad.** Bom para distinguir fala de outros sons *acima* do piso.

Um bloco so conta como fala se passar nas duas.
"""

from __future__ import annotations

import numpy as np
import webrtcvad

from .audio.entrada import EntradaAudio
from .config import AudioCfg, VadCfg


def rms(bloco: np.ndarray) -> float:
    """Valor eficaz do bloco int16, normalizado para [0, 1]."""
    x = bloco.astype(np.float32) / 32768.0
    return float(np.sqrt(np.mean(x * x)))


class DetectorFala:
    def __init__(self, cfg: VadCfg, audio: AudioCfg) -> None:
        self._vad = webrtcvad.Vad(cfg.agressividade)
        self.cfg = cfg
        self.audio = audio
        self.piso_ruido: float = 0.0
        self.limiar: float = cfg.piso_minimo

    def calibrar(self, entrada: EntradaAudio) -> float:
        """Mede o piso de ruido do ambiente e define o limiar do gate.

        Chame na inicializacao, com o usuario em silencio. Usa o percentil 90
        para nao se deixar levar por um estalo isolado.
        """
        n = max(4, int(self.cfg.calibracao_ms / self.audio.bloco_ms))
        entrada.limpar()
        niveis = []
        for _ in range(n):
            bloco = entrada.ler(timeout=1.0)
            if bloco is None:
                break
            niveis.append(rms(bloco))

        if not niveis:
            self.piso_ruido = 0.0
            self.limiar = self.cfg.piso_minimo
            return self.limiar

        self.piso_ruido = float(np.percentile(niveis, 90))
        self.limiar = max(self.piso_ruido * self.cfg.fator_acima_do_piso,
                          self.cfg.piso_minimo)
        return self.limiar

    def eh_fala(self, bloco: np.ndarray) -> bool:
        if rms(bloco) < self.limiar:
            return False
        return self._vad.is_speech(bloco.tobytes(), self.audio.taxa_amostragem)


def gravar_ate_silencio(
    entrada: EntradaAudio,
    detector: DetectorFala,
    cfg: VadCfg,
    usar_pre_roll: bool = True,
) -> np.ndarray | None:
    """Grava de agora ate detectar `silencio_final_ms` de silencio.

    Retorna float32 mono em [-1, 1], ou None se nao houve fala suficiente.
    """
    bloco_ms = entrada.cfg.bloco_ms
    max_silencio = max(1, cfg.silencio_final_ms // bloco_ms)
    min_fala = max(1, cfg.fala_minima_ms // bloco_ms)
    max_blocos = int(cfg.duracao_maxima_s * 1000 / bloco_ms)
    # Quanto esperar pela fala *comecar* antes de desistir.
    max_espera = max(1, int(cfg.espera_inicial_s * 1000 / bloco_ms))

    blocos: list[np.ndarray] = list(entrada.pre_roll()) if usar_pre_roll else []
    silencio_seguido = 0
    blocos_de_fala = 0
    houve_fala = False

    for i in range(max_blocos):
        bloco = entrada.ler(timeout=1.0)
        if bloco is None:
            break
        blocos.append(bloco)

        if detector.eh_fala(bloco):
            houve_fala = True
            blocos_de_fala += 1
            silencio_seguido = 0
        else:
            silencio_seguido += 1
            if houve_fala and silencio_seguido >= max_silencio:
                break
            # Nada foi dito: nao segura o usuario por duracao_maxima_s inteira.
            if not houve_fala and i >= max_espera:
                return None

    if blocos_de_fala < min_fala:
        return None

    return np.concatenate(blocos).astype(np.float32) / 32768.0
