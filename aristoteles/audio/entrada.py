"""Captura continua do microfone com buffer de pre-gravacao.

O buffer circular guarda os ultimos ~1,5 s de audio *antes* do gatilho. Sem ele,
quem fala "Aristoteles, que horas sao?" numa tirada so perde o "que horas".
"""

from __future__ import annotations

import queue
import sys
from collections import deque

import numpy as np
import sounddevice as sd

from ..config import AudioCfg


class EntradaAudio:
    """Stream de entrada. Use como context manager.

    Entrega blocos int16 mono de `cfg.bloco_ms` na fila `self.fila`.
    """

    def __init__(self, cfg: AudioCfg) -> None:
        self.cfg = cfg
        # Fila limitada: o stream captura o tempo todo, mas so `gravar_ate_silencio`
        # consome. Enquanto o programa espera no gatilho -- que pode ser a noite
        # inteira num servico systemd -- uma fila sem teto cresce 32 KB/s.
        self.fila: queue.Queue[np.ndarray] = queue.Queue(maxsize=cfg.blocos_da_fila)
        self.descartados = 0
        blocos_pre_roll = max(1, int(cfg.pre_roll_s * 1000 / cfg.bloco_ms))
        self._pre_roll: deque[np.ndarray] = deque(maxlen=blocos_pre_roll)
        self._stream: sd.InputStream | None = None

    def __enter__(self) -> "EntradaAudio":
        self._stream = sd.InputStream(
            samplerate=self.cfg.taxa_amostragem,
            blocksize=self.cfg.amostras_por_bloco,
            channels=1,
            dtype="int16",
            device=self.cfg.dispositivo_entrada,
            callback=self._callback,
        )
        self._stream.start()
        return self

    def __exit__(self, *_exc) -> None:
        if self._stream is not None:
            self._stream.stop()
            self._stream.close()
            self._stream = None

    def _callback(self, indata, _frames, _time, status) -> None:
        if status:
            print(f"[audio] {status}", file=sys.stderr)
        bloco = indata[:, 0].copy()
        self._pre_roll.append(bloco)
        # Fila cheia = ninguem esta gravando. Descarta o bloco mais antigo em vez
        # do atual, e nunca bloqueia: este callback roda na thread do PortAudio e
        # segurar ele ali produz estouro de buffer na captura.
        while True:
            try:
                self.fila.put_nowait(bloco)
                return
            except queue.Full:
                try:
                    self.fila.get_nowait()
                    self.descartados += 1
                except queue.Empty:
                    return  # outra thread esvaziou; desiste deste bloco

    def pre_roll(self) -> list[np.ndarray]:
        """Copia dos blocos anteriores ao gatilho."""
        return list(self._pre_roll)

    def limpar(self) -> None:
        """Descarta o que se acumulou na fila (ex.: audio da propria resposta falada)."""
        while True:
            try:
                self.fila.get_nowait()
            except queue.Empty:
                return

    def ler(self, timeout: float = 1.0) -> np.ndarray | None:
        try:
            return self.fila.get(timeout=timeout)
        except queue.Empty:
            return None


def listar_dispositivos() -> str:
    return str(sd.query_devices())
