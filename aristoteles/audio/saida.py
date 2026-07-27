"""Reproducao com fila e thread propria.

Permite sintetizar a frase N+1 enquanto a frase N ainda esta tocando, e da o
gancho para barge-in (interromper a fala) na fase 6.
"""

from __future__ import annotations

import queue
import threading

import numpy as np
import sounddevice as sd

from ..config import AudioCfg

_FIM = object()


class SaidaAudio:
    def __init__(self, cfg: AudioCfg, taxa_amostragem: int) -> None:
        self.cfg = cfg
        self.taxa = taxa_amostragem
        self._fila: queue.Queue = queue.Queue()
        self._parar = threading.Event()
        self._ocioso = threading.Event()
        self._ocioso.set()
        # `_ocioso` nao pode ser derivado de `_fila.empty()`: entre o clear() de
        # quem enfileira e o put() correspondente a fila esta vazia com um bloco
        # a caminho, e o worker sinalizava ocioso ali -- fazendo aguardar()
        # retornar com audio ainda por tocar. O contador sob trava e a verdade.
        self._trava = threading.Lock()
        self._pendentes = 0
        self._stream = sd.OutputStream(
            samplerate=taxa_amostragem,
            channels=1,
            dtype="int16",
            device=cfg.dispositivo_saida,
        )
        self._stream.start()
        self._worker = threading.Thread(target=self._rodar, daemon=True)
        self._worker.start()

    def _rodar(self) -> None:
        while True:
            item = self._fila.get()
            if item is _FIM:  # nao entra na contagem de pendentes
                return
            try:
                if not self._parar.is_set():
                    self._stream.write(item)
            except Exception as e:  # dispositivo sumiu, etc.
                print(f"[saida] falha ao tocar: {e}")
            finally:
                self._concluir_um()

    def _concluir_um(self) -> None:
        with self._trava:
            self._pendentes -= 1
            if self._pendentes <= 0:  # interromper() pode ter zerado por baixo
                self._pendentes = 0
                self._ocioso.set()

    def enfileirar(self, audio: np.ndarray) -> None:
        bloco = np.ascontiguousarray(audio, dtype=np.int16)
        with self._trava:
            if self._parar.is_set():
                return
            self._pendentes += 1
            self._ocioso.clear()
            self._fila.put(bloco)

    def aguardar(self) -> None:
        """Bloqueia ate a fila esvaziar e o ultimo bloco terminar."""
        self._ocioso.wait()

    def interromper(self) -> None:
        """Descarta o que falta tocar (barge-in).

        O bloco que ja esta dentro de stream.write() ainda toca ate o fim -- ver
        fase 6 no README.
        """
        self._parar.set()
        with self._trava:
            while not self._fila.empty():
                try:
                    self._fila.get_nowait()
                except queue.Empty:
                    break
            self._pendentes = 0
            self._ocioso.set()

    def retomar(self) -> None:
        self._parar.clear()

    def fechar(self) -> None:
        # Esvaziar a fila antes do sentinela: sem isso o worker tocava tudo o que
        # restava (uma resposta longa passa dos 2 s do join antigo) e o stream era
        # fechado por baixo de um stream.write() em andamento -- Ctrl-C durante a
        # fala podia derrubar o processo no PortAudio.
        self.interromper()
        self._fila.put(_FIM)
        self._worker.join(timeout=5)
        if self._worker.is_alive():
            # Melhor vazar o stream do que fecha-lo sob escrita: o worker e daemon
            # e o processo esta saindo de qualquer forma.
            print("[saida] thread de reproducao travada; deixando o stream aberto")
            return
        self._stream.stop()
        self._stream.close()


def beep(frequencia: float = 880.0, duracao: float = 0.12, taxa: int = 16_000,
         volume: float = 0.25, dispositivo=None) -> None:
    """Sinal sonoro curto. Sem isso o usuario nunca sabe se foi ouvido."""
    t = np.linspace(0, duracao, int(taxa * duracao), endpoint=False)
    onda = np.sin(2 * np.pi * frequencia * t)
    # envelope para nao estalar
    rampa = int(0.01 * taxa)
    if rampa * 2 < len(onda):
        onda[:rampa] *= np.linspace(0, 1, rampa)
        onda[-rampa:] *= np.linspace(1, 0, rampa)
    sd.play((onda * volume * 32767).astype(np.int16), taxa, device=dispositivo)
    sd.wait()
