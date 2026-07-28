"""Reproducao com fila e thread propria.

Permite sintetizar a frase N+1 enquanto a frase N ainda esta tocando, e sustenta o
barge-in (interromper a fala no meio).

Tres decisoes que parecem detalhe e nao sao:

1. **Escrever em fatias.** `stream.write()` bloqueia ate o bloco ser consumido, e
   o Piper entrega frases inteiras -- um bloco de 2 s significava 2 s entre pedir
   silencio e obte-lo. Escrevemos em fatias de 50 ms.

2. **Toda chamada ao stream mora na thread do worker.** Chamar `abort()` de outra
   thread enquanto o worker esta dentro de `write()` nao e seguro: medido, a fala
   seguinte levava 21 s para tocar 2 s de audio, a thread travava e o processo
   despejava nucleo no PortAudio. Quem pede a interrupcao so marca; o worker
   aborta e reabre.

3. **Contador de geracao, nao so uma flag.** Se `interromper()` apenas levantasse
   uma flag e `retomar()` a baixasse, havia corrida: o worker podia nao ver a flag
   antes do `retomar()`, e voltava a tocar o bloco que devia ter sido descartado --
   ou seja, a fala interrompida ressuscitava. A geracao e monotonica: um bloco
   enfileirado antes da interrupcao nunca toca depois dela.
"""

from __future__ import annotations

import queue
import threading

import numpy as np
import sounddevice as sd

from ..config import AudioCfg

_FIM = object()
_FATIA_MS = 50  # granularidade da checagem de interrupcao durante a escrita


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
        self._geracao = 0
        self._abortar = threading.Event()
        self._stream = sd.OutputStream(
            samplerate=taxa_amostragem,
            channels=1,
            dtype="int16",
            device=cfg.dispositivo_saida,
        )
        self._stream.start()
        self._worker = threading.Thread(target=self._rodar, daemon=True)
        self._worker.start()

    # --- thread do worker: a unica que fala com o stream ------------------------

    def _rodar(self) -> None:
        fatia = max(1, self.taxa * _FATIA_MS // 1000)
        while True:
            item = self._fila.get()
            if item is _FIM:  # nao entra na contagem de pendentes
                return
            geracao, bloco = item
            try:
                self._tocar(bloco, geracao, fatia)
            except Exception as e:  # dispositivo sumiu, etc.
                print(f"[saida] falha ao tocar: {e}")
            finally:
                self._concluir_um()

    def _tocar(self, bloco: np.ndarray, geracao: int, fatia: int) -> None:
        if geracao != self._geracao:
            return  # enfileirado antes de uma interrupcao; nao toca mais
        self._garantir_aberto()
        for i in range(0, len(bloco), fatia):
            if geracao != self._geracao:
                self._descartar_buffer()
                return
            self._stream.write(bloco[i:i + fatia])

    def _garantir_aberto(self) -> None:
        """Reabre o stream se uma interrupcao anterior o abortou."""
        if self._stream.stopped:
            self._stream.start()

    def _descartar_buffer(self) -> None:
        """abort() e nao stop(): stop() drena o buffer do PortAudio, ou seja,
        terminaria de tocar exatamente o que queremos cortar."""
        try:
            self._stream.abort()
        except Exception as e:
            print(f"[saida] falha ao abortar o stream: {e}")

    def _concluir_um(self) -> None:
        with self._trava:
            self._pendentes -= 1
            if self._pendentes <= 0:  # interromper() pode ter zerado por baixo
                self._pendentes = 0
                self._ocioso.set()

    # --- API de quem produz audio ---------------------------------------------

    def enfileirar(self, audio: np.ndarray) -> None:
        bloco = np.ascontiguousarray(audio, dtype=np.int16)
        with self._trava:
            if self._parar.is_set():
                return
            self._pendentes += 1
            self._ocioso.clear()
            self._fila.put((self._geracao, bloco))

    def aguardar(self) -> None:
        """Bloqueia ate a fila esvaziar e o ultimo bloco terminar."""
        self._ocioso.wait()

    def interromper(self) -> None:
        """Cala a fala agora. Seguro de chamar de qualquer thread.

        Nao toca no stream -- so invalida a geracao e esvazia a fila. O worker ve
        a geracao mudada e aborta o buffer do dispositivo.
        """
        self._parar.set()
        with self._trava:
            self._geracao += 1
            while not self._fila.empty():
                try:
                    self._fila.get_nowait()
                except queue.Empty:
                    break
            self._pendentes = 0
            self._ocioso.set()

    def retomar(self) -> None:
        """Libera a reproducao. O stream e reaberto pelo worker, sob demanda."""
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
