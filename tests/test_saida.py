"""A fila de reproducao, com um stream falso no lugar do PortAudio.

Os dois bugs que estes testes travam:
  1. `aguardar()` retornava com blocos ainda por tocar (corrida entre o clear()
     de `enfileirar` e o put() correspondente).
  2. `fechar()` fechava o stream por baixo de um write() em andamento.
"""

from __future__ import annotations

import threading
import time

import numpy as np
import pytest
import sounddevice as sd

from aristoteles.audio.saida import SaidaAudio
from aristoteles.config import AudioCfg


class StreamFalso:
    """Stream que demora para escrever e acusa uso depois do close()."""

    def __init__(self, atraso: float = 0.02, **_kwargs) -> None:
        self.atraso = atraso
        self.escritos: list[int] = []
        self.fechado = False
        self.escrita_apos_fechar = False
        self._trava = threading.Lock()

    def start(self) -> None:
        pass

    def write(self, bloco) -> None:
        if self.fechado:
            self.escrita_apos_fechar = True
        time.sleep(self.atraso)
        with self._trava:
            self.escritos.append(len(bloco))

    def stop(self) -> None:
        pass

    def close(self) -> None:
        self.fechado = True


@pytest.fixture
def saida(monkeypatch):
    falsos = []

    def fabrica(**kwargs):
        s = StreamFalso(**{k: v for k, v in kwargs.items() if k == "atraso"})
        falsos.append(s)
        return s

    monkeypatch.setattr(sd, "OutputStream", fabrica)
    s = SaidaAudio(AudioCfg(), 22_050)
    yield s, falsos[0]
    if not falsos[0].fechado:
        s.fechar()


def _bloco(n: int = 512) -> np.ndarray:
    return np.zeros(n, dtype=np.int16)


def test_aguardar_espera_todos_os_blocos(saida):
    s, stream = saida
    for _ in range(8):
        s.enfileirar(_bloco())
    s.aguardar()
    assert len(stream.escritos) == 8


def test_aguardar_nao_retorna_cedo_com_producao_intercalada(saida):
    """Enfileirar durante a reproducao, como o loop principal faz frase a frase."""
    s, stream = saida
    for _ in range(20):
        s.enfileirar(_bloco())
        time.sleep(0.005)  # deixa o worker alcancar a fila e ve-la vazia
    s.aguardar()
    assert len(stream.escritos) == 20


def test_aguardar_nao_retorna_cedo_com_a_janela_da_corrida_alargada(saida, monkeypatch):
    """Versao deterministica da corrida do `_ocioso`.

    A janela real dura microssegundos. Aqui ela e alargada tornando a conversao do
    bloco lenta: na versao com bug o `_ocioso.clear()` vinha *antes* dessa conversao
    e o put() depois, entao o worker terminava o bloco anterior no meio e sinalizava
    ocioso com um bloco ainda a caminho. Na versao corrigida a conversao acontece
    fora da trava e clear()+put() sao atomicos.
    """
    s, stream = saida
    original = np.ascontiguousarray

    def lento(a, **kwargs):
        time.sleep(0.05)
        return original(a, **kwargs)

    s.enfileirar(_bloco())  # worker comeca a tocar este
    monkeypatch.setattr(np, "ascontiguousarray", lento)
    s.enfileirar(_bloco())  # o worker termina o primeiro durante esta chamada
    monkeypatch.undo()

    s.aguardar()
    assert len(stream.escritos) == 2, "aguardar() voltou com um bloco pendente"


def test_fechar_durante_a_fala_nao_escreve_apos_o_close(saida):
    """Ctrl-C com muita fala pendente: nada pode tocar depois do close()."""
    s, stream = saida
    for _ in range(50):  # ~1 s de audio no stream falso
        s.enfileirar(_bloco())
    time.sleep(0.03)  # deixa comecar
    s.fechar()
    assert stream.fechado
    assert not stream.escrita_apos_fechar
    assert len(stream.escritos) < 50  # a fila foi descartada, nao drenada


def test_interromper_descarta_e_libera_aguardar(saida):
    s, stream = saida
    for _ in range(50):
        s.enfileirar(_bloco())
    s.interromper()
    s.aguardar()  # nao deve travar
    assert len(stream.escritos) < 50


def test_enfileirar_apos_interromper_e_ignorado(saida):
    s, _stream = saida
    s.interromper()
    s.enfileirar(_bloco())
    s.aguardar()


def test_retomar_volta_a_aceitar(saida):
    s, stream = saida
    s.interromper()
    s.retomar()
    s.enfileirar(_bloco())
    s.aguardar()
    assert len(stream.escritos) == 1
