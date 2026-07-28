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
        self.abortos = 0
        self.partidas = 0
        self.stopped = False
        self._trava = threading.Lock()

    def start(self) -> None:
        self.partidas += 1
        self.stopped = False

    def write(self, bloco) -> None:
        if self.fechado:
            self.escrita_apos_fechar = True
        time.sleep(self.atraso)
        with self._trava:
            self.escritos.append(len(bloco))

    def abort(self) -> None:
        """Como o PortAudio: descarta o buffer e deixa o stream parado."""
        self.abortos += 1
        self.stopped = True

    def stop(self) -> None:
        self.stopped = True

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


def _esperar(cond, prazo: float = 2.0) -> bool:
    fim = time.monotonic() + prazo
    while time.monotonic() < fim:
        if cond():
            return True
        time.sleep(0.005)
    return False


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


# --- barge-in: parar a fala de verdade (fase 6) --------------------------------

def test_blocos_longos_sao_escritos_em_fatias(saida):
    """`stream.write()` bloqueia ate consumir. Sem fatiar, um bloco de 2 s do
    Piper significava 2 s entre pedir silencio e obte-lo."""
    s, stream = saida
    fatia = 22_050 * 50 // 1000  # 50 ms a 22050 Hz
    s.enfileirar(_bloco(fatia * 4))
    s.aguardar()
    assert len(stream.escritos) == 4
    assert set(stream.escritos) == {fatia}


def test_interromper_nao_chama_o_stream(saida):
    """A invariante que custou um core dump.

    `abort()` chamado de outra thread enquanto o worker esta dentro de `write()`
    travava o PortAudio: a fala seguinte levava 21 s para tocar 2 s de audio e o
    processo despejava nucleo. Quem interrompe so marca; o worker mexe no stream.
    """
    s, stream = saida
    s.aguardar()
    antes = (stream.abortos, stream.partidas)
    s.interromper()
    assert (stream.abortos, stream.partidas) == antes


def test_worker_aborta_o_buffer_ao_ser_interrompido(saida):
    """abort() e nao stop(): stop() drenaria o buffer, tocando o que queremos cortar."""
    s, stream = saida
    fatia = 22_050 * 50 // 1000
    s.enfileirar(_bloco(fatia * 40))
    time.sleep(0.05)  # garante que o worker esta escrevendo
    s.interromper()
    s.aguardar()
    assert _esperar(lambda: stream.abortos == 1)


def test_interromper_para_no_meio_de_um_bloco_longo(saida):
    """O ganho real do fatiamento: a escrita para em ~1 fatia, nao no fim do bloco."""
    s, stream = saida
    fatia = 22_050 * 50 // 1000
    s.enfileirar(_bloco(fatia * 40))  # 2 s de audio = 40 fatias
    time.sleep(0.05)                  # deixa escrever uma ou duas
    s.interromper()
    s.aguardar()
    assert len(stream.escritos) < 40, "escreveu o bloco inteiro apesar do interromper()"


def test_fala_seguinte_ao_barge_in_sai(saida):
    """O caso que saiu mudo/travado na primeira versao: depois de abortar, o
    stream fica parado e alguem precisa reabri-lo -- o worker, sob demanda."""
    s, stream = saida
    fatia = 22_050 * 50 // 1000
    s.enfileirar(_bloco(fatia * 40))
    time.sleep(0.05)
    s.interromper()
    s.aguardar()
    # `interromper()` libera o `aguardar()` na hora, mas o worker pode estar no
    # meio de uma fatia -- cabe uma escrita a mais, e e assim mesmo (o limite e
    # uma fatia). Espera ela cair antes de contar.
    time.sleep(0.1)
    cortados = len(stream.escritos)

    s.retomar()
    s.enfileirar(_bloco(fatia * 2))
    s.aguardar()
    assert len(stream.escritos) == cortados + 2
    assert not stream.stopped
    assert stream.partidas >= 2  # a inicial e a reabertura


def test_retomar_sem_interrupcao_nao_reinicia(saida):
    s, stream = saida
    s.retomar()
    s.enfileirar(_bloco())
    s.aguardar()
    assert stream.partidas == 1


def test_bloco_de_antes_da_interrupcao_nao_ressuscita(saida):
    """A corrida que motivou o contador de geracao.

    Com uma flag simples, o worker podia nao ve-la antes do `retomar()` e voltava
    a tocar o bloco descartado -- a fala interrompida ressuscitava.
    """
    s, stream = saida
    fatia = 22_050 * 50 // 1000
    s.enfileirar(_bloco(fatia * 40))
    s.interromper()
    s.retomar()          # imediatamente, como o loop principal faz no turno seguinte
    time.sleep(0.2)
    s.aguardar()
    assert len(stream.escritos) < 40, "tocou o bloco que havia sido interrompido"


def test_erro_de_escrita_durante_interrupcao_nao_polui_o_log(saida, capsys, monkeypatch):
    """O abort() pode estourar o write() em andamento; nao e falha a relatar."""
    s, stream = saida

    def explode(_bloco):
        raise RuntimeError("PortAudio: stream aborted")

    s.enfileirar(_bloco())
    time.sleep(0.01)
    monkeypatch.setattr(stream, "write", explode)
    s.interromper()
    s.enfileirar(_bloco())  # ignorado: esta parado
    s.aguardar()
    assert "falha ao tocar" not in capsys.readouterr().out
