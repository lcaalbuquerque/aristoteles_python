"""Barge-in: interromper a fala dizendo a palavra de ativacao.

Nada aqui toca em audio real. O `SaidaAudio` e o `EntradaAudio` sao dubles, e o
gatilho e um objeto que dispara quando mandamos.
"""

from __future__ import annotations

import threading
import time

import numpy as np
import pytest

from aristoteles.barge_in import VigiaBargeIn

BLOCO = np.zeros(480, dtype=np.int16)


class GatilhoFalso:
    """Dispara no N-esimo bloco alimentado."""

    usa_pre_roll = True

    def __init__(self, dispara_em: int | None = None, suporta: bool = True,
                 erro: Exception | None = None) -> None:
        self.suporta_barge_in = suporta
        self._dispara_em = dispara_em
        self._erro = erro
        self.alimentados = 0
        self.reiniciado = 0

    def aguardar(self, entrada):
        return True

    def alimentar(self, bloco):
        self.alimentados += 1
        if self._erro is not None:
            raise self._erro
        return self._dispara_em is not None and self.alimentados >= self._dispara_em

    def reiniciar(self):
        self.reiniciado += 1


class EntradaFalsa:
    """Entrega blocos sem fim; conta as limpezas."""

    def __init__(self) -> None:
        self.limpezas = 0

    def ler(self, timeout=1.0):
        time.sleep(0.001)  # nao queima CPU no teste
        return BLOCO

    def limpar(self):
        self.limpezas += 1


class SaidaFalsa:
    def __init__(self) -> None:
        self.interrompida = threading.Event()

    def interromper(self):
        self.interrompida.set()


def _esperar(cond, prazo=3.0) -> bool:
    fim = time.monotonic() + prazo
    while time.monotonic() < fim:
        if cond():
            return True
        time.sleep(0.005)
    return False


def test_palavra_durante_a_fala_interrompe():
    gatilho, entrada, saida = GatilhoFalso(dispara_em=3), EntradaFalsa(), SaidaFalsa()
    with VigiaBargeIn(gatilho, entrada, saida) as vigia:
        assert _esperar(lambda: vigia.interrompido)
    assert vigia.interrompido
    assert saida.interrompida.is_set()


def test_sem_a_palavra_nao_interrompe():
    gatilho, entrada, saida = GatilhoFalso(dispara_em=None), EntradaFalsa(), SaidaFalsa()
    with VigiaBargeIn(gatilho, entrada, saida) as vigia:
        assert _esperar(lambda: gatilho.alimentados > 5)
    assert not vigia.interrompido
    assert not saida.interrompida.is_set()


def test_reinicia_o_gatilho_e_limpa_a_entrada_ao_comecar():
    """Restos da propria pergunta na fila dispararia o gatilho na hora."""
    gatilho, entrada, saida = GatilhoFalso(), EntradaFalsa(), SaidaFalsa()
    with VigiaBargeIn(gatilho, entrada, saida):
        pass
    assert gatilho.reiniciado == 1
    assert entrada.limpezas == 1


def test_gatilho_sem_suporte_nao_vigia():
    """push_to_talk ignora o microfone; nao faz sentido vigiar."""
    gatilho = GatilhoFalso(dispara_em=1, suporta=False)
    entrada, saida = EntradaFalsa(), SaidaFalsa()
    with VigiaBargeIn(gatilho, entrada, saida) as vigia:
        time.sleep(0.05)
    assert gatilho.alimentados == 0
    assert not vigia.interrompido
    assert entrada.limpezas == 0


def test_desativado_explicitamente_nao_vigia():
    gatilho = GatilhoFalso(dispara_em=1)
    with VigiaBargeIn(gatilho, EntradaFalsa(), SaidaFalsa(), ativo=False) as vigia:
        time.sleep(0.05)
    assert gatilho.alimentados == 0
    assert not vigia.interrompido


def test_falha_do_detector_nao_derruba_a_fala(capsys):
    gatilho = GatilhoFalso(erro=RuntimeError("modelo explodiu"))
    entrada, saida = EntradaFalsa(), SaidaFalsa()
    with VigiaBargeIn(gatilho, entrada, saida) as vigia:
        assert _esperar(lambda: "modelo explodiu" in capsys.readouterr().out
                        or gatilho.alimentados > 0)
    assert not vigia.interrompido
    assert not saida.interrompida.is_set()


def test_a_thread_encerra_ao_sair_do_with():
    gatilho, entrada, saida = GatilhoFalso(), EntradaFalsa(), SaidaFalsa()
    vigia = VigiaBargeIn(gatilho, entrada, saida)
    with vigia:
        assert _esperar(lambda: gatilho.alimentados > 0)
        interna = vigia._thread
    assert interna is not None and not interna.is_alive()
