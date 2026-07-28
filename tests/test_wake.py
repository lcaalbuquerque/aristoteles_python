"""Os gatilhos de ativacao.

O `OpenWakeWord` real carrega um .onnx, entao aqui injetamos um modelo dublê --
o que se testa e a maquina de estados em volta dele: cooldown, reinicio e a
interface que o barge-in consome.
"""

from __future__ import annotations

import numpy as np
import pytest

from aristoteles.config import WakeCfg
from aristoteles.wake import PushToTalk, criar

BLOCO = np.zeros(480, dtype=np.int16)


class ModeloFalso:
    """Devolve pontuacoes de uma lista, na ordem."""

    def __init__(self, pontuacoes: list[float]) -> None:
        self._p = list(pontuacoes)
        self.chamadas = 0
        self.resets = 0

    def predict(self, _bloco):
        self.chamadas += 1
        p = self._p.pop(0) if self._p else 0.0
        return {"aristoteles": p}

    def reset(self):
        self.resets += 1


def _oww(pontuacoes: list[float], cfg: WakeCfg | None = None):
    """Monta um OpenWakeWord sem carregar modelo de verdade."""
    from aristoteles.wake import OpenWakeWord

    g = OpenWakeWord.__new__(OpenWakeWord)  # pula o __init__, que baixa modelos
    g._modelo = ModeloFalso(pontuacoes)
    g.cfg = cfg or WakeCfg(limiar=0.5, cooldown_s=1.5)
    g._mudo_ate = 0.0
    return g


# --- push-to-talk --------------------------------------------------------------

def test_push_to_talk_nao_suporta_barge_in():
    """Modo que ignora o microfone no ocioso nao tem o que vigiar durante a fala."""
    p = PushToTalk()
    assert p.suporta_barge_in is False
    assert p.alimentar(BLOCO) is False
    p.reiniciar()  # nao pode explodir


def test_push_to_talk_encerra_no_eof(monkeypatch):
    monkeypatch.setattr("builtins.input", lambda _p="": (_ for _ in ()).throw(EOFError))
    assert PushToTalk().aguardar(None) is False


@pytest.mark.parametrize("dito", ["sair", "q", "quit", "exit", "SAIR", " Sair "])
def test_push_to_talk_palavras_de_saida(monkeypatch, dito):
    monkeypatch.setattr("builtins.input", lambda _p="": dito)
    assert PushToTalk().aguardar(None) is False


# --- openWakeWord --------------------------------------------------------------

def test_dispara_acima_do_limiar():
    g = _oww([0.1, 0.9])
    assert g.alimentar(BLOCO) is False
    assert g.alimentar(BLOCO) is True


def test_nao_dispara_no_limiar_por_baixo():
    assert _oww([0.49], WakeCfg(limiar=0.5)).alimentar(BLOCO) is False
    assert _oww([0.50], WakeCfg(limiar=0.5)).alimentar(BLOCO) is True


def test_cooldown_evita_disparo_repetido():
    """Uma pronuncia atravessa varias janelas e passa do limiar em mais de uma.

    Sem cooldown, interromper a fala com a palavra disparava de novo no bloco
    seguinte e o assistente entrava e saia da gravacao. O `wake.cooldown_s` existia
    no config sem nenhum efeito ate a fase 6.
    """
    g = _oww([0.9, 0.9, 0.9], WakeCfg(limiar=0.5, cooldown_s=60.0))
    assert g.alimentar(BLOCO) is True
    assert g.alimentar(BLOCO) is False
    assert g.alimentar(BLOCO) is False


def test_cooldown_ainda_alimenta_o_modelo():
    """Pular o predict() durante o cooldown deixaria o buffer de features com um
    furo, e a deteccao seguinte veria audio descontinuo."""
    g = _oww([0.9, 0.9, 0.9], WakeCfg(limiar=0.5, cooldown_s=60.0))
    for _ in range(3):
        g.alimentar(BLOCO)
    assert g._modelo.chamadas == 3


def test_cooldown_expira(monkeypatch):
    agora = [1000.0]
    monkeypatch.setattr("aristoteles.wake.time.monotonic", lambda: agora[0])
    g = _oww([0.9, 0.9], WakeCfg(limiar=0.5, cooldown_s=1.5))
    assert g.alimentar(BLOCO) is True
    agora[0] += 1.6
    assert g.alimentar(BLOCO) is True


def test_reiniciar_zera_modelo_e_cooldown():
    g = _oww([0.9, 0.9], WakeCfg(limiar=0.5, cooldown_s=60.0))
    g.alimentar(BLOCO)
    g.reiniciar()
    assert g._modelo.resets == 1
    assert g._mudo_ate == 0.0
    assert g.alimentar(BLOCO) is True  # cooldown nao sobreviveu ao reinicio


def test_suporta_barge_in_e_pre_roll():
    g = _oww([])
    assert g.suporta_barge_in is True
    assert g.usa_pre_roll is True  # sem pre-roll, "Aristoteles, que horas" perde o fim


# --- fabrica -------------------------------------------------------------------

def test_criar_modo_desconhecido():
    with pytest.raises(ValueError, match="wake.modo desconhecido"):
        criar(WakeCfg(modo="telepatia"), None)


def test_criar_push_to_talk():
    assert isinstance(criar(WakeCfg(modo="push_to_talk"), None), PushToTalk)


def test_criar_openwakeword_sem_modelo_avisa(tmp_path):
    cfg = WakeCfg(modo="openwakeword", modelo=tmp_path / "nao_existe.onnx")
    with pytest.raises(FileNotFoundError, match="wake.modo: push_to_talk"):
        criar(cfg, tmp_path)
