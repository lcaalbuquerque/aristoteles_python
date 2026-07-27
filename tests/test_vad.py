"""Testes do gate de energia.

Regressao que motivou o gate: com webrtcvad puro, ruido ambiente estacionario
(RMS ~0.03) era classificado como fala em 98% dos blocos, e o endpointing nunca
via silencio -- gravava os 20 s de `duracao_maxima_s` toda vez.
"""

import numpy as np
import pytest

from aristoteles.config import AudioCfg, VadCfg
from aristoteles.vad import DetectorFala, gravar_ate_silencio, rms

TAXA = 16_000
AMOSTRAS = 480  # 30 ms


def bloco(amplitude: float, seed: int = 0) -> np.ndarray:
    """Ruido branco na amplitude pedida, como int16."""
    r = np.random.default_rng(seed)
    x = r.normal(0, amplitude, AMOSTRAS)
    return np.clip(x * 32768, -32768, 32767).astype(np.int16)


def test_rms_de_silencio_e_zero():
    assert rms(np.zeros(AMOSTRAS, dtype=np.int16)) == 0.0


def test_rms_acompanha_amplitude():
    assert rms(bloco(0.1)) > rms(bloco(0.01))


class EntradaFalsa:
    """Dublê de EntradaAudio: entrega blocos de uma lista."""

    def __init__(self, blocos, bloco_ms=30):
        self._blocos = list(blocos)
        self.cfg = AudioCfg(bloco_ms=bloco_ms)
        self._pre_roll = []

    def ler(self, timeout=1.0):
        return self._blocos.pop(0) if self._blocos else None

    def limpar(self):
        pass

    def pre_roll(self):
        return list(self._pre_roll)


def test_calibracao_define_limiar_acima_do_piso():
    cfg = VadCfg(fator_acima_do_piso=3.0, piso_minimo=0.001, calibracao_ms=300)
    det = DetectorFala(cfg, AudioCfg())
    ambiente = [bloco(0.03, seed=i) for i in range(20)]
    limiar = det.calibrar(EntradaFalsa(ambiente))
    assert det.piso_ruido == pytest.approx(0.03, rel=0.3)
    assert limiar == pytest.approx(det.piso_ruido * 3.0, rel=0.01)


def test_gate_rejeita_ruido_ambiente():
    """O caso que quebrava: ruido de fundo nao pode contar como fala."""
    cfg = VadCfg(fator_acima_do_piso=3.0, piso_minimo=0.001, calibracao_ms=300)
    det = DetectorFala(cfg, AudioCfg())
    det.calibrar(EntradaFalsa([bloco(0.03, seed=i) for i in range(20)]))
    novos = [bloco(0.03, seed=100 + i) for i in range(30)]
    assert sum(det.eh_fala(b) for b in novos) == 0


def test_piso_minimo_protege_ambiente_silencioso():
    """Em sala silenciosa o piso ~0 nao deve zerar o limiar."""
    cfg = VadCfg(piso_minimo=0.02, fator_acima_do_piso=3.0, calibracao_ms=300)
    det = DetectorFala(cfg, AudioCfg())
    limiar = det.calibrar(EntradaFalsa([np.zeros(AMOSTRAS, dtype=np.int16)] * 20))
    assert limiar == 0.02
    assert not det.eh_fala(bloco(0.005))


def test_desiste_se_ninguem_fala():
    """Nao pode segurar o usuario por duracao_maxima_s quando nao houve fala."""
    cfg = VadCfg(espera_inicial_s=0.6, duracao_maxima_s=20.0,
                 piso_minimo=0.5, calibracao_ms=300)
    det = DetectorFala(cfg, AudioCfg())
    silencio = [np.zeros(AMOSTRAS, dtype=np.int16)] * 700  # 21 s disponiveis
    entrada = EntradaFalsa(silencio)
    assert gravar_ate_silencio(entrada, det, cfg, usar_pre_roll=False) is None
    # consumiu ~20 blocos (0,6 s), nao os 667 de duracao_maxima_s
    assert len(entrada._blocos) > 600


def test_fala_curta_demais_e_descartada():
    cfg = VadCfg(fala_minima_ms=300, piso_minimo=0.001, espera_inicial_s=5.0)
    det = DetectorFala(cfg, AudioCfg())
    det.limiar = 0.001
    # 2 blocos (60 ms) de fala, abaixo do minimo de 300 ms
    blocos = [bloco(0.3, seed=1), bloco(0.3, seed=2)] + [np.zeros(AMOSTRAS, np.int16)] * 60
    assert gravar_ate_silencio(EntradaFalsa(blocos), det, cfg, usar_pre_roll=False) is None
