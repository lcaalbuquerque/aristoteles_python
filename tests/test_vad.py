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


def test_limiar_tem_teto_e_nao_tranca_a_fala():
    """A regressao que matou o serviço por 6 minutos.

    Sob systemd, ruido durante os 600 ms de calibracao deu piso=0,1167 e limiar
    0,35. Fala normal fica entre 0,05 e 0,3 de RMS, entao NADA passou o gate:
    cada disparo da wake word terminava em "ninguem falou em 6s".
    """
    cfg = VadCfg(fator_acima_do_piso=3.0, piso_minimo=0.001, calibracao_ms=300,
                 limiar_maximo=0.12, descartar_aquecimento=0)
    det = DetectorFala(cfg, AudioCfg())
    # ambiente barulhento como o medido sob systemd
    det.calibrar(EntradaFalsa([bloco(0.1167, seed=i) for i in range(30)]))

    assert det.limiar <= 0.12, "limiar acima do teto tranca a fala"
    assert det.limiar_no_teto, "deveria sinalizar que bateu no teto"
    # fala normal (RMS ~0,15) precisa passar
    assert rms(bloco(0.15, seed=99)) > det.limiar


def test_limiar_no_teto_e_falso_em_ambiente_normal():
    cfg = VadCfg(fator_acima_do_piso=3.0, piso_minimo=0.001, calibracao_ms=300,
                 limiar_maximo=0.12, descartar_aquecimento=0)
    det = DetectorFala(cfg, AudioCfg())
    det.calibrar(EntradaFalsa([bloco(0.01, seed=i) for i in range(30)]))
    assert not det.limiar_no_teto
    assert det.limiar == pytest.approx(0.03, rel=0.3)


def test_calibracao_descarta_o_aquecimento_do_stream():
    """Medido: a abertura do stream entrega 0,0 no primeiro bloco e um pico de
    ~2x o ambiente no segundo."""
    cfg = VadCfg(fator_acima_do_piso=3.0, piso_minimo=0.0001, calibracao_ms=300,
                 descartar_aquecimento=3, limiar_maximo=1.0)
    det = DetectorFala(cfg, AudioCfg())
    lixo = [np.zeros(AMOSTRAS, np.int16), bloco(0.4, seed=1), bloco(0.4, seed=2)]
    ambiente = [bloco(0.01, seed=10 + i) for i in range(20)]
    det.calibrar(EntradaFalsa(lixo + ambiente))
    # se o pico de 0,4 entrasse na conta, o piso passaria de 0,01
    assert det.piso_ruido == pytest.approx(0.01, rel=0.3)


def test_estalo_isolado_nao_infla_o_piso():
    """Mediana, nao percentil 90: o p90 e MAIS sujeito a transiente que a mediana,
    e o fator_acima_do_piso ja fornece a margem."""
    cfg = VadCfg(fator_acima_do_piso=3.0, piso_minimo=0.0001, calibracao_ms=600,
                 descartar_aquecimento=0, limiar_maximo=1.0)
    det = DetectorFala(cfg, AudioCfg())
    # 18 blocos de ambiente + 2 estalos altos (10% da amostra: exatamente onde o
    # p90 se apoiava)
    blocos = ([bloco(0.01, seed=i) for i in range(18)]
              + [bloco(0.5, seed=100), bloco(0.5, seed=101)])
    det.calibrar(EntradaFalsa(blocos))
    assert det.piso_ruido == pytest.approx(0.01, rel=0.3)


def test_calibracao_sem_audio_nao_explode():
    cfg = VadCfg(piso_minimo=0.02, calibracao_ms=300)
    det = DetectorFala(cfg, AudioCfg())
    assert det.calibrar(EntradaFalsa([])) == 0.02
    assert det.piso_ruido == 0.0


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


def _com_pre_roll(blocos, pre_roll=()):
    e = EntradaFalsa(blocos)
    e._pre_roll = list(pre_roll)
    return e


def test_cauda_do_gatilho_nao_arma_o_endpointing():
    """A regressao relatada: a janela de escuta parecia instantanea.

    A cauda de "Aristoteles" ficava na fila e era contada como fala do usuario,
    ligando `houve_fala`. Dali em diante valia o silencio_final_ms de 700 ms em vez
    do espera_inicial_s, e quem pausava para formular a pergunta perdia a vez.
    """
    cfg = VadCfg(silencio_final_ms=700, espera_inicial_s=6.0, fala_minima_ms=300,
                 piso_minimo=0.001, absorver_max_ms=1300)
    det = DetectorFala(cfg, AudioCfg())
    det.limiar = 0.001

    # cauda do gatilho (10 blocos = 300 ms), pausa de 1,5 s (50 blocos), pergunta
    blocos = ([bloco(0.3, seed=i) for i in range(10)]
              + [np.zeros(AMOSTRAS, np.int16)] * 50
              + [bloco(0.3, seed=100 + i) for i in range(20)]
              + [np.zeros(AMOSTRAS, np.int16)] * 30)

    audio = gravar_ate_silencio(EntradaFalsa(blocos), det, cfg,
                                usar_pre_roll=False, absorver_gatilho=True)
    assert audio is not None, "desistiu durante a pausa depois do gatilho"


def test_sem_absorver_o_gatilho_a_pausa_encerra_cedo():
    """Documenta o comportamento antigo, para deixar claro o que a flag muda."""
    cfg = VadCfg(silencio_final_ms=700, espera_inicial_s=6.0, fala_minima_ms=300,
                 piso_minimo=0.001)
    det = DetectorFala(cfg, AudioCfg())
    det.limiar = 0.001
    blocos = ([bloco(0.3, seed=i) for i in range(10)]
              + [np.zeros(AMOSTRAS, np.int16)] * 50
              + [bloco(0.3, seed=100 + i) for i in range(20)]
              + [np.zeros(AMOSTRAS, np.int16)] * 30)

    entrada = EntradaFalsa(blocos)
    gravar_ate_silencio(entrada, det, cfg, usar_pre_roll=False,
                        absorver_gatilho=False)
    # Parou no silencio depois da cauda, sem chegar na pergunta.
    assert len(entrada._blocos) > 40


def test_tirada_unica_nao_e_engolida_pela_absorcao():
    """"Aristoteles, que horas sao?" sem pausa: o teto de absorcao protege."""
    cfg = VadCfg(silencio_final_ms=700, espera_inicial_s=6.0, fala_minima_ms=300,
                 piso_minimo=0.001, absorver_max_ms=300)  # teto baixo: 10 blocos
    det = DetectorFala(cfg, AudioCfg())
    det.limiar = 0.001
    # 40 blocos de fala contínua (1,2 s), depois silencio
    blocos = ([bloco(0.3, seed=i) for i in range(40)]
              + [np.zeros(AMOSTRAS, np.int16)] * 30)

    audio = gravar_ate_silencio(EntradaFalsa(blocos), det, cfg,
                                usar_pre_roll=False, absorver_gatilho=True)
    assert audio is not None, "absorveu a pergunta inteira como se fosse o gatilho"


def test_estalo_curto_nao_arma_o_endpointing():
    """A segunda metade da regressao relatada.

    Absorver o gatilho resolveu a cauda da palavra, mas qualquer ruido curto
    *depois* dela ainda armava o endpointing: medido, um estalo de 90 ms (tosse,
    cadeira, teclado) derrubava a janela de 6 s para 1,4 s. Por isso `houve_fala`
    exige `fala_minima_ms` acumulados, nao um bloco isolado.
    """
    cfg = VadCfg(silencio_final_ms=700, espera_inicial_s=6.0, fala_minima_ms=300,
                 piso_minimo=0.001, absorver_max_ms=1300)
    det = DetectorFala(cfg, AudioCfg())
    det.limiar = 0.001

    # gatilho, pausa, estalo de 3 blocos (90 ms), e silencio de sobra
    blocos = ([bloco(0.3, seed=i) for i in range(30)]
              + [np.zeros(AMOSTRAS, np.int16)] * 30
              + [bloco(0.3, seed=99 + i) for i in range(3)]
              + [np.zeros(AMOSTRAS, np.int16)] * 400)
    entrada = EntradaFalsa(blocos)
    gravar_ate_silencio(entrada, det, cfg, usar_pre_roll=False,
                        absorver_gatilho=True)
    consumidos = len(blocos) - len(entrada._blocos)
    # 6 s de paciencia = 200 blocos; com o bug parava em ~45
    assert consumidos > 150, f"desistiu em {consumidos * 0.03:.2f}s por causa de um estalo"


def test_estalo_reinicia_a_paciencia():
    """Fez um som mas nao falou: ganha os 6 s de novo, nao perde a vez."""
    cfg = VadCfg(espera_inicial_s=0.6, fala_minima_ms=300, piso_minimo=0.001,
                 absorver_max_ms=30)  # absorve 1 bloco so
    det = DetectorFala(cfg, AudioCfg())
    det.limiar = 0.001
    # 15 blocos de silencio (0,45 s), estalo, e a pergunta comeca depois
    blocos = ([bloco(0.3, seed=0)]
              + [np.zeros(AMOSTRAS, np.int16)] * 15
              + [bloco(0.3, seed=50)]                      # reinicia a paciencia
              + [np.zeros(AMOSTRAS, np.int16)] * 15
              + [bloco(0.3, seed=100 + i) for i in range(20)]
              + [np.zeros(AMOSTRAS, np.int16)] * 30)
    audio = gravar_ate_silencio(EntradaFalsa(blocos), det, cfg,
                                usar_pre_roll=False, absorver_gatilho=True)
    assert audio is not None, "desistiu apesar de o estalo ter reiniciado a espera"


def test_fala_longa_arma_o_endpointing():
    """O contrapeso: fala de verdade precisa encerrar em silencio_final_ms."""
    cfg = VadCfg(silencio_final_ms=300, espera_inicial_s=6.0, fala_minima_ms=300,
                 piso_minimo=0.001, absorver_max_ms=30)
    det = DetectorFala(cfg, AudioCfg())
    det.limiar = 0.001
    blocos = ([bloco(0.3, seed=0)]
              + [np.zeros(AMOSTRAS, np.int16)] * 5
              + [bloco(0.3, seed=100 + i) for i in range(20)]  # 600 ms: arma
              + [np.zeros(AMOSTRAS, np.int16)] * 200)
    entrada = EntradaFalsa(blocos)
    audio = gravar_ate_silencio(entrada, det, cfg, usar_pre_roll=False,
                                absorver_gatilho=True)
    assert audio is not None
    # encerrou nos 300 ms de silencio, nao esperou os 200 blocos
    assert len(entrada._blocos) > 150


def test_absorcao_ainda_desiste_se_ninguem_fala():
    """Absorver o gatilho nao pode virar espera infinita."""
    cfg = VadCfg(silencio_final_ms=700, espera_inicial_s=0.6, fala_minima_ms=300,
                 piso_minimo=0.001, absorver_max_ms=1300)
    det = DetectorFala(cfg, AudioCfg())
    det.limiar = 0.001
    blocos = ([bloco(0.3, seed=i) for i in range(10)]
              + [np.zeros(AMOSTRAS, np.int16)] * 300)
    entrada = EntradaFalsa(blocos)
    assert gravar_ate_silencio(entrada, det, cfg, usar_pre_roll=False,
                               absorver_gatilho=True) is None
    assert len(entrada._blocos) > 250  # desistiu rapido, nao gravou os 20 s


def test_audio_absorvido_vai_para_o_whisper():
    """O gatilho fica no audio: na tirada unica a pergunta comeca junto dele."""
    cfg = VadCfg(silencio_final_ms=700, espera_inicial_s=6.0, fala_minima_ms=300,
                 piso_minimo=0.001, absorver_max_ms=1300)
    det = DetectorFala(cfg, AudioCfg())
    det.limiar = 0.001
    blocos = ([bloco(0.3, seed=i) for i in range(10)]
              + [np.zeros(AMOSTRAS, np.int16)] * 10
              + [bloco(0.3, seed=100 + i) for i in range(20)]
              + [np.zeros(AMOSTRAS, np.int16)] * 30)
    audio = gravar_ate_silencio(EntradaFalsa(blocos), det, cfg,
                                usar_pre_roll=False, absorver_gatilho=True)
    assert audio is not None
    # 10 absorvidos + 10 silencio + 20 fala + 24 de silencio final, tudo presente
    assert len(audio) > 60 * AMOSTRAS


def test_fala_curta_demais_e_descartada():
    cfg = VadCfg(fala_minima_ms=300, piso_minimo=0.001, espera_inicial_s=5.0)
    det = DetectorFala(cfg, AudioCfg())
    det.limiar = 0.001
    # 2 blocos (60 ms) de fala, abaixo do minimo de 300 ms
    blocos = [bloco(0.3, seed=1), bloco(0.3, seed=2)] + [np.zeros(AMOSTRAS, np.int16)] * 60
    assert gravar_ate_silencio(EntradaFalsa(blocos), det, cfg, usar_pre_roll=False) is None
