"""Testes da ferramentaria de treino da wake word (fase 5).

Cobrem o que ja quebrou ou pode quebrar em silencio:

* **PCM de 24 bits** -- as respostas impulsivas do MIT vem em 24 bits
  (WAVE_FORMAT_EXTENSIBLE) e o leitor so tratava 16, o que abortava a aumentacao.
* **vazamento treino/validacao** -- a divisao acontece por arquivo, antes da
  aumentacao. Se variantes do mesmo clipe caissem nos dois lados, o recall de
  validacao viria inflado e ninguem notaria.
* **contrato do .onnx** -- se a arquitetura sair do formato que o
  `openwakeword.model.Model` carrega, o treino "funciona" e o assistente nao usa.
"""

from __future__ import annotations

import wave

import numpy as np
import pytest

from treino import aumentar, features
from treino.comum import (AMOSTRAS_CLIPE, DIM, FRAMES, TAXA, _para_int16,
                          aparar_silencio, encaixar, escrever_wav, ler_wav,
                          reamostrar, rms)


def fala(n: int = 8000, amp: float = 0.3, seed: int = 0) -> np.ndarray:
    r = np.random.default_rng(seed)
    return np.clip(r.normal(0, amp, n) * 32768, -32768, 32767).astype(np.int16)


# --- comum: leitura de audio ---------------------------------------------------

def test_para_int16_de_16_bits_e_identidade():
    x = fala(100)
    assert np.array_equal(_para_int16(x.tobytes(), 2), x)


def _bytes_24(valores: np.ndarray) -> bytes:
    """Empacota amostras de 24 bits em little-endian, como num wav de verdade."""
    v = valores.astype(np.int32)
    return np.stack([v & 0xFF, (v >> 8) & 0xFF, (v >> 16) & 0xFF],
                    axis=1).astype(np.uint8).tobytes()


def test_para_int16_de_24_bits():
    """As RIRs do MIT sao 24 bits; sem isso a aumentacao nem comeca."""
    valores = np.array([0, 1 << 16, -(1 << 16), 8_388_607, -8_388_608], dtype=np.int32)
    obtido = _para_int16(_bytes_24(valores), 3)
    assert obtido.dtype == np.int16
    assert len(obtido) == 5
    assert obtido[0] == 0
    # 24 -> 16 bits descarta os 8 bits baixos: a amostra vira S >> 8
    assert obtido[1] == 256 and obtido[2] == -256
    assert obtido[3] == 32_767 and obtido[4] == -32_768  # extremos sem estourar


def test_para_int16_largura_nao_suportada():
    with pytest.raises(ValueError, match="largura"):
        _para_int16(b"\x00" * 10, 5)


def test_ler_wav_24_bits_de_verdade(tmp_path):
    """Escreve um wav de 24 bits na mao e le de volta."""
    caminho = tmp_path / "r.wav"
    valores = (np.arange(300, dtype=np.int32) - 150) * 4096
    with wave.open(str(caminho), "wb") as f:
        f.setnchannels(1)
        f.setsampwidth(3)
        f.setframerate(TAXA)
        f.writeframes(_bytes_24(valores))
    lido = ler_wav(caminho)
    assert lido.dtype == np.int16
    assert len(lido) == 300
    assert lido[0] < 0 < lido[-1]
    assert np.array_equal(lido, (valores >> 8).astype(np.int16))


def test_wav_ida_e_volta(tmp_path):
    x = fala(1600)
    escrever_wav(tmp_path / "a.wav", x)
    assert np.array_equal(ler_wav(tmp_path / "a.wav"), x)


def test_ler_wav_reamostra_para_16k(tmp_path):
    """As vozes medium do Piper saem a 22050 Hz."""
    x = fala(22_050)
    escrever_wav(tmp_path / "b.wav", x, taxa=22_050)
    lido = ler_wav(tmp_path / "b.wav")
    assert abs(len(lido) - TAXA) <= 1


def test_ler_wav_pega_um_canal(tmp_path):
    caminho = tmp_path / "estereo.wav"
    esq = np.full(100, 1000, dtype=np.int16)
    dir_ = np.full(100, -1000, dtype=np.int16)
    with wave.open(str(caminho), "wb") as f:
        f.setnchannels(2)
        f.setsampwidth(2)
        f.setframerate(TAXA)
        f.writeframes(np.stack([esq, dir_], axis=1).tobytes())
    assert np.all(ler_wav(caminho) == 1000)


def test_reamostrar_sem_mudanca_nao_toca_no_sinal():
    x = fala(500)
    assert np.array_equal(reamostrar(x, TAXA, TAXA), x)


# --- comum: preparo do clipe ---------------------------------------------------

def test_aparar_silencio_corta_as_bordas():
    meio = fala(4000, amp=0.5)
    x = np.concatenate([np.zeros(8000, np.int16), meio, np.zeros(8000, np.int16)])
    aparado = aparar_silencio(x)
    assert len(aparado) < len(x)
    assert len(aparado) >= len(meio)  # a folga de 30 ms nao pode comer o sinal


def test_aparar_silencio_de_silencio_puro_nao_explode():
    x = np.zeros(1000, dtype=np.int16)
    assert len(aparar_silencio(x)) == 1000


def test_aparar_silencio_de_vazio():
    assert aparar_silencio(np.zeros(0, dtype=np.int16)).size == 0


def test_encaixar_devolve_janela_exata():
    rng = np.random.default_rng(0)
    assert encaixar(fala(5000), rng).size == AMOSTRAS_CLIPE


def test_encaixar_preserva_o_fim_quando_nao_cabe():
    """O modelo dispara no fim da palavra, entao o fim e o que se preserva."""
    rng = np.random.default_rng(0)
    x = np.arange(AMOSTRAS_CLIPE + 5000, dtype=np.int16)
    obtido = encaixar(x, rng)
    assert np.array_equal(obtido, x[-AMOSTRAS_CLIPE:])


def test_encaixar_varia_a_posicao():
    """Posicao fixa ensinaria o modelo a esperar a palavra sempre no mesmo lugar."""
    rng = np.random.default_rng(0)
    curto = fala(3000, amp=0.5)
    inicios = set()
    for _ in range(30):
        nz = np.flatnonzero(encaixar(curto, rng))
        inicios.add(int(nz[0]) if nz.size else -1)
    assert len(inicios) > 5


# --- aumentacao ----------------------------------------------------------------

def test_ruido_colorido_tem_o_tamanho_pedido():
    rng = np.random.default_rng(0)
    assert aumentar.ruido_colorido(AMOSTRAS_CLIPE, 1.0, rng).size == AMOSTRAS_CLIPE


def test_ruido_rosa_tem_mais_energia_grave_que_o_branco():
    """Ventoinha e ar-condicionado sao ruido de banda larga com inclinacao 1/f."""
    rng = np.random.default_rng(0)
    def razao(exp):
        x = aumentar.ruido_colorido(16_384, exp, rng)
        p = np.abs(np.fft.rfft(x)) ** 2
        return p[1:100].sum() / p[1000:2000].sum()
    assert razao(2.0) > razao(1.0) > razao(0.0)


def test_aplicar_rir_preserva_o_rms():
    """A reverberacao muda o timbre, nao o nivel -- senao a SNR viraria loteria."""
    rng = np.random.default_rng(0)
    x = fala(AMOSTRAS_CLIPE)
    rir = aumentar.ruido_colorido(800, 1.0, rng)
    saida = aumentar.aplicar_rir(x, rir)
    assert len(saida) == len(x)
    assert rms(saida) == pytest.approx(rms(x), rel=0.01)


def test_aumentar_clipe_devolve_int16_do_tamanho_certo():
    rng = np.random.default_rng(0)
    for _ in range(20):
        c = aumentar.aumentar_clipe(fala(6000), [], [], rng)
        assert c.dtype == np.int16 and c.size == AMOSTRAS_CLIPE
        assert abs(c).max() <= 32_767


def test_aumentar_clipe_com_rir_nao_estoura():
    rng = np.random.default_rng(1)
    rirs = [aumentar.ruido_colorido(600, 1.0, rng) for _ in range(3)]
    for _ in range(20):
        c = aumentar.aumentar_clipe(fala(6000, amp=0.9), rirs, [], rng)
        assert abs(c).max() <= 32_767


def test_aumentar_clipes_gera_voltas_vezes_os_clipes(tmp_path):
    for i in range(4):
        escrever_wav(tmp_path / f"c{i}.wav", fala(5000, seed=i))
    caminhos = sorted(tmp_path.glob("*.wav"))
    saida = list(aumentar.aumentar_clipes(caminhos, 3, [], []))
    assert len(saida) == 12


def test_aumentar_clipes_sem_arquivos_reclama():
    with pytest.raises(FileNotFoundError):
        list(aumentar.aumentar_clipes([], 2, [], []))


def test_carregar_rirs_de_diretorio_ausente(tmp_path):
    assert aumentar.carregar_rirs(tmp_path / "nao_existe") == []
    assert aumentar.carregar_fundos(None) == []


# --- features ------------------------------------------------------------------

def test_janelas_de_features_recorta_com_passo_1(tmp_path):
    caminho = tmp_path / "f.npy"
    np.save(caminho, np.arange(40 * DIM, dtype=np.float32).reshape(40, DIM))
    j = features.janelas_de_features(caminho)
    assert j.shape == (40 - FRAMES, FRAMES, DIM)
    assert np.array_equal(j[0], np.arange(FRAMES * DIM).reshape(FRAMES, DIM))


def test_janelas_de_features_aceita_ja_janelado(tmp_path):
    caminho = tmp_path / "g.npy"
    np.save(caminho, np.zeros((5, FRAMES, DIM), dtype=np.float32))
    assert features.janelas_de_features(caminho).shape == (5, FRAMES, DIM)


def test_janelas_de_features_recusa_curto(tmp_path):
    caminho = tmp_path / "h.npy"
    np.save(caminho, np.zeros((FRAMES - 1, DIM), dtype=np.float32))
    with pytest.raises(ValueError):
        features.janelas_de_features(caminho)


def test_extrair_recusa_clipe_do_tamanho_errado(tmp_path):
    with pytest.raises(ValueError, match="amostras"):
        features.extrair([np.zeros(1000, dtype=np.int16)], tmp_path / "x.npy",
                         progresso=False)


# --- divisao treino/validacao --------------------------------------------------

def test_divisao_nao_vaza_entre_treino_e_validacao(tmp_path):
    from treino.__main__ import _dividir

    for i in range(40):
        escrever_wav(tmp_path / f"c{i:03d}.wav", fala(1000, seed=i))
    treino_, val = _dividir(tmp_path, semente=7)
    assert set(treino_).isdisjoint(val)
    assert len(treino_) + len(val) == 40
    assert 1 <= len(val) < 40


def test_divisao_e_estavel_para_a_mesma_semente(tmp_path):
    from treino.__main__ import _dividir

    for i in range(20):
        escrever_wav(tmp_path / f"c{i:02d}.wav", fala(500, seed=i))
    assert _dividir(tmp_path, semente=3) == _dividir(tmp_path, semente=3)


def test_divisao_de_diretorio_vazio(tmp_path):
    from treino.__main__ import _dividir

    assert _dividir(tmp_path) == ([], [])


# --- treino e contrato do .onnx ------------------------------------------------

def test_fonte_negativa_de_arquivo_continuo(tmp_path):
    """Layout do validation_set_features.npy: (N, 96) float32."""
    from treino.treinar import FonteNegativa

    caminho = tmp_path / "neg.npy"
    np.save(caminho, np.random.default_rng(0).normal(size=(500, DIM)).astype(np.float32))
    fonte = FonteNegativa(caminho)
    assert not fonte.janelado
    lote = fonte.amostrar(9, np.random.default_rng(0))
    assert lote.shape == (9, FRAMES, DIM) and lote.dtype == np.float32
    assert fonte.horas == pytest.approx(500 * 0.08 / 3600)


def test_fonte_negativa_de_arquivo_janelado(tmp_path):
    """Layout do ACAV100M: (N, 16, 96) float16 -- ja vem janelado."""
    from treino.treinar import FonteNegativa

    caminho = tmp_path / "acav.npy"
    np.save(caminho, np.random.default_rng(0).normal(
        size=(300, FRAMES, DIM)).astype(np.float16))
    fonte = FonteNegativa(caminho)
    assert fonte.janelado and fonte.n == 300
    lote = fonte.amostrar(11, np.random.default_rng(0))
    assert lote.shape == (11, FRAMES, DIM) and lote.dtype == np.float32
    # 300 janelas x 16 frames x 80 ms
    assert fonte.horas == pytest.approx(300 * FRAMES * 0.08 / 3600)


def test_fonte_negativa_recusa_formato_errado(tmp_path):
    from treino.treinar import FonteNegativa

    for forma in [(10, 5), (10, FRAMES, 5), (10, 3, DIM)]:
        caminho = tmp_path / f"ruim_{len(forma)}_{forma[-1]}.npy"
        np.save(caminho, np.zeros(forma, dtype=np.float32))
        with pytest.raises(ValueError):
            FonteNegativa(caminho)


def test_ponto_de_operacao_acha_o_menor_limiar_dentro_do_alvo():
    from treino.treinar import _ponto_de_operacao

    p_pos = np.linspace(0.6, 1.0, 100)
    p_fp = np.linspace(0.0, 0.4, 1000)   # nada acima de 0.4
    t, rec, fp_h = _ponto_de_operacao(p_pos, p_fp, horas=10.0, alvo_fp=0.2)
    assert fp_h <= 0.2
    assert rec == 1.0            # todos os positivos acima de 0.4
    assert 0.35 <= t <= 0.45     # nao sobe o limiar mais que o necessario


def test_ponto_de_operacao_prefere_quem_separa_a_quem_nao_dispara():
    """A regressao que motivou este criterio.

    Num treino real o criterio antigo -- recall no limiar fixo 0.5 -- escolheu um
    checkpoint com recall 0.602 e fp/h 0 por ser o unico "dentro do alvo",
    descartando outro com recall 0.947. O certo e comparar o recall no limiar que
    cada checkpoint precisa para cumprir o alvo: quem separa bem so precisa subir
    o limiar, quem nao separa ja perdeu a informacao.
    """
    from treino.treinar import _ponto_de_operacao

    # A separa perfeitamente, mas so acima de 0.6 -- no limiar fixo 0.5 TODOS os
    # negativos disparariam, e o criterio antigo descartaria A.
    pos_a, fp_a = np.full(100, 0.80), np.full(1000, 0.60)
    assert (fp_a >= 0.5).sum() == 1000  # o que enganava o criterio antigo

    # B mal separa: 60% dos positivos ficam abaixo dos negativos.
    pos_b = np.concatenate([np.full(60, 0.30), np.full(40, 0.99)])
    fp_b = np.full(1000, 0.50)

    a = _ponto_de_operacao(pos_a, fp_a, horas=10.0, alvo_fp=0.2)
    b = _ponto_de_operacao(pos_b, fp_b, horas=10.0, alvo_fp=0.2)

    assert a[1] == pytest.approx(1.0)   # A: recall cheio
    assert b[1] == pytest.approx(0.4)   # B: perde os 60% abaixo dos negativos
    assert a[1] > b[1]
    assert a[2] <= 0.2 and b[2] <= 0.2  # ambos cumprem o alvo de fp/h


def test_ponto_de_operacao_com_distribuicoes_sobrepostas():
    """Sem limiar viavel, devolve o mais alto em vez de mentir."""
    from treino.treinar import _ponto_de_operacao

    p = np.linspace(0.0, 1.0, 1000)
    t, rec, fp_h = _ponto_de_operacao(p[:100], p, horas=0.001, alvo_fp=0.2)
    assert t >= 0.99 and 0.0 <= rec <= 1.0


def test_rede_devolve_probabilidade_por_exemplo():
    torch = pytest.importorskip("torch")
    from treino.treinar import construir_rede

    rede = construir_rede(128, 1)
    saida = rede(torch.rand(5, FRAMES, DIM))
    assert saida.shape == (5, 1)
    assert bool(((saida >= 0) & (saida <= 1)).all())


def test_onnx_exportado_carrega_no_openwakeword(tmp_path):
    """O contrato que importa: o runtime carrega pelo openwakeword.model.Model."""
    pytest.importorskip("torch")
    oww = pytest.importorskip("openwakeword.model")
    from treino.treinar import construir_rede, exportar_onnx

    destino = tmp_path / "aristoteles.onnx"
    exportar_onnx(construir_rede(128, 1), destino)

    m = oww.Model(wakeword_models=[str(destino)], inference_framework="onnx")
    assert "aristoteles" in m.models
    # shape[1] e o que o openwakeword le para saber quantos frames pedir
    assert m.model_inputs["aristoteles"] == FRAMES
    pontos = m.predict(np.zeros(1280, dtype=np.int16))
    assert set(pontos) == {"aristoteles"}
    assert 0.0 <= pontos["aristoteles"] <= 1.0


def test_export_nao_tenta_rebaixar_o_layernorm(tmp_path, capfd):
    """Regressao: `opset_version=13`, copiado do train.py do openWakeWord, fazia
    o exportador tentar rebaixar `LayerNormalization` (que só existe do 17 em
    diante) e cuspir um RuntimeError a cada export. Era não-fatal, mas assustava.
    """
    pytest.importorskip("torch")
    onnx = pytest.importorskip("onnx")
    from treino.treinar import construir_rede, exportar_onnx

    destino = tmp_path / "m.onnx"
    exportar_onnx(construir_rede(128, 1), destino)

    assert "No Previous Version" not in (capfd.readouterr().err or "")
    m = onnx.load(destino)
    onnx.checker.check_model(m)
    versoes = {(i.domain or "ai.onnx"): i.version for i in m.opset_import}
    assert versoes["ai.onnx"] >= 17
    assert "LayerNormalization" in [n.op_type for n in m.graph.node]


def test_pesos_ida_e_volta(tmp_path):
    """Reexportar com outro opset não deve exigir retreinar."""
    torch = pytest.importorskip("torch")
    from treino.treinar import carregar_pesos, construir_rede, salvar_pesos

    original = construir_rede(128, 1)
    salvar_pesos(original, tmp_path / "m.pt")
    recarregada = carregar_pesos(tmp_path / "m.pt")
    x = torch.rand(4, FRAMES, DIM)
    with torch.no_grad():
        assert torch.allclose(original.eval()(x), recarregada(x))


def test_onnx_exportado_e_um_arquivo_so(tmp_path):
    """Sem isso o exportador dynamo deixa os pesos num `.onnx.data` ao lado, e
    quem copiasse só o .onnx para modelos/wake/ levaria um modelo quebrado."""
    pytest.importorskip("torch")
    from treino.treinar import construir_rede, exportar_onnx

    exportar_onnx(construir_rede(128, 1), tmp_path / "m.onnx")
    assert sorted(p.name for p in tmp_path.iterdir()) == ["m.onnx"]


def test_onnx_exportado_aceita_lote(tmp_path):
    """Lote dinamico e o que torna a avaliacao das ~481 mil janelas viavel."""
    pytest.importorskip("torch")
    ort = pytest.importorskip("onnxruntime")
    from treino.treinar import construir_rede, exportar_onnx

    destino = tmp_path / "m.onnx"
    exportar_onnx(construir_rede(128, 1), destino)
    s = ort.InferenceSession(str(destino), providers=["CPUExecutionProvider"])
    assert s.get_inputs()[0].shape[1:] == [FRAMES, DIM]
    saida = s.run(None, {"x": np.zeros((13, FRAMES, DIM), dtype=np.float32)})[0]
    assert saida.shape == (13, 1)
