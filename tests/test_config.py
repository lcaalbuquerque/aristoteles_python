import pytest

from aristoteles.config import Config


def test_defaults_validos():
    cfg = Config()
    cfg.validar()
    assert cfg.audio.amostras_por_bloco == 480  # 30 ms a 16 kHz


def test_carrega_yaml(tmp_path):
    (tmp_path / "c.yaml").write_text(
        "stt:\n  backend: vulkan\nllm:\n  effort: medium\n", encoding="utf-8"
    )
    cfg = Config.carregar(tmp_path / "c.yaml")
    assert cfg.stt.backend == "vulkan"
    assert cfg.llm.effort == "medium"
    assert cfg.audio.taxa_amostragem == 16_000  # default preservado


def test_chave_desconhecida_falha(tmp_path):
    (tmp_path / "c.yaml").write_text("llm:\n  temperatura: 0.7\n", encoding="utf-8")
    with pytest.raises(ValueError, match="desconhecida"):
        Config.carregar(tmp_path / "c.yaml")


def test_bloco_ms_invalido():
    cfg = Config()
    cfg.audio.bloco_ms = 25
    with pytest.raises(ValueError, match="webrtcvad"):
        cfg.validar()


def test_effort_alto_exige_thinking():
    """A API rejeita thinking desativado acima de effort 'high'."""
    cfg = Config()
    cfg.llm.pensar = False
    cfg.llm.effort = "max"
    with pytest.raises(ValueError, match="pensar"):
        cfg.validar()
