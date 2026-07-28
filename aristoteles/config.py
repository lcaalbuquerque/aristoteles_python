"""Carga de configuracao: defaults no codigo, sobrescritos por config.yaml."""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

RAIZ = Path(__file__).resolve().parent.parent


@dataclass
class AudioCfg:
    taxa_amostragem: int = 16_000
    bloco_ms: int = 30
    pre_roll_s: float = 1.5
    # Teto da fila de captura. Sem teto, esperar no gatilho acumula 32 KB/s de
    # audio que ninguem vai ler -- ~2,7 GB por dia num servico systemd.
    fila_maxima_s: float = 30.0
    dispositivo_entrada: str | int | None = None
    dispositivo_saida: str | int | None = None

    @property
    def amostras_por_bloco(self) -> int:
        return self.taxa_amostragem * self.bloco_ms // 1000

    @property
    def blocos_da_fila(self) -> int:
        return max(1, int(self.fila_maxima_s * 1000 / self.bloco_ms))


@dataclass
class VadCfg:
    agressividade: int = 2
    silencio_final_ms: int = 700
    fala_minima_ms: int = 300
    duracao_maxima_s: float = 20.0
    # Tempo para o usuario *comecar* a falar depois do gatilho. Era 4,0 e parecia
    # instantaneo, mas a culpa nao era deste numero: a cauda da palavra de ativacao
    # armava o endpointing e valia o silencio_final_ms de 700 ms. Corrigido no
    # gravar_ate_silencio; 6,0 da folga para formular a pergunta.
    espera_inicial_s: float = 6.0
    # Teto do que se absorve como sendo a palavra de ativacao. "Aristoteles" dura
    # ~0,9 s; o teto evita engolir a pergunta de quem fala tudo sem pausa.
    absorver_max_ms: int = 1_300
    # Gate de energia: o webrtcvad sozinho toma ruido de ventoinha/mic USB por fala.
    calibracao_ms: int = 600      # duracao da medicao do piso de ruido na inicializacao
    fator_acima_do_piso: float = 3.0
    piso_minimo: float = 0.01     # limiar absoluto minimo (protege ambiente silencioso)


@dataclass
class SttCfg:
    backend: str = "cpu"
    idioma: str = "pt"
    modelo_cpu: str = "small"
    compute_type: str = "int8"
    threads: int = 8
    servidor_url: str = "http://127.0.0.1:8178/inference"


@dataclass
class LlmCfg:
    modelo: str = "claude-opus-5"
    max_tokens: int = 1024
    effort: str = "low"
    pensar: bool = False
    turnos_historico: int = 10
    # Timeouts do cliente HTTP. O default do SDK e read=600s: se a API travar no
    # meio do stream, o assistente fica dez minutos calado. Num assistente de voz
    # e melhor desistir e dizer que deu erro. `read` e por leitura sem dados, nao
    # o total, entao 60s ja e muito folgado para uma resposta de 3 frases.
    timeout_conexao_s: float = 10.0
    timeout_leitura_s: float = 60.0
    tentativas: int = 2
    # Reconexao acima do que o SDK faz: ele retenta antes do stream comecar, mas
    # se a conexao cai no meio nao remonta a chamada. O cerebro reemite o turno --
    # so enquanto nada tiver sido falado, porque depois da primeira frase no
    # alto-falante repetir a resposta do zero seria pior que admitir o erro.
    reconexoes: int = 2
    espera_reconexao_s: float = 1.0  # dobra a cada tentativa
    prompt_sistema: str = (
        "Voce e Aristoteles, um assistente de voz. Responda em portugues do Brasil, "
        "em no maximo 3 frases curtas, sem markdown, listas ou emojis. "
        "Nunca inclua tags XML internas ou de sistema na resposta."
    )


@dataclass
class TtsCfg:
    voz: Path = Path("modelos/piper/pt_BR-faber-medium.onnx")
    velocidade: float = 1.0


@dataclass
class LogCfg:
    """Copia para arquivo de tudo que o app imprime no console."""
    ativo: bool = True
    arquivo: Path = Path("logs/aristoteles.log")
    # Rodizio, porque como servico systemd o app fica ligado por dias.
    max_bytes: int = 5_000_000
    backups: int = 3
    # `\r` e usado para reescrever a linha de status ("[ouvindo]" -> "voce: ...").
    # No arquivo isso viraria uma linha ilegivel, entao cada reescrita e gravada
    # como sua propria linha. Desligue para ver o fluxo exatamente como saiu.
    expandir_retorno: bool = True


@dataclass
class WakeCfg:
    modo: str = "push_to_talk"
    modelo: Path = Path("modelos/wake/aristoteles.onnx")
    limiar: float = 0.5
    cooldown_s: float = 1.5


@dataclass
class Config:
    audio: AudioCfg = field(default_factory=AudioCfg)
    vad: VadCfg = field(default_factory=VadCfg)
    stt: SttCfg = field(default_factory=SttCfg)
    llm: LlmCfg = field(default_factory=LlmCfg)
    tts: TtsCfg = field(default_factory=TtsCfg)
    wake: WakeCfg = field(default_factory=WakeCfg)
    log: LogCfg = field(default_factory=LogCfg)
    raiz: Path = RAIZ

    @classmethod
    def carregar(cls, caminho: Path | str | None = None) -> "Config":
        caminho = Path(caminho) if caminho else RAIZ / "config.yaml"
        bruto: dict[str, Any] = {}
        if caminho.exists():
            bruto = yaml.safe_load(caminho.read_text(encoding="utf-8")) or {}

        secoes = {f.name: f.type for f in dataclasses.fields(cls) if f.name != "raiz"}
        kwargs: dict[str, Any] = {}
        for nome in secoes:
            classe = {
                "audio": AudioCfg, "vad": VadCfg, "stt": SttCfg,
                "llm": LlmCfg, "tts": TtsCfg, "wake": WakeCfg, "log": LogCfg,
            }[nome]
            kwargs[nome] = _montar(classe, bruto.get(nome) or {}, nome)

        cfg = cls(**kwargs)
        cfg.validar()
        return cfg

    def caminho(self, relativo: Path | str) -> Path:
        """Resolve um caminho do config contra a raiz do projeto."""
        p = Path(relativo)
        return p if p.is_absolute() else self.raiz / p

    def validar(self) -> None:
        if self.audio.bloco_ms not in (10, 20, 30):
            raise ValueError("audio.bloco_ms deve ser 10, 20 ou 30 (exigencia do webrtcvad)")
        if self.audio.taxa_amostragem not in (8_000, 16_000, 32_000, 48_000):
            raise ValueError("audio.taxa_amostragem deve ser 8000, 16000, 32000 ou 48000")
        if not 0 <= self.vad.agressividade <= 3:
            raise ValueError("vad.agressividade deve estar entre 0 e 3")
        if self.stt.backend not in ("cpu", "vulkan"):
            raise ValueError("stt.backend deve ser 'cpu' ou 'vulkan'")
        # Restricao real da API: desligar o thinking so e aceito ate effort "high".
        if not self.llm.pensar and self.llm.effort in ("xhigh", "max"):
            raise ValueError(
                "llm.effort 'xhigh'/'max' exige llm.pensar: true "
                "(a API rejeita thinking desativado acima de 'high')"
            )


def _montar(classe, dados: dict, secao: str):
    """Instancia uma dataclass a partir do dict do YAML, com checagem de chave desconhecida."""
    validos = {f.name: f for f in dataclasses.fields(classe)}
    desconhecidas = set(dados) - set(validos)
    if desconhecidas:
        raise ValueError(f"config.yaml: chave(s) desconhecida(s) em '{secao}': {sorted(desconhecidas)}")
    kwargs = {}
    for chave, valor in dados.items():
        if valor is None and validos[chave].default is None:
            kwargs[chave] = None
        elif validos[chave].type in (Path, "Path"):
            kwargs[chave] = Path(valor)
        else:
            kwargs[chave] = valor
    return classe(**kwargs)
