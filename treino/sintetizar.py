"""Gera os clipes base com as vozes pt-BR do Piper.

Sao duas classes de clipe:

* **positivos** -- a palavra de ativacao, variando prosodia e contexto.
* **negativos adversariais** -- o que mais causa disparo falso: pedacos da propria
  palavra e vizinhos foneticos. Nao da para usar o `generate_adversarial_texts` do
  openWakeWord aqui, porque ele deriva os vizinhos do dicionario CMU (ingles). A
  lista abaixo e curada na mao para o portugues.

So 4 vozes existem em pt-BR, o que e pouca diversidade de falante. Duas defesas:
as gravacoes do dono da maquina (`treino.gravar`, que sao as amostras que mais
importam num assistente de um unico usuario) e a aumentacao (`treino.aumentar`).
"""

from __future__ import annotations

import itertools
from pathlib import Path
from typing import Iterator

import numpy as np

from .comum import PALAVRA, aparar_silencio, escrever_wav, reamostrar

# Contextos do positivo. Quem chama o assistente raramente diz a palavra isolada
# e com entonacao neutra -- as bordas coarticuladas mudam o fim da palavra, e o
# fim e o que dispara o modelo.
CONTEXTOS_POSITIVOS = [
    "{p}",
    "{p}.",
    "{p}?",
    "{p}!",
    "Ô {p}.",
    "Oi {p}.",
    "Ei {p}!",
    "{p}, ",
    "{p}, me ajuda.",
    "{p}, que horas são?",
]

# Negativos adversariais, em tres grupos:
#  1. pedacos da palavra -- o erro classico e disparar em "tóteles" ou "aris".
#  2. vizinhos foneticos em portugues.
#  3. fala cotidiana e palavras que aparecem perto do assistente (inclusive
#     "Sócrates"/"Platão", que quem conversa com um "Aristóteles" vai dizer).
NEGATIVOS_ADVERSARIAIS = [
    # 1. pedacos
    "aris", "ris", "risto", "tóteles", "óteles", "teles", "tótel", "arist",
    "aristó", "istóteles", "ristóteles", "aristote", "telis", "stóteles",
    # 2. vizinhos foneticos
    "Aristides", "Aristeu", "aristocrata", "aristocracia", "aristocrático",
    "artérias", "artístico", "artista", "arisco", "aviso", "avisos",
    "totalmente", "total", "protótipo", "hipotético", "hipótese", "apóstolo",
    "Teles", "Telêmaco", "Aristarco", "cartéis", "hotéis", "pastéis",
    "cristais", "arrastões", "registros", "ministério", "mistério",
    "estatísticas", "estatística", "escritório", "diretório", "território",
    # 3. fala cotidiana
    "Sócrates", "Platão", "filosofia", "filósofo", "olá", "oi", "bom dia",
    "boa noite", "boa tarde", "obrigado", "tudo bem", "que horas são",
    "toca música", "aumenta o volume", "desliga", "para", "cancela",
    "certo", "beleza", "então", "espera", "não", "sim", "claro",
    "como você está", "me explica", "qual é a previsão do tempo",
    "liga a luz", "apaga a luz", "que dia é hoje", "quanto é",
    "conta uma piada", "lembra disso", "esquece", "repete",
]


def _vozes(dir_vozes: Path) -> list[tuple[str, object, int]]:
    """Carrega as vozes disponiveis. Devolve (nome, PiperVoice, taxa)."""
    from piper import PiperVoice

    caminhos = sorted(dir_vozes.glob("*.onnx"))
    if not caminhos:
        raise FileNotFoundError(
            f"nenhuma voz do Piper em {dir_vozes}\n"
            "Baixe as 4 vozes pt-BR: for v in faber/medium cadu/medium "
            "jeff/medium edresson/low; do ./scripts/02_baixar_modelos.sh $v; done"
        )
    vozes = []
    for c in caminhos:
        v = PiperVoice.load(str(c))
        taxa = int(getattr(v.config, "sample_rate", 22_050))
        vozes.append((c.stem, v, taxa))
    return vozes


def _prosodias(rng: np.random.Generator, n: int) -> list[dict]:
    """Combinacoes de prosodia. length_scale > 1 = mais lento."""
    grade = list(itertools.product(
        [0.85, 1.0, 1.15],        # length_scale (velocidade)
        [0.55, 0.667, 0.8],       # noise_scale (variabilidade timbrica)
        [0.6, 0.8, 1.0],          # noise_w_scale (variabilidade de duracao)
    ))
    escolhidas = []
    for i in range(n):
        ls, ns, nw = grade[i % len(grade)]
        # jitter para nao repetir exatamente a mesma combinacao a cada volta
        escolhidas.append({
            "length_scale": float(ls + rng.uniform(-0.05, 0.05)),
            "noise_scale": float(ns + rng.uniform(-0.05, 0.05)),
            "noise_w_scale": float(nw + rng.uniform(-0.08, 0.08)),
            "volume": float(rng.uniform(0.7, 1.0)),
        })
    return escolhidas


def _sintetizar(voz, texto: str, cfg: dict, taxa: int) -> np.ndarray:
    from piper import SynthesisConfig

    syn = SynthesisConfig(**cfg)
    blocos = []
    for chunk in voz.synthesize(texto, syn_config=syn):
        dados = getattr(chunk, "audio_int16_bytes", None)
        blocos.append(np.frombuffer(dados, dtype=np.int16) if dados is not None
                      else np.asarray(chunk, dtype=np.int16))
    if not blocos:
        return np.zeros(0, dtype=np.int16)
    return reamostrar(np.concatenate(blocos), taxa)


def gerar(dir_vozes: Path, destino: Path, textos: list[str],
          por_voz: int, semente: int = 0) -> int:
    """Sintetiza `por_voz` clipes de cada voz, ciclando por `textos`.

    Devolve quantos clipes foram escritos.
    """
    rng = np.random.default_rng(semente)
    vozes = _vozes(dir_vozes)
    destino.mkdir(parents=True, exist_ok=True)
    n = 0
    for nome, voz, taxa in vozes:
        for i, cfg in enumerate(_prosodias(rng, por_voz)):
            texto = textos[i % len(textos)]
            audio = aparar_silencio(_sintetizar(voz, texto, cfg, taxa))
            if audio.size < 800:  # < 50 ms: sintese falhou
                continue
            escrever_wav(destino / f"{nome}_{i:04d}.wav", audio)
            n += 1
    return n


def gerar_positivos(dir_vozes: Path, destino: Path, por_voz: int = 90,
                    semente: int = 0) -> int:
    textos = [c.format(p=PALAVRA) for c in CONTEXTOS_POSITIVOS]
    return gerar(dir_vozes, destino, textos, por_voz, semente)


def gerar_negativos(dir_vozes: Path, destino: Path, por_voz: int = 240,
                    semente: int = 1) -> int:
    return gerar(dir_vozes, destino, NEGATIVOS_ADVERSARIAIS, por_voz, semente)
