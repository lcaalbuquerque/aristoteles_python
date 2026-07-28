"""Converte clipes de 2 s nas features (16, 96) que o modelo consome.

Usa o `AudioFeatures` do proprio openWakeWord, que roda em ONNX -- então a
extracao nao precisa de torch. E obrigatorio usar o dele: o melspectrogram e o
`speech_embedding` do Google sao exatamente os mesmos que rodam na inferencia,
e features calculadas de outro jeito nao serviriam para nada.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, Iterator

import numpy as np

from .comum import AMOSTRAS_CLIPE, DIM, FRAMES


def _lotes(clipes: Iterable[np.ndarray], tamanho: int) -> Iterator[np.ndarray]:
    lote: list[np.ndarray] = []
    for c in clipes:
        if c.size != AMOSTRAS_CLIPE:
            raise ValueError(f"clipe com {c.size} amostras, esperado {AMOSTRAS_CLIPE}")
        lote.append(c)
        if len(lote) == tamanho:
            yield np.stack(lote)
            lote = []
    if lote:
        yield np.stack(lote)


def extrair(clipes: Iterable[np.ndarray], destino: Path, lote: int = 128,
            ncpu: int = 8, progresso: bool = True) -> int:
    """Calcula as features de `clipes` e salva um .npy (N, 16, 96) float32.

    Devolve N. Acumula em memoria: a 6 KB por exemplo, 100 mil exemplos sao
    ~600 MB, o que cabe nos 31 GB desta maquina.
    """
    from openwakeword.utils import AudioFeatures

    af = AudioFeatures(inference_framework="onnx", ncpu=ncpu)
    partes: list[np.ndarray] = []
    n = 0
    for bloco in _lotes(clipes, lote):
        f = af.embed_clips(bloco, batch_size=lote, ncpu=ncpu)
        if f.shape[1:] != (FRAMES, DIM):
            raise ValueError(f"features {f.shape[1:]}, esperado {(FRAMES, DIM)}")
        partes.append(f.astype(np.float32))
        n += len(bloco)
        if progresso:
            print(f"\r  features: {n}", end="", flush=True)
    if progresso:
        print()
    if not partes:
        raise ValueError("nenhum clipe para extrair")
    dados = np.concatenate(partes)
    destino.parent.mkdir(parents=True, exist_ok=True)
    np.save(destino, dados)
    return len(dados)


def janelas_de_features(caminho: Path) -> np.ndarray:
    """Recorta um .npy continuo de features em janelas de 16 frames.

    Os negativos pre-computados do openWakeWord (ACAV100M, conjunto de validacao)
    vem como um fluxo continuo (N, 96), nao em janelas. O train.py oficial faz o
    mesmo recorte com passo 1.
    """
    dados = np.load(caminho, mmap_mode="r")
    if dados.ndim == 3:
        return np.asarray(dados)
    if dados.ndim != 2 or dados.shape[1] != DIM:
        raise ValueError(f"formato inesperado {dados.shape}")
    n = dados.shape[0] - FRAMES
    if n <= 0:
        raise ValueError(f"{caminho} tem só {dados.shape[0]} frames")
    return np.stack([dados[i:i + FRAMES] for i in range(n)]).astype(np.float32)
