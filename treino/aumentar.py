"""Aumentacao dos clipes: e daqui que sai a diversidade que 4 vozes nao dao.

Substitui o audiomentations/torch-audiomentations/speechbrain do pipeline oficial
por numpy + scipy. Cada volta de aumentacao aplica, com sorteio:

1. **posicao** aleatoria na janela de 2 s -- sem isso o modelo aprende que a
   palavra sempre comeca no mesmo instante e falha no uso real.
2. **reverberacao** por convolucao com uma resposta impulsiva de sala real
   (dataset MIT, 271 salas). E o que faz o modelo funcionar a 2 m do mic e nao so
   colado nele.
3. **ruido de fundo** numa SNR sorteada. Alem de clipes reais, se houver, geramos
   ruido colorido: o branco cobre chiado de mic, o rosa/marrom cobre ventoinha e
   ar-condicionado -- exatamente o ruido estacionario de banda larga que o
   README documenta como problema nesta maquina.
4. **ganho** aleatorio, para nao amarrar a decisao ao volume.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterator, Sequence

import numpy as np
import scipy.signal

from .comum import AMOSTRAS_CLIPE, TAXA, encaixar, ler_wav, rms

SNR_DB = (0.0, 25.0)      # faixa de sorteio da relacao sinal/ruido
GANHO = (0.25, 1.0)       # fator multiplicativo no sinal final
PROB_RIR = 0.7            # fracao dos clipes que leva reverberacao
PROB_FUNDO = 0.85         # fracao que leva ruido de fundo


def carregar_rirs(dir_rir: Path | None) -> list[np.ndarray]:
    """Le as respostas impulsivas, normalizadas para nao mudar o ganho."""
    if dir_rir is None or not dir_rir.is_dir():
        return []
    rirs = []
    for p in sorted(dir_rir.rglob("*.wav")):
        h = ler_wav(p).astype(np.float32)
        pico = np.abs(h).max()
        if pico > 0:
            rirs.append(h / pico)
    return rirs


def carregar_fundos(dir_fundo: Path | None, max_arquivos: int = 400) -> list[np.ndarray]:
    """Le clipes de ruido de fundo reais (opcional)."""
    if dir_fundo is None or not dir_fundo.is_dir():
        return []
    fundos = []
    for p in sorted(dir_fundo.rglob("*.wav"))[:max_arquivos]:
        a = ler_wav(p)
        if a.size >= AMOSTRAS_CLIPE:
            fundos.append(a)
    return fundos


def ruido_colorido(n: int, expoente: float, rng: np.random.Generator) -> np.ndarray:
    """Ruido com espectro 1/f**expoente. 0=branco, 1=rosa, 2=marrom."""
    # Constroi no dominio da frequencia: mais simples e mais barato que filtrar.
    freqs = np.fft.rfftfreq(n, d=1.0 / TAXA)
    escala = np.ones_like(freqs)
    escala[1:] = freqs[1:] ** (-expoente / 2.0)
    escala[0] = escala[1] if len(escala) > 1 else 1.0
    fase = rng.uniform(0, 2 * np.pi, len(freqs))
    espectro = escala * np.exp(1j * fase)
    sinal = np.fft.irfft(espectro, n=n)
    pico = np.abs(sinal).max()
    return (sinal / pico if pico > 0 else sinal).astype(np.float32)


def _fundo_sorteado(fundos: Sequence[np.ndarray], rng: np.random.Generator) -> np.ndarray:
    """Um trecho de 2 s de ruido: real se houver, senao colorido."""
    if fundos and rng.random() < 0.6:
        f = fundos[int(rng.integers(len(fundos)))]
        i = int(rng.integers(0, len(f) - AMOSTRAS_CLIPE + 1))
        trecho = f[i:i + AMOSTRAS_CLIPE].astype(np.float32)
        pico = np.abs(trecho).max()
        return trecho / pico if pico > 0 else trecho
    expoente = float(rng.choice([0.0, 1.0, 1.5, 2.0]))
    return ruido_colorido(AMOSTRAS_CLIPE, expoente, rng)


def aplicar_rir(audio: np.ndarray, rir: np.ndarray) -> np.ndarray:
    """Convolui e devolve o mesmo numero de amostras, preservando o RMS."""
    antes = rms(audio)
    seco = audio.astype(np.float32)
    molhado = scipy.signal.fftconvolve(seco, rir, mode="full")[:len(seco)]
    depois = rms(molhado)
    if depois > 0 and antes > 0:
        molhado *= antes / depois
    return molhado


def aumentar_clipe(audio: np.ndarray, rirs: Sequence[np.ndarray],
                   fundos: Sequence[np.ndarray],
                   rng: np.random.Generator) -> np.ndarray:
    """Uma amostra aumentada de 2 s (int16) a partir de um clipe seco."""
    x = encaixar(audio, rng).astype(np.float32)

    if rirs and rng.random() < PROB_RIR:
        x = aplicar_rir(x, rirs[int(rng.integers(len(rirs)))])

    if rng.random() < PROB_FUNDO:
        fundo = _fundo_sorteado(fundos, rng)
        snr = float(rng.uniform(*SNR_DB))
        r_sinal, r_fundo = rms(x), rms(fundo)
        if r_sinal > 0 and r_fundo > 0:
            # fundo escalado para atingir a SNR pedida
            x = x + fundo * (r_sinal / r_fundo) * (10.0 ** (-snr / 20.0))

    x *= float(rng.uniform(*GANHO))

    # Normaliza so se estourou -- cortar em 32767 criaria distorcao que o modelo
    # poderia usar como atalho para reconhecer os positivos.
    pico = np.abs(x).max()
    if pico > 32_767:
        x *= 32_767 / pico
    return x.astype(np.int16)


def aumentar_clipes(caminhos: Sequence[Path], voltas: int,
                    rirs: Sequence[np.ndarray], fundos: Sequence[np.ndarray],
                    semente: int = 0) -> Iterator[np.ndarray]:
    """Gera `voltas` variantes de cada wav da lista, em ordem embaralhada.

    Recebe uma lista de arquivos, e nao um diretorio, porque a divisao
    treino/validacao tem de acontecer ANTES da aumentacao: se variantes do mesmo
    clipe caissem nos dois lados, o recall de validacao viria inflado.
    """
    rng = np.random.default_rng(semente)
    clipes = [ler_wav(p) for p in caminhos]
    if not clipes:
        raise FileNotFoundError("nenhum wav para aumentar")
    for _ in range(voltas):
        for i in rng.permutation(len(clipes)):
            yield aumentar_clipe(clipes[i], rirs, fundos, rng)
