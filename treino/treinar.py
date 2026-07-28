"""Treina o classificador da wake word e exporta o .onnx.

A arquitetura e copiada do `openwakeword/train.py` (classe `Model`, tipo "dnn")
porque o runtime carrega o resultado pelo `openwakeword.model.Model`: mudar as
camadas quebraria o carregamento. E uma rede densa pequena sobre as features
(16, 96) -- ~200 mil parametros, treina em CPU em minutos.

Duas diferencas deliberadas em relacao ao pipeline oficial:

* **Negativos grandes ficam em memmap.** O arquivo do ACAV100M tem 17,3 GB e nao
  cabe confortavelmente na memoria; sorteamos linhas direto do disco, com os
  indices ordenados para o acesso ficar quase sequencial.
* **Validacao com `sliding_window_view`**, que e uma vista sem copia -- o oficial
  faz um `np.stack` que aloca alguns GB para as ~481 mil janelas.

O criterio de selecao e o mesmo do `auto_train`: entre os checkpoints que ficam
dentro do alvo de falsos positivos por hora, guarda o de maior recall.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from .comum import DIM, FRAMES

# O `auto_train` oficial cravou 11.3 h para o validation_set_features.npy, mas o
# arquivo publicado tem 481.345 frames de 80 ms = 10,7 h. Calculamos da propria
# matriz; a diferenca de 5% e irrelevante para calibrar limiar, mas nao ha motivo
# para herdar um numero errado.
SEGUNDOS_POR_FRAME = 0.08


@dataclass
class Config:
    passos: int = 12_000
    lote_por_classe: int = 256
    lr: float = 1e-4
    peso_negativo_max: float = 1_500.0
    alvo_fp_por_hora: float = 0.2
    limiar: float = 0.5
    camada: int = 128
    blocos: int = 1
    semente: int = 0
    validar_a_cada: int = 500


@dataclass
class Dados:
    """Caminhos dos .npy de features."""
    positivos: Path
    positivos_val: Path
    negativos_adv: Path
    negativos_adv_val: Path
    negativos_grandes: list[Path] = field(default_factory=list)
    fp_validacao: Path | None = None


def construir_rede(camada: int, blocos: int):
    """Replica exatamente a `Net` do openwakeword/train.py (model_type="dnn")."""
    import torch
    from torch import nn

    class FCNBlock(nn.Module):
        def __init__(self, dim):
            super().__init__()
            self.fcn_layer = nn.Linear(dim, dim)
            self.relu = nn.ReLU()
            self.layer_norm = nn.LayerNorm(dim)

        def forward(self, x):
            return self.relu(self.layer_norm(self.fcn_layer(x)))

    class Net(nn.Module):
        def __init__(self):
            super().__init__()
            self.flatten = nn.Flatten()
            self.layer1 = nn.Linear(FRAMES * DIM, camada)
            self.relu1 = nn.ReLU()
            self.layernorm1 = nn.LayerNorm(camada)
            self.blocks = nn.ModuleList([FCNBlock(camada) for _ in range(blocos)])
            self.last_layer = nn.Linear(camada, 1)
            self.last_act = nn.Sigmoid()

        def forward(self, x):
            x = self.relu1(self.layernorm1(self.layer1(self.flatten(x))))
            for b in self.blocks:
                x = b(x)
            return self.last_act(self.last_layer(x))

    return Net()


class FonteNegativa:
    """Sorteia exemplos negativos de um .npy grande, sempre por memmap.

    Os dois arquivos publicados pelo openWakeWord tem layouts diferentes, e
    confundi-los custa uma excecao no meio do treino:

    * `openwakeword_features_ACAV100M_2000_hrs_16bit.npy` ja vem **janelado**:
      (5.625.000, 16, 96) em float16 -- 5,6 milhoes de exemplos independentes,
      que sao as 2000 h anunciadas (5.625.000 x 16 x 80 ms). Basta sortear linhas.
    * `validation_set_features.npy` vem **continuo**: (481.345, 96) em float32.
      Ali as janelas se recortam com passo 1, porque para medir falso positivo por
      hora interessa cada posicao possivel, nao amostras independentes.

    Esta classe aceita os dois. Em nenhum caso o arquivo inteiro entra na memoria.
    """

    def __init__(self, caminho: Path) -> None:
        self.dados = np.load(caminho, mmap_mode="r")
        forma = self.dados.shape
        if self.dados.ndim == 3:
            if forma[1:] != (FRAMES, DIM):
                raise ValueError(
                    f"{caminho}: esperado (N, {FRAMES}, {DIM}), veio {forma}")
            self.janelado = True
            self.n = forma[0]
        elif self.dados.ndim == 2 and forma[1] == DIM:
            self.janelado = False
            self.n = forma[0] - FRAMES
        else:
            raise ValueError(
                f"{caminho}: esperado (N, {DIM}) ou (N, {FRAMES}, {DIM}), veio {forma}")
        if self.n <= 0:
            raise ValueError(f"{caminho}: dados insuficientes ({forma})")

    @property
    def horas(self) -> float:
        frames = self.n * FRAMES if self.janelado else self.dados.shape[0]
        return frames * SEGUNDOS_POR_FRAME / 3600.0

    def amostrar(self, quantos: int, rng: np.random.Generator) -> np.ndarray:
        idx = rng.integers(0, self.n, quantos)
        if self.janelado:
            # Ordenar os indices deixa o acesso ao memmap quase sequencial, o que
            # importa num arquivo de 17 GB que nao cabe todo no cache de paginas.
            return self.dados[np.sort(idx)].astype(np.float32)
        return np.stack([self.dados[i:i + FRAMES] for i in np.sort(idx)]).astype(np.float32)


def _janelas(caminho: Path) -> np.ndarray:
    """Vista deslizante (sem copia) de um .npy continuo -> (N, 16, 96)."""
    dados = np.load(caminho, mmap_mode="r")
    if dados.ndim == 3:
        return dados
    return np.lib.stride_tricks.sliding_window_view(
        dados, (FRAMES, DIM)).squeeze(1)


def _carregar(caminho: Path) -> np.ndarray:
    d = np.load(caminho)
    if d.ndim != 3 or d.shape[1:] != (FRAMES, DIM):
        raise ValueError(f"{caminho}: esperado (N, {FRAMES}, {DIM}), veio {d.shape}")
    return d.astype(np.float32)


def _pontuar(rede, dados, lote: int = 8192) -> np.ndarray:
    """Probabilidades da rede para todos os exemplos, em blocos."""
    import torch

    saidas = []
    with torch.no_grad():
        for i in range(0, len(dados), lote):
            x = torch.from_numpy(np.ascontiguousarray(
                dados[i:i + lote], dtype=np.float32))
            saidas.append(rede(x).squeeze(-1).numpy())
    return np.concatenate(saidas) if saidas else np.zeros(0, dtype=np.float32)


def _ponto_de_operacao(p_pos: np.ndarray, p_fp: np.ndarray, horas: float,
                       alvo_fp: float) -> tuple[float, float, float]:
    """Melhor (limiar, recall, fp/h) deste checkpoint para o alvo de fp/h.

    Escolher o checkpoint por recall num limiar FIXO (0.5) e enganoso: quem
    entrega recall 0.95 com 15 fp/h costuma virar um modelo melhor -- basta subir
    o limiar -- do que quem entrega recall 0.60 com 0 fp/h, que ja perdeu a
    informacao. Entao, para cada checkpoint, procuramos o menor limiar que cumpre
    o alvo e medimos o recall ali; e o mesmo criterio do `treino.avaliar`.

    Se nenhum limiar cumpre o alvo, devolve o mais alto (o de menor fp/h), para o
    chamador poder comparar checkpoints ainda fora do alvo.
    """
    candidatos = np.unique(np.concatenate([
        np.linspace(0.01, 0.99, 99), np.array([0.995, 0.999])]))
    melhor = None
    for t in candidatos:
        fp_h = float((p_fp >= t).sum() / horas)
        rec = float((p_pos >= t).mean()) if p_pos.size else 0.0
        if fp_h <= alvo_fp:
            return float(t), rec, fp_h
        melhor = (float(t), rec, fp_h)
    return melhor if melhor is not None else (1.0, 0.0, 0.0)


def treinar(dados: Dados, cfg: Config = Config()):
    """Treina e devolve (rede, historico). A rede devolvida e a melhor vista."""
    import copy

    import torch

    rng = np.random.default_rng(cfg.semente)
    torch.manual_seed(cfg.semente)

    pos = _carregar(dados.positivos)
    pos_val = _carregar(dados.positivos_val)
    adv = _carregar(dados.negativos_adv)
    adv_val = _carregar(dados.negativos_adv_val)
    grandes = [FonteNegativa(p) for p in dados.negativos_grandes]

    print(f"  positivos: {len(pos)} (val {len(pos_val)})")
    print(f"  negativos adversariais: {len(adv)} (val {len(adv_val)})")
    for p, g in zip(dados.negativos_grandes, grandes):
        print(f"  negativos: {p.name} -> {g.horas:.0f} h")

    if dados.fp_validacao is not None:
        fp_janelas = _janelas(dados.fp_validacao)
        horas_fp = np.load(dados.fp_validacao, mmap_mode="r").shape[0] \
            * SEGUNDOS_POR_FRAME / 3600.0
        print(f"  validação de falso positivo: {len(fp_janelas)} janelas / "
              f"{horas_fp:.1f} h")
    else:
        # Sem o conjunto oficial, usa os negativos adversariais de validacao. Da
        # uma taxa de fp/hora que NAO e comparavel com a do openWakeWord, porque
        # essas amostras sao dificeis de proposito.
        fp_janelas = adv_val
        horas_fp = len(adv_val) * 2.0 / 3600.0
        print(f"  validação de falso positivo: adversariais ({horas_fp:.2f} h) "
              "-- fp/h não comparável com o oficial")

    rede = construir_rede(cfg.camada, cfg.blocos)
    otim = torch.optim.Adam(rede.parameters(), lr=cfg.lr)
    pesos_neg = np.linspace(1.0, cfg.peso_negativo_max, cfg.passos)

    melhor = {"rede": None, "recall": -1.0, "fp": float("inf"), "passo": -1}
    historico: list[dict] = []

    for passo in range(cfg.passos):
        n = cfg.lote_por_classe
        lote_pos = pos[rng.integers(0, len(pos), n)]
        negs = [adv[rng.integers(0, len(adv), n)]]
        for g in grandes:
            negs.append(g.amostrar(n, rng))
        lote_neg = np.concatenate(negs)

        x = torch.from_numpy(np.concatenate([lote_pos, lote_neg]))
        y = torch.cat([torch.ones(len(lote_pos)), torch.zeros(len(lote_neg))])[:, None]
        w = torch.full_like(y, float(pesos_neg[passo]))
        w[y == 1] = 1.0

        pred = rede(x)
        perda = torch.nn.functional.binary_cross_entropy(pred, y, weight=w)
        otim.zero_grad()
        perda.backward()
        otim.step()

        ultimo = passo == cfg.passos - 1
        if (passo + 1) % cfg.validar_a_cada == 0 or ultimo:
            rede.eval()
            p_pos = _pontuar(rede, pos_val)
            p_fp = _pontuar(rede, fp_janelas)
            rede.train()

            t, rec, fp_h = _ponto_de_operacao(p_pos, p_fp, horas_fp,
                                              cfg.alvo_fp_por_hora)
            # recall no limiar nominal, so para acompanhar a evolucao
            rec_nominal = float((p_pos >= cfg.limiar).mean())
            historico.append({"passo": passo + 1, "perda": float(perda),
                              "limiar": t, "recall": rec, "fp_por_hora": fp_h,
                              "recall_nominal": rec_nominal})
            print(f"  passo {passo + 1:6d}  perda {float(perda):7.4f}  "
                  f"limiar {t:5.3f}  recall {rec:.3f}  fp/h {fp_h:6.2f}"
                  f"   (recall@{cfg.limiar:.1f} {rec_nominal:.3f})")

            # Vence o maior recall no ponto de operacao. Empate ou alvo nao
            # atingido por ninguem: desempata pelo menor fp/h.
            if rec > melhor["recall"] or (rec == melhor["recall"]
                                          and fp_h < melhor["fp"]):
                melhor = {"rede": copy.deepcopy(rede), "recall": rec,
                          "fp": fp_h, "passo": passo + 1, "limiar": t}

    if melhor["rede"] is None:
        melhor["rede"] = rede
    print(f"\n  melhor: passo {melhor['passo']}, recall {melhor['recall']:.3f} "
          f"com limiar {melhor.get('limiar', cfg.limiar):.3f} "
          f"({melhor['fp']:.2f} fp/h)")
    if melhor["fp"] > cfg.alvo_fp_por_hora:
        print(f"  nenhum checkpoint atingiu {cfg.alvo_fp_por_hora} fp/h -- "
              "confira o `treino avaliar` antes de confiar no modelo.")
    return melhor["rede"], historico


def salvar_pesos(rede, destino: Path) -> None:
    """Guarda o state_dict ao lado do .onnx.

    Sem isto, mudar qualquer detalhe do export (opset, eixo dinamico) obriga a
    retreinar do zero -- que sao ~15 minutos so para reescrever um arquivo.
    """
    import torch

    destino.parent.mkdir(parents=True, exist_ok=True)
    torch.save(rede.state_dict(), destino)


def carregar_pesos(caminho: Path, camada: int = 128, blocos: int = 1):
    """Reconstroi a rede a partir do state_dict salvo por `salvar_pesos`."""
    import torch

    rede = construir_rede(camada, blocos)
    rede.load_state_dict(torch.load(caminho, weights_only=True))
    rede.eval()
    return rede


def exportar_onnx(rede, destino: Path) -> None:
    """Exporta no formato que o openwakeword.model.Model espera carregar.

    Com eixo de lote dinamico, ao contrario dos modelos oficiais (que sao
    [1, 16, 96] fixo). O runtime nao muda -- ele sempre passa uma janela por vez,
    e o `openwakeword.model.Model` le apenas `shape[1]` para descobrir o numero
    de frames, nunca a dimensao 0. Mas avaliar as ~481 mil janelas do conjunto de
    validacao uma a uma seria inviavel, e com lote dinamico da para faze-lo em
    blocos.
    """
    import torch

    destino.parent.mkdir(parents=True, exist_ok=True)
    rede.eval()
    # opset 17, nao 13. O `train.py` do openWakeWord pede 13, e copiar esse numero
    # produz um traceback feio a cada export:
    #
    #   RuntimeError: No Previous Version of LayerNormalization exists
    #
    # `LayerNormalization` so existe a partir do opset 17. O torch antigo que eles
    # usavam decompunha o LayerNorm em operacoes primitivas (ReduceMean, Sub, Div),
    # que cabiam no 13; o torch moderno emite o operador fundido, e nao ha como
    # rebaixa-lo. O erro era nao-fatal -- o exportador desistia da conversao e
    # salvava em opset 18, que funciona -- mas pedir 17 direto evita o ruido.
    # Qualquer onnxruntime >= 1.13 le opset 17.
    comum = dict(opset_version=17, input_names=["x"], output_names=["y"])
    exemplo = torch.rand(2, FRAMES, DIM)  # 2, nao 1: lote 1 viraria eixo estatico
    # external_data=False forca um .onnx autocontido. O exportador dynamo, por
    # padrao, joga os pesos num `.onnx.data` ao lado -- e quem copiasse so o
    # .onnx para modelos/wake/ levaria um modelo quebrado. Sao ~850 KB; nao ha
    # motivo para separar.
    try:
        # torch >= 2.5 (exportador dynamo) prefere `dynamic_shapes`; `dynamic_axes`
        # ainda funciona mas emite aviso de descontinuidade.
        torch.onnx.export(
            rede.to("cpu"), exemplo, str(destino), external_data=False,
            dynamic_shapes={"x": {0: torch.export.Dim("lote")}}, **comum)
    except (AttributeError, TypeError):
        torch.onnx.export(rede.to("cpu"), exemplo, str(destino),
                          dynamic_axes={"x": {0: "lote"}, "y": {0: "lote"}}, **comum)
