"""Mede o modelo treinado e sugere o limiar.

O README manda calibrar `wake.limiar` na mao, comecando em 0.5 e subindo se
disparar sozinho. Isso funciona, mas gasta a sua paciencia. Aqui a curva sai dos
dados: para cada limiar candidato, o recall nos positivos de validacao e os falsos
positivos por hora nas ~10,7 h de audio sem a palavra.

A escolha e um trade-off explicito, nao um numero mágico: 1 fp/h e um disparo
sozinho por hora, e recall 0.95 e uma chamada perdida em vinte.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from .comum import DIM, FRAMES

LIMIARES = [0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95]


def _prever(sessao, x: np.ndarray, lote: int = 8192) -> np.ndarray:
    nome = sessao.get_inputs()[0].name
    saidas = []
    for i in range(0, len(x), lote):
        b = np.ascontiguousarray(x[i:i + lote], dtype=np.float32)
        saidas.append(sessao.run(None, {nome: b})[0].reshape(-1))
    return np.concatenate(saidas) if saidas else np.zeros(0, dtype=np.float32)


def avaliar(modelo: Path, dir_features: Path, dir_dados: Path,
            tolerancia_fp: float = 0.2) -> dict:
    import onnxruntime as ort

    sessao = ort.InferenceSession(str(modelo), providers=["CPUExecutionProvider"])
    forma = sessao.get_inputs()[0].shape
    print(f"modelo: {modelo.name}  entrada={forma}")
    if list(forma[1:]) != [FRAMES, DIM]:
        print(f"  AVISO: esperado [_, {FRAMES}, {DIM}]")

    pos = np.load(dir_features / "positivos_val.npy").astype(np.float32)
    p_pos = _prever(sessao, pos)

    adv = np.load(dir_features / "negativos_adv_val.npy").astype(np.float32)
    p_adv = _prever(sessao, adv)

    fp_arq = dir_dados / "validation_set_features.npy"
    if fp_arq.exists():
        cont = np.load(fp_arq, mmap_mode="r")
        janelas = np.lib.stride_tricks.sliding_window_view(
            cont, (FRAMES, DIM)).squeeze(1)
        horas = cont.shape[0] * 0.08 / 3600.0
        p_fp = _prever(sessao, janelas)
        print(f"  {len(pos)} positivos de validação, {len(adv)} adversariais, "
              f"{horas:.1f} h de áudio negativo")
    else:
        p_fp, horas = None, 0.0
        print(f"  {len(pos)} positivos de validação, {len(adv)} adversariais")
        print("  (sem validation_set_features.npy: fp/h indisponível)")

    print(f"\n  {'limiar':>7} {'recall':>7} {'fp/h':>9} {'adv.disparados':>16}")
    linhas = []
    for t in LIMIARES:
        rec = float((p_pos >= t).mean())
        adv_taxa = float((p_adv >= t).mean())
        fp_h = float((p_fp >= t).sum() / horas) if p_fp is not None else float("nan")
        linhas.append({"limiar": t, "recall": rec, "fp_por_hora": fp_h,
                       "adversariais": adv_taxa})
        print(f"  {t:7.2f} {rec:7.3f} {fp_h:9.2f} {adv_taxa:15.1%}")

    # Menor limiar dentro da tolerancia -- dentro dela, quanto menor o limiar,
    # maior o recall. A tolerancia default e 0,2 fp/h, a mesma que o treino
    # otimiza: nao faz sentido treinar para 0,2 e sugerir um ponto com 5x isso.
    #
    # Cuidado com a tentacao de afrouxar para 1 fp/h: 0,93 fp/h sao ~22 despertares
    # espurios por dia, contra ~4,5 em 0,19 -- e o que se compra com isso costuma
    # ser 3 ou 4 pontos de recall. Mau negocio num aparelho sempre ligado.
    viaveis = [l for l in linhas
               if l["fp_por_hora"] <= tolerancia_fp] if p_fp is not None else []
    if viaveis:
        s = min(viaveis, key=lambda l: l["limiar"])
        print(f"\n  sugestão: wake.limiar = {s['limiar']:.2f}  "
              f"(recall {s['recall']:.1%}, {s['fp_por_hora']:.2f} fp/h "
              f"~ {s['fp_por_hora'] * 24:.0f} despertares espúrios/dia)")
        if s["recall"] < 0.8:
            print("  recall baixo: mais gravações suas (python -m treino gravar)")
            print("  ajudam mais que mais passos de treino.")
        print(f"  (tolerância {tolerancia_fp} fp/h; a tabela acima mostra o "
              "trade-off se você preferir outro ponto)")
    elif p_fp is not None:
        print(f"\n  nenhum limiar fica abaixo de {tolerancia_fp} fp/h.")
        print("  Se faltam os negativos em escala: ./scripts/06_baixar_dados_wake.sh --tudo")
    return {"linhas": linhas, "tolerancia_fp": tolerancia_fp}
