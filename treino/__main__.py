"""Orquestra o treino da wake word.

    python -m treino sintetizar   # clipes das 4 vozes pt-BR do Piper
    python -m treino gravar       # ~100 amostras da SUA voz (o dado que mais vale)
    python -m treino features     # aumentacao + features (16, 96)
    python -m treino treinar      # treina e exporta modelos/wake/aristoteles.onnx
    python -m treino avaliar      # mede o modelo pronto, sugere o limiar

    python -m treino tudo         # sintetizar -> features -> treinar -> avaliar

Estado fica em `dados_wake/`, que esta no .gitignore. Cada etapa e retomavel:
reexecutar nao refaz o que ja existe.
"""

from __future__ import annotations

import argparse
import sys
import zlib
from pathlib import Path

import numpy as np

from . import aumentar, features, sintetizar, treinar
from .comum import PALAVRA

RAIZ = Path(__file__).resolve().parent.parent
DADOS = RAIZ / "dados_wake"
CLIPES = DADOS / "clipes"
FEATURES = DADOS / "features"
VOZES = RAIZ / "modelos" / "piper"
SAIDA_ONNX = RAIZ / "modelos" / "wake" / "aristoteles.onnx"

FRACAO_VAL = 0.15  # fracao dos clipes reservada para validacao


def _dividir(diretorio: Path, semente: int = 0) -> tuple[list[Path], list[Path]]:
    """Divide os wavs em (treino, validacao) por arquivo, de forma estavel."""
    todos = sorted(diretorio.glob("*.wav"))
    if not todos:
        return [], []
    rng = np.random.default_rng(semente)
    ordem = rng.permutation(len(todos))
    n_val = max(1, int(len(todos) * FRACAO_VAL))
    val = [todos[i] for i in ordem[:n_val]]
    treino = [todos[i] for i in ordem[n_val:]]
    return treino, val


def cmd_sintetizar(args) -> int:
    n = sintetizar.gerar_positivos(VOZES, CLIPES / "positivos_tts",
                                   por_voz=args.positivos_por_voz)
    print(f"positivos (TTS): {n} clipes em {CLIPES / 'positivos_tts'}")
    n = sintetizar.gerar_negativos(VOZES, CLIPES / "negativos_adv",
                                   por_voz=args.negativos_por_voz)
    print(f"negativos adversariais: {n} clipes em {CLIPES / 'negativos_adv'}")
    return 0


def cmd_gravar(args) -> int:
    from . import gravar

    gravar.gravar(CLIPES / "positivos_eu", args.quantas, args.dispositivo)
    return 0


def cmd_features(args) -> int:
    rirs = aumentar.carregar_rirs(DADOS / "rir")
    fundos = aumentar.carregar_fundos(DADOS / "fundo")
    print(f"aumentação: {len(rirs)} RIRs, {len(fundos)} clipes de fundo "
          f"(+ ruído colorido sintético)")
    if not rirs:
        print("  AVISO: sem RIRs. Rode ./scripts/06_baixar_dados_wake.sh --")
        print("  sem reverberação o modelo só funciona colado no microfone.")

    dir_tts = CLIPES / "positivos_tts"
    dir_eu = CLIPES / "positivos_eu"
    dir_adv = CLIPES / "negativos_adv"
    if not dir_tts.is_dir():
        print("rode primeiro: python -m treino sintetizar", file=sys.stderr)
        return 1

    pos_tts_tr, pos_tts_val = _dividir(dir_tts, semente=1)
    pos_eu_tr, pos_eu_val = _dividir(dir_eu, semente=2)
    adv_tr, adv_val = _dividir(dir_adv, semente=3)

    if not pos_eu_tr:
        print("\n  AVISO: nenhuma gravação sua em positivos_eu/.")
        print("  O modelo vai conhecer só as 4 vozes do Piper e pode não")
        print("  reconhecer a sua. Rode: python -m treino gravar\n")

    # As gravacoes proprias sao poucas (~100) contra ~360 do TTS, e sao as que
    # importam -- por isso levam mais voltas de aumentacao, equilibrando o peso.
    voltas_eu = args.voltas * 3
    FEATURES.mkdir(parents=True, exist_ok=True)

    tarefas = [
        ("positivos_treino", pos_tts_tr, args.voltas, pos_eu_tr, voltas_eu),
        ("positivos_val", pos_tts_val, args.voltas, pos_eu_val, voltas_eu),
        ("negativos_adv_treino", adv_tr, args.voltas, [], 0),
        ("negativos_adv_val", adv_val, args.voltas, [], 0),
    ]
    for nome, base, voltas_base, extra, voltas_extra in tarefas:
        destino = FEATURES / f"{nome}.npy"
        if destino.exists() and not args.refazer:
            print(f"ja existe: {destino.name} "
                  f"({np.load(destino, mmap_mode='r').shape[0]} exemplos)")
            continue
        if not base and not extra:
            print(f"pulando {nome}: sem clipes")
            continue
        print(f"{nome}: {len(base)} clipes x{voltas_base}"
              + (f" + {len(extra)} seus x{voltas_extra}" if extra else ""))

        # crc32 e nao hash(): o hash de strings no Python e randomizado por
        # processo (PYTHONHASHSEED), o que tornaria a aumentacao irreproduzivel
        # entre execucoes.
        semente = zlib.crc32(nome.encode())

        def fluxo():
            if base:
                yield from aumentar.aumentar_clipes(base, voltas_base, rirs, fundos,
                                                    semente=semente)
            if extra:
                yield from aumentar.aumentar_clipes(extra, voltas_extra, rirs, fundos,
                                                    semente=semente + 7)

        n = features.extrair(fluxo(), destino, lote=args.lote_features,
                             ncpu=args.ncpu)
        print(f"  -> {destino.name}: {n} exemplos")
    return 0


def cmd_treinar(args) -> int:
    faltando = [n for n in ["positivos_treino", "positivos_val",
                            "negativos_adv_treino", "negativos_adv_val"]
                if not (FEATURES / f"{n}.npy").exists()]
    if faltando:
        print(f"faltam features: {faltando}\nrode: python -m treino features",
              file=sys.stderr)
        return 1

    grandes = [p for p in [DADOS / "acav100m_2000h.npy"] if p.exists()]
    if not grandes:
        print("AVISO: sem os negativos do ACAV100M (17,3 GB).")
        print("O modelo vai disparar mais em fala qualquer. Para o modelo de")
        print("verdade: ./scripts/06_baixar_dados_wake.sh --tudo\n")

    fp_val = DADOS / "validation_set_features.npy"
    dados = treinar.Dados(
        positivos=FEATURES / "positivos_treino.npy",
        positivos_val=FEATURES / "positivos_val.npy",
        negativos_adv=FEATURES / "negativos_adv_treino.npy",
        negativos_adv_val=FEATURES / "negativos_adv_val.npy",
        negativos_grandes=grandes,
        fp_validacao=fp_val if fp_val.exists() else None,
    )
    cfg = treinar.Config(passos=args.passos, alvo_fp_por_hora=args.alvo_fp,
                         lote_por_classe=args.lote_treino)
    print(f"treinando ({cfg.passos} passos, alvo {cfg.alvo_fp_por_hora} fp/h):")
    rede, _ = treinar.treinar(dados, cfg)
    treinar.exportar_onnx(rede, SAIDA_ONNX)
    treinar.salvar_pesos(rede, SAIDA_ONNX.with_suffix(".pt"))
    print(f"\nmodelo salvo em {SAIDA_ONNX.relative_to(RAIZ)}")
    print(f"pesos em {SAIDA_ONNX.with_suffix('.pt').relative_to(RAIZ)} "
          "(para reexportar sem retreinar)")
    print("Ative com wake.modo: openwakeword no config.yaml, e confira o limiar")
    print("com: python -m treino avaliar")
    return 0


def cmd_avaliar(args) -> int:
    from .avaliar import avaliar

    if not SAIDA_ONNX.exists():
        print(f"modelo não encontrado: {SAIDA_ONNX}", file=sys.stderr)
        return 1
    avaliar(SAIDA_ONNX, FEATURES, DADOS, tolerancia_fp=args.tolerancia_fp)
    return 0


def cmd_tudo(args) -> int:
    for etapa in (cmd_sintetizar, cmd_features, cmd_treinar, cmd_avaliar):
        print(f"\n{'=' * 60}\n{etapa.__name__.removeprefix('cmd_')}\n{'=' * 60}")
        rc = etapa(args)
        if rc != 0:
            return rc
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        prog="treino", description=f'Treina a wake word "{PALAVRA}"')
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("sintetizar", help="clipes das vozes do Piper")
    p.add_argument("--positivos-por-voz", type=int, default=90)
    p.add_argument("--negativos-por-voz", type=int, default=240)
    p.set_defaults(func=cmd_sintetizar)

    p = sub.add_parser("gravar", help="grava amostras da sua voz")
    p.add_argument("-n", "--quantas", type=int, default=100)
    p.add_argument("--dispositivo", default=None)
    p.set_defaults(func=cmd_gravar)

    p = sub.add_parser("features", help="aumentação + extração de features")
    p.add_argument("--voltas", type=int, default=8,
                   help="variantes aumentadas por clipe (default: 8)")
    p.add_argument("--lote-features", type=int, default=128)
    p.add_argument("--ncpu", type=int, default=8)
    p.add_argument("--refazer", action="store_true")
    p.set_defaults(func=cmd_features)

    p = sub.add_parser("treinar", help="treina e exporta o .onnx")
    p.add_argument("--passos", type=int, default=12_000)
    p.add_argument("--alvo-fp", type=float, default=0.2,
                   help="falsos positivos por hora aceitáveis (default: 0.2)")
    p.add_argument("--lote-treino", type=int, default=256,
                   help="exemplos por classe em cada passo (default: 256)")
    p.set_defaults(func=cmd_treinar)

    p = sub.add_parser("avaliar", help="mede o modelo e sugere o limiar")
    p.add_argument("--tolerancia-fp", type=float, default=0.2,
                   help="falsos positivos por hora aceitáveis ao sugerir o "
                        "limiar (default: 0.2, o mesmo alvo do treino)")
    p.set_defaults(func=cmd_avaliar)

    p = sub.add_parser("tudo", help="sintetizar -> features -> treinar -> avaliar")
    p.add_argument("--positivos-por-voz", type=int, default=90)
    p.add_argument("--negativos-por-voz", type=int, default=240)
    p.add_argument("--voltas", type=int, default=8)
    p.add_argument("--lote-features", type=int, default=128)
    p.add_argument("--ncpu", type=int, default=8)
    p.add_argument("--refazer", action="store_true")
    p.add_argument("--passos", type=int, default=12_000)
    p.add_argument("--alvo-fp", type=float, default=0.2)
    p.add_argument("--lote-treino", type=int, default=256)
    p.add_argument("--tolerancia-fp", type=float, default=0.2)
    p.set_defaults(func=cmd_tudo)

    args = ap.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
