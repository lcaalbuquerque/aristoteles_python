"""Gravador das suas proprias amostras da palavra de ativacao.

Num assistente de um unico usuario, generalizar entre falantes vale pouco e
acertar *a sua* voz vale tudo -- essas gravacoes sao o dado de maior valor do
treino, e sao ~10 minutos de trabalho.

Fluxo por amostra: bipe curto, grava uma janela fixa, apara o silencio, valida
que tem fala e salva. Amostras vazias ou estouradas sao recusadas na hora, para
voce nao descobrir no fim que metade nao servia.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

from .comum import TAXA, aparar_silencio, escrever_wav, rms

JANELA_S = 2.0       # tempo de gravacao por amostra
MIN_DURACAO_S = 0.25  # menos que isso nao e a palavra
MAX_DURACAO_S = 1.8   # mais que isso e frase, nao a palavra sozinha
PICO_MAXIMO = 32_000  # perto de 32767 = clipping


def _bipe(freq: int = 880, dur: float = 0.08) -> None:
    import sounddevice as sd

    t = np.linspace(0, dur, int(TAXA * dur), endpoint=False)
    onda = (0.2 * np.sin(2 * np.pi * freq * t) * 32767).astype(np.int16)
    try:
        sd.play(onda, TAXA, blocking=True)
    except Exception:
        pass  # sem saida de audio nao e motivo para abortar a gravacao


def _avaliar(audio: np.ndarray, piso: float) -> str | None:
    """Devolve o motivo da recusa, ou None se a amostra serve."""
    if audio.size < MIN_DURACAO_S * TAXA:
        return "curta demais (falou?)"
    if audio.size > MAX_DURACAO_S * TAXA:
        return "longa demais (fale só a palavra)"
    if abs(audio).max() >= PICO_MAXIMO:
        return "estourada (baixe o ganho do mic)"
    if rms(audio) < 3 * piso:
        return "fraca demais (mais perto do mic)"
    return None


def _medir_piso(dur_s: float = 0.6) -> float:
    import sounddevice as sd

    print("  medindo o ruido de fundo (silêncio, por favor)...", end="", flush=True)
    bloco = sd.rec(int(dur_s * TAXA), samplerate=TAXA, channels=1, dtype="int16",
                   blocking=True)[:, 0]
    piso = rms(bloco)
    print(f" piso={piso:.0f}")
    return max(piso, 20.0)  # evita piso zero num mic mudo


def gravar(destino: Path, quantas: int = 100, dispositivo=None) -> int:
    """Grava `quantas` amostras validas em `destino`. Devolve quantas salvou.

    Retoma de onde parou: conta os wav que ja existem.
    """
    import sounddevice as sd

    if dispositivo is not None:
        sd.default.device = dispositivo

    destino.mkdir(parents=True, exist_ok=True)
    ja = len(list(destino.glob("*.wav")))
    if ja >= quantas:
        print(f"já existem {ja} amostras em {destino}; nada a fazer.")
        return ja

    print(f"\nVou gravar {quantas - ja} amostras de \"Aristóteles\" ({ja} já feitas).")
    print("Diga a palavra logo depois do bipe, do jeito que você chamaria o")
    print("assistente. Varie: mais rápido, mais devagar, mais longe, mais perto.")
    print("Ctrl-C para parar (o que já gravou fica salvo).\n")
    piso = _medir_piso()

    n = ja
    recusadas = 0
    try:
        while n < quantas:
            _bipe()
            bruto = sd.rec(int(JANELA_S * TAXA), samplerate=TAXA, channels=1,
                           dtype="int16", blocking=True)[:, 0]
            audio = aparar_silencio(bruto)
            motivo = _avaliar(audio, piso)
            if motivo:
                recusadas += 1
                print(f"  [{n}/{quantas}] recusada: {motivo}")
                continue
            escrever_wav(destino / f"eu_{n:04d}.wav", audio)
            n += 1
            print(f"  [{n}/{quantas}] ok ({audio.size / TAXA:.2f}s)")
    except KeyboardInterrupt:
        print("\ninterrompido.")

    print(f"\n{n} amostras em {destino} ({recusadas} recusadas).")
    return n


def main(argv: list[str] | None = None) -> int:
    import argparse

    ap = argparse.ArgumentParser(prog="treino.gravar",
                                 description="Grava amostras da wake word")
    ap.add_argument("-n", "--quantas", type=int, default=100)
    ap.add_argument("-d", "--destino", type=Path,
                    default=Path("dados_wake/positivos_eu"))
    ap.add_argument("--dispositivo", default=None,
                    help="índice ou nome do mic (veja: python -m aristoteles --listar-audio)")
    args = ap.parse_args(argv)
    gravar(args.destino, args.quantas, args.dispositivo)
    return 0


if __name__ == "__main__":
    sys.exit(main())
