"""Loop principal: a maquina de estados do assistente.

    OCIOSO -> DESPERTO(beep) -> OUVINDO -> TRANSCREVENDO -> PENSANDO -> FALANDO -> OCIOSO
"""

from __future__ import annotations

import argparse
import signal
import sys
import time
from pathlib import Path

from . import stt as stt_mod
from . import wake as wake_mod
from .audio.entrada import EntradaAudio, listar_dispositivos
from .audio.saida import SaidaAudio, beep
from .barge_in import VigiaBargeIn
from .cerebro import Cerebro
from .config import Config
from .registro import Registro
from .tts import Voz
from .vad import DetectorFala, gravar_ate_silencio


def _tratar_sigterm() -> None:
    """Faz o SIGTERM se comportar como Ctrl-C.

    `systemctl stop` manda SIGTERM, cuja acao default encerra o processo sem rodar
    os `finally`. Sem isto o stream de audio nao era fechado, o resumo de blocos
    descartados nao saia, e o arquivo de log ficava sem a linha de encerramento e
    sem a ultima linha pendente. Levantar KeyboardInterrupt reaproveita o caminho
    de saida limpa que ja existe.
    """
    def encerrar(_sig, _frame):
        raise KeyboardInterrupt

    signal.signal(signal.SIGTERM, encerrar)


def main() -> int:
    ap = argparse.ArgumentParser(prog="aristoteles", description="Assistente de voz local")
    ap.add_argument("-c", "--config", type=Path, default=None, help="caminho do config.yaml")
    ap.add_argument("--listar-audio", action="store_true", help="lista dispositivos e sai")
    ap.add_argument("--texto", action="store_true", help="entrada por teclado (pula STT) -- util para depurar")
    ap.add_argument("--sem-log", action="store_true",
                    help="nao copia o console para o arquivo de log")
    args = ap.parse_args()

    if args.listar_audio:
        print(listar_dispositivos())
        return 0

    cfg = Config.carregar(args.config)
    if args.sem_log:
        cfg.log.ativo = False

    _tratar_sigterm()

    # O tee comeca aqui, o mais cedo possivel: tudo abaixo -- inclusive falhas de
    # inicializacao e tracebacks -- vai para o arquivo junto com o console.
    with Registro(cfg.log, cfg.raiz) as registro:
        try:
            return _executar(args, cfg, registro)
        except KeyboardInterrupt:
            # Rede de seguranca: se o sinal chegar durante a carga dos modelos,
            # antes do try do loop, ainda saimos sem traceback -- e com o
            # `__exit__` do Registro rodando, que e o que fecha o log.
            print("\nEncerrando.")
            return 130


def _executar(args, cfg: Config, registro: Registro) -> int:
    print("Aristoteles iniciando...")
    print(f"  STT.....: {cfg.stt.backend} ({cfg.stt.modelo_cpu if cfg.stt.backend == 'cpu' else cfg.stt.servidor_url})")
    print(f"  LLM.....: {cfg.llm.modelo} (effort={cfg.llm.effort}, thinking={'on' if cfg.llm.pensar else 'off'})")
    print(f"  TTS.....: {Path(cfg.tts.voz).name}")
    print(f"  Gatilho.: {cfg.wake.modo}")
    print(f"  Log.....: {registro.caminho if cfg.log.ativo else 'desligado'}")

    try:
        voz = Voz(cfg.tts, cfg.raiz)
        cerebro = Cerebro(cfg.llm)
        gatilho = wake_mod.criar(cfg.wake, cfg.raiz)
        transcritor = stt_mod.criar(cfg.stt, cfg.audio.taxa_amostragem)
    except Exception as e:
        print(f"\nFalha na inicializacao: {e}", file=sys.stderr)
        return 1

    if not args.texto:
        print("  carregando modelo de STT...", end="", flush=True)
        t0 = time.perf_counter()
        try:
            transcritor.aquecer()
        except Exception as e:
            print(f"\nFalha no STT: {e}", file=sys.stderr)
            return 1
        print(f" ok ({time.perf_counter() - t0:.1f}s)")

    detector = DetectorFala(cfg.vad, cfg.audio)
    saida = SaidaAudio(cfg.audio, voz.taxa_amostragem)
    # Construir fora do try: o `with` abre o stream, e se isso falhar o nome ainda
    # precisa existir para o finally la embaixo.
    captura = EntradaAudio(cfg.audio)

    try:
        with captura as entrada:
            def calibrar(motivo: str = "") -> None:
                rotulo = f"recalibrando ({motivo})" if motivo else "calibrando piso de ruido"
                print(f"  {rotulo} (fique em silencio)...", end="", flush=True)
                limiar = detector.calibrar(entrada)
                print(f" piso={detector.piso_ruido:.4f} limiar={limiar:.4f}")
                if detector.limiar_no_teto:
                    print(f"  AVISO: ambiente ruidoso -- limiar preso no teto de "
                          f"{cfg.vad.limiar_maximo}. Baixe o ganho do microfone;")
                    print("  sem isso o gate deixa passar ruido junto com a fala.")
                elif detector.piso_ruido > 0.05:
                    print("  AVISO: ambiente ruidoso. Considere baixar o ganho do microfone.")

            if not args.texto:
                calibrar()
            def ouvir() -> str | None:
                """Bipe, grava ate o silencio, transcreve. None = nao deu."""
                beep(880, dispositivo=cfg.audio.dispositivo_saida)
                print("  [ouvindo]", end="", flush=True)
                audio = gravar_ate_silencio(entrada, detector, cfg.vad,
                                            gatilho.usa_pre_roll,
                                            absorver_gatilho=gatilho.usa_pre_roll)
                beep(660, 0.08, dispositivo=cfg.audio.dispositivo_saida)
                if audio is None:
                    print(f"\r  [nao ouvi nada: {detector.ultimo_motivo}]")
                    return None
                t0 = time.perf_counter()
                dito = transcritor.transcrever(audio)
                if not dito:
                    print("\r  [nao entendi]        ")
                    return None
                print(f"\r  voce ({time.perf_counter() - t0:.1f}s): {dito}")
                return dito

            print("\nPronto.\n")
            # Preenchido quando um barge-in ja capturou a proxima pergunta: nesse
            # caso nao voltamos ao ocioso, porque o usuario acabou de falar.
            pendente: str | None = None
            # "Ninguem falou" seguidas vezes quase sempre significa gate errado, e
            # nao usuario mudo: sob systemd um piso medido num instante ruidoso
            # trancou a fala por 6 minutos. Recalibrar cura sem reiniciar.
            falhas_seguidas = 0
            while True:
                # --- OCIOSO -> DESPERTO -> OUVINDO ---
                if pendente is not None:
                    pergunta, pendente = pendente, None
                elif args.texto:
                    # NAO passa pelo gatilho: `--texto` existe para depurar sem
                    # microfone, e com wake.modo: openwakeword o `aguardar()`
                    # ficaria esperando a palavra falada -- o oposto do proposito.
                    try:
                        pergunta = input("voce > ").strip()
                    except EOFError:
                        break
                    if pergunta.lower() in ("sair", "q", "quit", "exit"):
                        break
                    if not pergunta:
                        continue
                else:
                    if not gatilho.aguardar(entrada):
                        break
                    if (pergunta := ouvir()) is None:
                        if "ninguem falou" in detector.ultimo_motivo:
                            falhas_seguidas += 1
                            if falhas_seguidas >= cfg.vad.recalibrar_apos_falhas:
                                calibrar(f"{falhas_seguidas} tentativas sem fala")
                                falhas_seguidas = 0
                        else:
                            falhas_seguidas = 0
                        continue
                    falhas_seguidas = 0

                # --- PENSANDO -> FALANDO ---
                saida.retomar()
                t0 = time.perf_counter()
                primeira = True
                print("  aristoteles: ", end="", flush=True)
                # Vigia a palavra de ativacao durante a fala: dize-la interrompe.
                with VigiaBargeIn(gatilho, entrada, saida,
                                  ativo=not args.texto) as vigia:
                    for frase in cerebro.responder(pergunta):
                        if vigia.interrompido:
                            break
                        if primeira:
                            print(f"[{time.perf_counter() - t0:.1f}s] ", end="", flush=True)
                            primeira = False
                        print(frase, end=" ", flush=True)
                        for bloco in voz.sintetizar(frase):
                            saida.enfileirar(bloco)
                    saida.aguardar()
                print()

                if vigia.interrompido:
                    print("  [interrompido]")
                    pendente = ouvir()  # ja ouviu a palavra; nao volta ao ocioso
                else:
                    entrada.limpar()  # nao escutar a propria resposta

    except KeyboardInterrupt:
        print("\nEncerrando.")
    finally:
        saida.fechar()
        if captura.descartados:
            # Esperado: e o audio capturado enquanto ninguem estava gravando.
            print(f"  ({captura.descartados} blocos de captura descartados no ocioso)")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
