"""Deteccao de fala (VAD) e endpointing: saber quando o usuario parou de falar.

Duas camadas, porque uma nao basta:

1. **Gate de energia auto-calibrado.** O webrtcvad classifica ruido estacionario
   de banda larga (ventoinha, ar-condicionado, ganho alto de mic USB) como fala --
   medido nesta maquina: 98/100 blocos de ambiente viravam "fala" mesmo na
   agressividade 3. Sem o gate, o endpointing nunca ve silencio e grava ate o
   timeout. O piso de ruido e medido na inicializacao, entao adapta a cada sala.
2. **webrtcvad.** Bom para distinguir fala de outros sons *acima* do piso.

Um bloco so conta como fala se passar nas duas.
"""

from __future__ import annotations

import numpy as np
import webrtcvad

from .audio.entrada import EntradaAudio
from .config import AudioCfg, VadCfg


def rms(bloco: np.ndarray) -> float:
    """Valor eficaz do bloco int16, normalizado para [0, 1]."""
    x = bloco.astype(np.float32) / 32768.0
    return float(np.sqrt(np.mean(x * x)))


class DetectorFala:
    def __init__(self, cfg: VadCfg, audio: AudioCfg) -> None:
        self._vad = webrtcvad.Vad(cfg.agressividade)
        self.cfg = cfg
        self.audio = audio
        self.piso_ruido: float = 0.0
        self.limiar: float = cfg.piso_minimo
        # True quando o limiar bateu no teto -- sinal de que o ambiente estava
        # ruidoso na calibracao e o gate pode estar mais permissivo que o ideal.
        self.limiar_no_teto: bool = False
        # Por que a ultima gravacao devolveu None. Diagnosticar "a janela fechou
        # rapido" sem isto custou duas rodadas de investigacao.
        self.ultimo_motivo: str = ""

    def calibrar(self, entrada: EntradaAudio) -> float:
        """Mede o piso de ruido do ambiente e define o limiar do gate.

        Chame na inicializacao, com o usuario em silencio.

        Tres cuidados, todos vindos de falhas reais:

        * **Descarta os primeiros blocos.** Medido, a abertura do stream entrega
          lixo: 0,0 no primeiro bloco e um pico de ~2x o ambiente no segundo.
        * **Mediana, nao percentil 90.** O p90 e *mais* sujeito a um estalo isolado
          que a mediana -- o comentario anterior afirmava o contrario. E o fator
          `fator_acima_do_piso` ja fornece a margem de seguranca; usar p90 embaixo
          dele contava a margem duas vezes.
        * **Teto no limiar.** Sem ele, ruido durante os 600 ms trancava a fala pela
          sessao inteira. Medido sob systemd: piso=0,1167 gerou limiar 0,35 e
          nenhuma pergunta passou o gate por 6 minutos.
        """
        n = max(4, int(self.cfg.calibracao_ms / self.audio.bloco_ms))
        entrada.limpar()
        niveis = []
        for i in range(n + self.cfg.descartar_aquecimento):
            bloco = entrada.ler(timeout=1.0)
            if bloco is None:
                break
            if i < self.cfg.descartar_aquecimento:
                continue
            niveis.append(rms(bloco))

        if not niveis:
            self.piso_ruido = 0.0
            self.limiar = self.cfg.piso_minimo
            return self.limiar

        self.piso_ruido = float(np.median(niveis))
        bruto = self.piso_ruido * self.cfg.fator_acima_do_piso
        self.limiar = min(max(bruto, self.cfg.piso_minimo), self.cfg.limiar_maximo)
        self.limiar_no_teto = bruto > self.cfg.limiar_maximo
        return self.limiar

    def eh_fala(self, bloco: np.ndarray) -> bool:
        if rms(bloco) < self.limiar:
            return False
        return self._vad.is_speech(bloco.tobytes(), self.audio.taxa_amostragem)


def gravar_ate_silencio(
    entrada: EntradaAudio,
    detector: DetectorFala,
    cfg: VadCfg,
    usar_pre_roll: bool = True,
    absorver_gatilho: bool = False,
) -> np.ndarray | None:
    """Grava de agora ate detectar `silencio_final_ms` de silencio.

    Retorna float32 mono em [-1, 1], ou None se nao houve fala suficiente.

    `absorver_gatilho` conserta um bug que fazia a janela de escuta parecer
    instantanea no modo wake word. Quando o gatilho dispara, a cauda da propria
    palavra "Aristoteles" ainda esta na fila; sem tratamento ela era contada como
    fala do usuario, o que **armava o endpointing**. Dali em diante valia o
    `silencio_final_ms` (700 ms) em vez do `espera_inicial_s` (segundos), e quem
    pausasse para formular a pergunta perdia a vez -- voltando ao ocioso "num
    instante".

    Com a flag ligada, a primeira sequencia de fala e absorvida como sendo o
    gatilho e nao arma o endpointing. A absorcao tem teto (`absorver_max_ms`)
    para nao engolir a pergunta de quem diz tudo numa tirada so, sem pausa:
    "Aristoteles, que horas sao?".
    """
    bloco_ms = entrada.cfg.bloco_ms
    max_silencio = max(1, cfg.silencio_final_ms // bloco_ms)
    min_fala = max(1, cfg.fala_minima_ms // bloco_ms)
    max_blocos = int(cfg.duracao_maxima_s * 1000 / bloco_ms)
    # Quanto esperar pela fala *comecar* antes de desistir.
    max_espera = max(1, int(cfg.espera_inicial_s * 1000 / bloco_ms))
    max_absorcao = max(1, int(cfg.absorver_max_ms / bloco_ms))

    blocos: list[np.ndarray] = list(entrada.pre_roll()) if usar_pre_roll else []
    silencio_seguido = 0
    blocos_de_fala = 0
    houve_fala = False
    espera = 0          # blocos de silencio desde que comecamos a esperar o usuario
    absorvidos = 0      # blocos da palavra de ativacao ja descartados do endpointing
    absorvendo = absorver_gatilho
    detector.ultimo_motivo = ""

    for _ in range(max_blocos):
        bloco = entrada.ler(timeout=1.0)
        if bloco is None:
            detector.ultimo_motivo = "captura parou (microfone sumiu?)"
            break
        blocos.append(bloco)
        fala = detector.eh_fala(bloco)

        if absorvendo:
            if fala and absorvidos < max_absorcao:
                absorvidos += 1
                continue  # ainda e a palavra de ativacao
            # Silencio, ou a fala passou do teto: o que vier agora e do usuario.
            # O audio absorvido fica em `blocos` -- Whisper precisa dele para o
            # caso da tirada unica.
            absorvendo = False
            if not fala:
                espera = 1
                continue

        if fala:
            blocos_de_fala += 1
            silencio_seguido = 0
            espera = 0  # fez algum som: recomeca a contagem da paciencia
            # So arma o endpointing depois de `fala_minima_ms` acumulados. Um unico
            # bloco nao basta: medido, um estalo de 90 ms (tosse, cadeira, teclado)
            # ligava `houve_fala` e a janela caia dos 6 s para 1,4 s. E o mesmo bug
            # da cauda do gatilho, mas vindo do ambiente.
            if blocos_de_fala >= min_fala:
                houve_fala = True
        else:
            silencio_seguido += 1
            if houve_fala and silencio_seguido >= max_silencio:
                break
            # Nada foi dito ainda: nao segura o usuario por duracao_maxima_s.
            if not houve_fala:
                espera += 1
                if espera >= max_espera:
                    detector.ultimo_motivo = (
                        f"ninguem falou em {cfg.espera_inicial_s:.0f}s "
                        f"(vad.espera_inicial_s)")
                    return None
    else:
        detector.ultimo_motivo = f"atingiu duracao_maxima_s={cfg.duracao_maxima_s:.0f}s"

    # Rede de seguranca. Desde que armar o endpointing passou a exigir `min_fala`,
    # sair pelo endpoint implica ter fala suficiente -- entao este ramo so e
    # alcancavel por parada da captura ou pelo teto de duracao, que ja registraram
    # o proprio motivo. Fica pelo custo zero, e o `if` preserva o mais especifico.
    if blocos_de_fala < min_fala:
        if not detector.ultimo_motivo:
            detector.ultimo_motivo = (
                f"fala de {blocos_de_fala * bloco_ms}ms, minimo "
                f"{cfg.fala_minima_ms}ms (vad.fala_minima_ms)")
        return None

    return np.concatenate(blocos).astype(np.float32) / 32768.0
