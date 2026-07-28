"""Vigia o microfone enquanto o assistente fala, para poder ser interrompido.

O gatilho da interrupcao e a **propria palavra de ativacao**, nao energia de voz.
Isso e deliberado: sem cancelamento de eco, o microfone ouve o alto-falante, e
qualquer detector baseado em energia faria o assistente se interromper na primeira
silaba que ele mesmo pronuncia. O modelo da wake word e discriminativo -- so
dispara na palavra -- entao atravessa o proprio audio do assistente sem se
confundir.

Resta um caso: se o assistente *disser* "Aristoteles" na resposta, ele se
interrompe. O `llm.prompt_sistema` pede que ele nao diga o proprio nome, e o
cooldown do gatilho evita a repeticao em cascata.
"""

from __future__ import annotations

import threading

from .audio.entrada import EntradaAudio
from .audio.saida import SaidaAudio
from .wake import Gatilho


class VigiaBargeIn:
    """Context manager: vigia durante o `with`, para ao sair.

    Use como:

        with VigiaBargeIn(gatilho, entrada, saida) as vigia:
            ...fala...
            saida.aguardar()
        if vigia.interrompido:
            ...
    """

    def __init__(self, gatilho: Gatilho, entrada: EntradaAudio,
                 saida: SaidaAudio, ativo: bool = True) -> None:
        self._gatilho = gatilho
        self._entrada = entrada
        self._saida = saida
        self._ativo = ativo and gatilho.suporta_barge_in
        self._parar = threading.Event()
        self._interrompido = threading.Event()
        self._thread: threading.Thread | None = None

    @property
    def interrompido(self) -> bool:
        return self._interrompido.is_set()

    def __enter__(self) -> "VigiaBargeIn":
        if not self._ativo:
            return self
        # Descarta o que entrou antes de comecarmos a falar: sao restos da propria
        # pergunta do usuario, e alimentar isso ao gatilho dispararia na hora.
        self._entrada.limpar()
        self._gatilho.reiniciar()
        self._thread = threading.Thread(target=self._rodar, daemon=True)
        self._thread.start()
        return self

    def __exit__(self, *_exc) -> None:
        self._parar.set()
        if self._thread is not None:
            # timeout > o timeout do ler(), senao encerramos antes da thread ver a flag
            self._thread.join(timeout=2.0)
            self._thread = None

    def _rodar(self) -> None:
        while not self._parar.is_set():
            bloco = self._entrada.ler(timeout=0.2)
            if bloco is None:
                continue
            try:
                if self._gatilho.alimentar(bloco):
                    self._interrompido.set()
                    self._saida.interromper()
                    return
            except Exception as e:  # nao derruba a fala por falha do detector
                print(f"[barge-in] detector falhou: {e}")
                return
