"""Quebra o texto que chega em streaming do Claude em frases completas.

E o que permite comecar a falar antes de a resposta terminar -- o maior ganho
de latencia percebida do projeto inteiro.
"""

from __future__ import annotations

import re

TERMINADORES = ".!?…"

# Nao quebrar depois destas abreviacoes nem em numeros decimais (3.14).
_ABREVIACOES = {
    "sr", "sra", "srta", "dr", "dra", "prof", "profa", "eng", "av", "r",
    "etc", "ex", "obs", "pag", "num", "tel", "cel", "ltda",
}

_FIM_DE_FRASE = re.compile(rf"[{re.escape(TERMINADORES)}]+[\"')\]]*(?=\s|$)")

# Tag XML/HTML isolada: <thinking>, </thinking>, <system-reminder foo="bar">.
# Exige que o nome comece com letra para nao comer "3 < 5 e 7 > 2".
_TAG = re.compile(r"</?[A-Za-z][\w:-]*(?:\s[^<>]*)?/?>")


def sem_tags(frase: str) -> str:
    """Remove tags XML de uma frase antes de ela virar audio.

    Com `llm.pensar: false` o Opus 5 as vezes escreve `<thinking>` (ou outro XML
    interno) no texto *visivel* em vez de emitir o bloco proprio. Sem este filtro
    o Piper pronuncia "menor que thinking maior que" em voz alta. O prompt de
    sistema tambem proibe tags -- isto e a rede de seguranca, porque o prompt nao
    e garantia.

    >>> sem_tags("<thinking>Vou calcular.</thinking> Sao quinze graus.")
    'Vou calcular. Sao quinze graus.'
    >>> sem_tags("Sete e maior que 3 > 2.")
    'Sete e maior que 3 > 2.'
    """
    if "<" not in frase:
        return frase
    return " ".join(_TAG.sub(" ", frase).split())


class AcumuladorDeFrases:
    """Recebe pedacos de texto e devolve frases completas assim que fecham.

    >>> a = AcumuladorDeFrases()
    >>> a.alimentar("Oi. Tudo ")
    ['Oi.']
    >>> a.alimentar("bem?")
    ['Tudo bem?']
    >>> a.finalizar()
    []

    Sem pontuacao final a frase espera no buffer ate o `finalizar()`:

    >>> b = AcumuladorDeFrases()
    >>> b.alimentar("Sao dez")
    []
    >>> b.finalizar()
    ['Sao dez']
    """

    def __init__(self, max_chars: int = 220) -> None:
        self._buffer = ""
        self._max_chars = max_chars

    def alimentar(self, pedaco: str) -> list[str]:
        self._buffer += pedaco
        return self._extrair()

    def finalizar(self) -> list[str]:
        frases = self._extrair()
        resto = self._buffer.strip()
        self._buffer = ""
        if resto:
            frases.append(resto)
        return frases

    def _extrair(self) -> list[str]:
        frases: list[str] = []
        while True:
            corte = self._proximo_corte()
            if corte is None:
                break
            frase = self._buffer[:corte].strip()
            self._buffer = self._buffer[corte:].lstrip()
            if frase:
                frases.append(frase)
        return frases

    def _proximo_corte(self) -> int | None:
        for m in _FIM_DE_FRASE.finditer(self._buffer):
            if not self._e_falso_fim(m.start()):
                return m.end()

        # Frase longa demais sem pontuacao final: corta na virgula para nao
        # segurar o audio indefinidamente.
        if len(self._buffer) > self._max_chars:
            virgula = self._buffer.rfind(",", 0, self._max_chars)
            if virgula > self._max_chars // 3:
                return virgula + 1
        return None

    def _e_falso_fim(self, pos: int) -> bool:
        if self._buffer[pos] != ".":
            return False
        # numero decimal: 3.14
        if pos + 1 < len(self._buffer) and self._buffer[pos + 1].isdigit():
            return True
        anterior = re.search(r"(\w+)$", self._buffer[:pos])
        if anterior and anterior.group(1).lower() in _ABREVIACOES:
            return True
        # inicial isolada: "J. Silva"
        return bool(anterior and len(anterior.group(1)) == 1 and anterior.group(1).isalpha())
