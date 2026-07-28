"""O tee do console para arquivo.

Testa o que de fato pode dar errado: perder linha, quebrar bibliotecas que
inspecionam o fluxo, embaralhar entre threads, e nao restaurar `sys.stdout` na
saida -- o pior deles, porque contamina todo o resto do processo.
"""

from __future__ import annotations

import sys
import threading

import pytest

from aristoteles.config import LogCfg
from aristoteles.registro import Registro


def _ler(caminho) -> str:
    return caminho.read_text(encoding="utf-8")


def _cfg(tmp_path, **kw) -> LogCfg:
    return LogCfg(arquivo=tmp_path / "console.log", **kw)


def test_grava_o_que_foi_impresso(tmp_path):
    cfg = _cfg(tmp_path)
    with Registro(cfg, tmp_path) as r:
        print("primeira linha")
        print("segunda linha")
    texto = _ler(r.caminho)
    assert "primeira linha" in texto
    assert "segunda linha" in texto


def test_console_continua_recebendo(tmp_path, capsys):
    """O tee copia; nao desvia. Quebrar isso deixaria o terminal mudo."""
    with Registro(_cfg(tmp_path), tmp_path):
        print("visivel no terminal")
    assert "visivel no terminal" in capsys.readouterr().out


def test_stderr_tambem_e_gravado(tmp_path):
    """Tracebacks e o `[audio]` do PortAudio saem por aqui."""
    with Registro(_cfg(tmp_path), tmp_path) as r:
        print("erro de teste", file=sys.stderr)
    texto = _ler(r.caminho)
    assert "erro de teste" in texto
    assert "ERR" in texto  # marcado como stderr


def test_traceback_de_excecao_nao_tratada_e_gravado(tmp_path):
    with Registro(_cfg(tmp_path), tmp_path) as r:
        try:
            raise ValueError("estourou")
        except ValueError:
            import traceback
            traceback.print_exc()
    texto = _ler(r.caminho)
    assert "ValueError: estourou" in texto


def test_linha_sem_newline_nao_e_perdida(tmp_path):
    """`print(..., end="")` e comum no app -- a linha de status usa isso."""
    with Registro(_cfg(tmp_path), tmp_path) as r:
        print("  [ouvindo]", end="", flush=True)
    assert "[ouvindo]" in _ler(r.caminho)


def test_retorno_de_carro_vira_linha_propria(tmp_path):
    """No arquivo, "[ouvindo]\\rvoce: oi" numa linha so seria ilegivel."""
    with Registro(_cfg(tmp_path, expandir_retorno=True), tmp_path) as r:
        print("  [ouvindo]", end="", flush=True)
        print("\r  voce: que horas sao")
    linhas = [l for l in _ler(r.caminho).splitlines() if "ouvindo" in l or "voce" in l]
    assert len(linhas) == 2


def test_sem_expandir_retorno_preserva_o_fluxo(tmp_path):
    with Registro(_cfg(tmp_path, expandir_retorno=False), tmp_path) as r:
        print("  [ouvindo]", end="", flush=True)
        print("\r  voce: oi")
    # bytes, nao read_text(): a leitura com newlines universais converteria o \r
    assert b"\r" in r.caminho.read_bytes()


def test_restaura_os_fluxos_na_saida(tmp_path):
    """O mais importante: nao vazar o wrapper para o resto do processo."""
    antes = (sys.stdout, sys.stderr)
    with Registro(_cfg(tmp_path), tmp_path):
        assert sys.stdout is not antes[0]
    assert (sys.stdout, sys.stderr) == antes


def test_restaura_os_fluxos_mesmo_com_excecao(tmp_path):
    antes = (sys.stdout, sys.stderr)
    with pytest.raises(RuntimeError):
        with Registro(_cfg(tmp_path), tmp_path):
            raise RuntimeError("falha no meio")
    assert (sys.stdout, sys.stderr) == antes


def test_desligado_nao_cria_arquivo(tmp_path):
    cfg = _cfg(tmp_path, ativo=False)
    antes = sys.stdout
    with Registro(cfg, tmp_path) as r:
        print("nada disso vai para arquivo")
        assert sys.stdout is antes  # nem embrulha
    assert not r.caminho.exists()


def test_cria_o_diretorio_do_log(tmp_path):
    cfg = LogCfg(arquivo=tmp_path / "fundo" / "do" / "poco" / "c.log")
    with Registro(cfg, tmp_path) as r:
        print("oi")
    assert r.caminho.exists()


def test_caminho_relativo_resolve_contra_a_raiz(tmp_path):
    cfg = LogCfg(arquivo="logs/app.log")
    r = Registro(cfg, tmp_path)
    assert r.caminho == tmp_path / "logs" / "app.log"


def test_caminho_absoluto_e_respeitado(tmp_path):
    alvo = tmp_path / "outro" / "app.log"
    assert Registro(LogCfg(arquivo=alvo), tmp_path / "raiz").caminho == alvo


def test_isatty_nao_mente(tmp_path, capsys):
    """Quem desenha barra de progresso decide por isatty(); mentir aqui mudaria o
    comportamento do proprio programa que estamos observando."""
    with Registro(_cfg(tmp_path), tmp_path):
        assert sys.stdout.isatty() == capsys._capture.out.tmpfile.isatty()


def test_fileno_delegado(tmp_path):
    """Subprocessos herdam pelo fileno; sem delegar, quebraria."""
    with Registro(_cfg(tmp_path), tmp_path):
        assert isinstance(sys.stdout.fileno(), int)


def test_marca_a_sessao(tmp_path):
    with Registro(_cfg(tmp_path), tmp_path) as r:
        print("meio da sessao")
    texto = _ler(r.caminho)
    assert "sessao iniciada" in texto
    assert "sessao encerrada" in texto


def test_sessoes_acumulam_no_mesmo_arquivo(tmp_path):
    cfg = _cfg(tmp_path)
    for i in range(3):
        with Registro(cfg, tmp_path) as r:
            print(f"sessao numero {i}")
    texto = _ler(r.caminho)
    assert texto.count("sessao iniciada") == 3
    assert "sessao numero 2" in texto


def test_rodizio_limita_o_tamanho(tmp_path):
    """Como servico o app fica dias ligado; sem rodizio o arquivo cresce sem fim."""
    cfg = _cfg(tmp_path, max_bytes=2_000, backups=2)
    with Registro(cfg, tmp_path) as r:
        for i in range(400):
            print(f"linha de enchimento numero {i:04d} com texto suficiente")
    assert r.caminho.stat().st_size <= 4_000
    assert (tmp_path / "console.log.1").exists()


def test_threads_nao_embaralham_linhas(tmp_path):
    """As threads de barge-in e de reproducao tambem imprimem."""
    with Registro(_cfg(tmp_path), tmp_path) as r:
        def trabalhar(n):
            for i in range(20):
                print(f"thread-{n}-linha-{i:02d}")

        ts = [threading.Thread(target=trabalhar, args=(n,)) for n in range(4)]
        for t in ts:
            t.start()
        for t in ts:
            t.join()

    texto = _ler(r.caminho)
    for n in range(4):
        for i in range(20):
            assert f"thread-{n}-linha-{i:02d}" in texto, f"perdeu thread-{n}-linha-{i:02d}"


def test_sigterm_vira_keyboardinterrupt():
    """`systemctl stop` manda SIGTERM, cuja acao default encerra sem rodar os
    `finally` -- o log ficava sem a linha de encerramento e sem a ultima linha
    pendente, e o stream de audio nao era fechado."""
    import os
    import signal

    from aristoteles.__main__ import _tratar_sigterm

    anterior = signal.getsignal(signal.SIGTERM)
    try:
        _tratar_sigterm()
        with pytest.raises(KeyboardInterrupt):
            os.kill(os.getpid(), signal.SIGTERM)
    finally:
        signal.signal(signal.SIGTERM, anterior)


def test_cada_linha_tem_carimbo_de_tempo(tmp_path):
    import re
    with Registro(_cfg(tmp_path), tmp_path) as r:
        print("com data")
    linha = [l for l in _ler(r.caminho).splitlines() if "com data" in l][0]
    assert re.match(r"^\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2} out ", linha), linha
