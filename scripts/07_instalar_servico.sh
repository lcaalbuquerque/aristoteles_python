#!/usr/bin/env bash
# Instala o Aristoteles como servico de usuario do systemd (fase 6).
#
#   ./scripts/07_instalar_servico.sh          # instala e valida, sem habilitar
#   ./scripts/07_instalar_servico.sh --agora  # ... e ja inicia
#
# Servico de USUARIO, nao de sistema: precisa da sessao de audio (PipeWire) do
# seu login. Um servico de sistema nao alcanca o microfone nem o alto-falante.
set -euo pipefail

RAIZ="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
UNIT_DIR="$HOME/.config/systemd/user"
ENV_DIR="$HOME/.config/aristoteles"
ENV_FILE="$ENV_DIR/env"

# --- 1. pre-requisitos que fazem o servico falhar em silencio se faltarem ------

modo=$(sed -nE 's/^[[:space:]]*modo:[[:space:]]*([a-z_]+).*/\1/p' "$RAIZ/config.yaml" | tail -1)
if [[ "$modo" != "openwakeword" ]]; then
    echo "ERRO: wake.modo esta '$modo' no config.yaml." >&2
    echo "  O servico roda sem terminal, e push_to_talk espera um Enter que" >&2
    echo "  nunca vem. Ponha 'modo: openwakeword'." >&2
    exit 1
fi

if [[ ! -f "$RAIZ/modelos/wake/aristoteles.onnx" ]]; then
    echo "ERRO: modelos/wake/aristoteles.onnx nao existe." >&2
    echo "  Treine a wake word primeiro (README, fase 5)." >&2
    exit 1
fi

# --- 2. arquivo de ambiente com a credencial ----------------------------------

mkdir -p "$ENV_DIR"
chmod 700 "$ENV_DIR"

if [[ -f "$ENV_FILE" ]]; then
    echo "ja existe: $ENV_FILE (nao vou sobrescrever)"
else
    chave="${ANTHROPIC_API_KEY:-}"
    if [[ -z "$chave" && -f "$HOME/.anthropic_api_key" ]]; then
        chave="$(cat "$HOME/.anthropic_api_key")"
    fi
    if [[ -z "$chave" ]]; then
        echo "ERRO: nao achei a credencial." >&2
        echo "  export ANTHROPIC_API_KEY=... e rode de novo, ou crie na mao:" >&2
        echo "    printf 'ANTHROPIC_API_KEY=%s\\n' 'sk-ant-...' > $ENV_FILE" >&2
        exit 1
    fi
    # `printf %s` com a expansao ja aparada: um \n no meio do valor viraria
    # LocalProtocolError na primeira pergunta, e o assistente culparia a rede.
    # O systemd nao apara o valor por voce.
    printf 'ANTHROPIC_API_KEY=%s\n' "$(printf %s "$chave" | tr -d '[:space:]')" > "$ENV_FILE"
    chmod 600 "$ENV_FILE"
    echo "criado: $ENV_FILE (modo 600)"
fi

# --- 3. o unit ----------------------------------------------------------------

mkdir -p "$UNIT_DIR"
install -m 644 "$RAIZ/systemd/aristoteles.service" "$UNIT_DIR/aristoteles.service"
echo "instalado: $UNIT_DIR/aristoteles.service"

systemctl --user daemon-reload
if ! systemd-analyze --user verify "$UNIT_DIR/aristoteles.service" 2>&1 | grep -q .; then
    echo "unit valido."
else
    echo "AVISO: systemd-analyze reclamou:" >&2
    systemd-analyze --user verify "$UNIT_DIR/aristoteles.service" >&2 || true
fi

# --- 4. habilitar (opcional) --------------------------------------------------

if [[ "${1:-}" == "--agora" ]]; then
    systemctl --user enable --now aristoteles
    echo
    echo "iniciado e habilitado no boot. Acompanhe:"
    echo "  journalctl --user -u aristoteles -f"
else
    echo
    echo "Instalado, mas NAO iniciado. Para ligar agora e no boot:"
    echo "  systemctl --user enable --now aristoteles"
    echo "Para so testar sem persistir:"
    echo "  systemctl --user start aristoteles && journalctl --user -u aristoteles -f"
    echo "Para desfazer:"
    echo "  systemctl --user disable --now aristoteles"
fi

echo
echo "Lembre: se trocar a chave da API, atualize $ENV_FILE e rode"
echo "  systemctl --user restart aristoteles"
