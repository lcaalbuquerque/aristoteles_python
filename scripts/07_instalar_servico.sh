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

# NAO copiamos a chave para ca. O app le ~/.anthropic_api_key quando
# ANTHROPIC_API_KEY nao esta no ambiente, e uma copia significa divergencia: numa
# rotacao o original foi atualizado, a copia ficou com a revogada, e o app
# funcionava no terminal e devolvia 401 so como servico.
ARQ_CHAVE="$HOME/.anthropic_api_key"

if [[ ! -s "$ARQ_CHAVE" ]]; then
    if [[ -n "${ANTHROPIC_API_KEY:-}" ]]; then
        printf %s "$(printf %s "$ANTHROPIC_API_KEY" | tr -d '[:space:]')" > "$ARQ_CHAVE"
        chmod 600 "$ARQ_CHAVE"
        echo "criado: $ARQ_CHAVE a partir do ambiente (modo 600)"
    else
        echo "ERRO: nao achei a credencial." >&2
        echo "  O servico nao herda o seu ambiente. Grave a chave em $ARQ_CHAVE:" >&2
        echo "    printf %s 'sk-ant-...' > $ARQ_CHAVE && chmod 600 $ARQ_CHAVE" >&2
        exit 1
    fi
else
    echo "credencial: $ARQ_CHAVE ($(stat -c%a "$ARQ_CHAVE"))"
    if [[ "$(stat -c%a "$ARQ_CHAVE")" != "600" ]]; then
        echo "  AVISO: permissao frouxa; corrija com chmod 600 $ARQ_CHAVE" >&2
    fi
fi

# Se um env file antigo ainda carrega a chave, ele VENCE (o systemd o injeta no
# ambiente, e o ambiente tem prioridade). Avisa quando divergir do original.
if [[ -f "$ENV_FILE" ]] && grep -q '^ANTHROPIC_API_KEY=' "$ENV_FILE"; then
    a=$(sed -nE 's/^ANTHROPIC_API_KEY=//p' "$ENV_FILE" | tr -d '[:space:]')
    b=$(tr -d '[:space:]' < "$ARQ_CHAVE")
    if [[ "$a" != "$b" ]]; then
        echo "  AVISO: $ENV_FILE tem uma chave DIFERENTE de $ARQ_CHAVE." >&2
        echo "  Ela tem prioridade e vai ser usada pelo servico. Se foi rotacao," >&2
        echo "  remova a linha ANTHROPIC_API_KEY de $ENV_FILE." >&2
    fi
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
