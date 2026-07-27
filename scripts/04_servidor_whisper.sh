#!/usr/bin/env bash
# Sobe o whisper-server (Vulkan) com o modelo residente na GPU.
#
# Manter o servidor no ar evita recarregar o modelo (1-2 s) a cada frase --
# por isso falamos HTTP em vez de invocar o binario por frase.
set -euo pipefail

RAIZ="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
WCPP="$RAIZ/vendor/whisper.cpp"
MODELO="${1:-small}"
PORTA="${2:-8178}"

BIN="$WCPP/build/bin/whisper-server"
GGML="$WCPP/models/ggml-$MODELO.bin"

[[ -x "$BIN"  ]] || { echo "whisper-server nao compilado. Rode ./scripts/03_build_whisper_vulkan.sh"; exit 1; }
[[ -f "$GGML" ]] || { echo "modelo ausente: $GGML"; exit 1; }

echo "Servindo $MODELO em http://127.0.0.1:$PORTA (Ctrl-C para parar)"
echo "Lembre de por 'stt.backend: vulkan' no config.yaml."
exec "$BIN" \
    --model "$GGML" \
    --language pt \
    --host 127.0.0.1 \
    --port "$PORTA" \
    --threads "$(nproc)" \
    --no-timestamps
