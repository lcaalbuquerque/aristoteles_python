#!/usr/bin/env bash
# Baixa a voz do Piper (pt-BR). O modelo do Whisper para o backend "cpu" e
# baixado automaticamente pelo faster-whisper no primeiro uso.
set -euo pipefail

RAIZ="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DESTINO="$RAIZ/modelos/piper"
mkdir -p "$DESTINO"

BASE="https://huggingface.co/rhasspy/piper-voices/resolve/main/pt/pt_BR"

# Vozes pt-BR disponiveis: faber/medium (padrao, masculina) e edresson/low.
VOZ="${1:-faber/medium}"
NOME="pt_BR-$(echo "$VOZ" | tr '/' '-')"

for ext in onnx onnx.json; do
    arquivo="$DESTINO/$NOME.$ext"
    if [[ -f "$arquivo" ]]; then
        echo "ja existe: $arquivo"
        continue
    fi
    echo "baixando $NOME.$ext ..."
    curl -fL --progress-bar -o "$arquivo" "$BASE/$VOZ/$NOME.$ext"
done

echo
echo "Voz instalada em $DESTINO/$NOME.onnx"
echo "Ajuste 'tts.voz' no config.yaml se usou uma voz diferente da padrao."
