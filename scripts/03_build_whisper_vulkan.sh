#!/usr/bin/env bash
# Compila o whisper.cpp com backend Vulkan para usar a Radeon (Polaris/gfx803).
#
# Por que Vulkan: o ROCm removeu o suporte a gfx803 na serie 5.x, entao PyTorch+ROCm
# nao roda nessa placa. O backend Vulkan do ggml usa o driver Mesa RADV e funciona.
#
# Opcional -- o backend "cpu" (faster-whisper) ja funciona sem isso.
set -euo pipefail

RAIZ="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENDOR="$RAIZ/vendor"
MODELO="${1:-small}"   # tiny | base | small | medium | large-v3

mkdir -p "$VENDOR"
cd "$VENDOR"

if [[ ! -d whisper.cpp ]]; then
    git clone --depth 1 https://github.com/ggml-org/whisper.cpp
fi
cd whisper.cpp
git pull --ff-only || true

echo "== Compilando com Vulkan =="
cmake -B build -DCMAKE_BUILD_TYPE=Release -DGGML_VULKAN=ON
cmake --build build -j"$(nproc)" --config Release

echo
echo "== Baixando modelo ggml '$MODELO' =="
./models/download-ggml-model.sh "$MODELO"

echo
echo "Pronto. Binarios em $VENDOR/whisper.cpp/build/bin/"
echo "Teste rapido na GPU:"
echo "  $VENDOR/whisper.cpp/build/bin/whisper-cli -m models/ggml-$MODELO.bin -l pt samples/jfk.wav"
echo "(procure por 'Vulkan' / 'AMD Radeon' na saida de log -- se aparecer 'CPU', o Vulkan nao pegou)"
echo
echo "Depois suba o servidor com: ./scripts/04_servidor_whisper.sh $MODELO"
