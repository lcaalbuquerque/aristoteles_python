#!/usr/bin/env bash
# Dependencias de sistema (Ubuntu 24.04). Roda com sudo.
set -euo pipefail

echo "== Pacotes base =="
sudo apt-get update
sudo apt-get install -y \
    python3-venv python3-dev \
    libportaudio2 portaudio19-dev \
    pipewire-audio-client-libraries pulseaudio-utils \
    build-essential cmake git curl

echo
echo "== Vulkan (para o backend de STT na Radeon) =="
sudo apt-get install -y \
    mesa-vulkan-drivers libvulkan-dev vulkan-tools

echo
echo "== Verificacao =="
echo -n "PortAudio: "; python3 -c "import ctypes; ctypes.CDLL('libportaudio.so.2'); print('ok')"
echo "Vulkan:"
vulkaninfo --summary 2>/dev/null | grep -E 'deviceName|driverName' || \
    echo "  AVISO: vulkaninfo nao listou dispositivos. Reinicie a sessao e tente de novo."
echo
echo "Feito. Agora: python3 -m venv .venv && source .venv/bin/activate && pip install -e '.[cpu,dev]'"
