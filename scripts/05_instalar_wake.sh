#!/usr/bin/env bash
# Instala o openWakeWord (fase 5) contornando a dependencia de tflite-runtime.
#
# O openwakeword declara `tflite-runtime>=2.8,<3` no Linux, e o Google parou de
# publicar wheels desse pacote depois do Python 3.11 -- nao existe para o 3.12.
# Mas o wake.py carrega o modelo com inference_framework="onnx", e o
# tflite_runtime so e importado dentro do ramo "tflite" (openwakeword/model.py).
# Ou seja: para nos e codigo morto. Instalamos sem as deps e trazemos as reais.
#
# O `pip check` vai continuar reclamando de tflite-runtime ausente. E esperado.
set -euo pipefail

cd "$(dirname "$0")/.."

if [[ -z "${VIRTUAL_ENV:-}" && -x .venv/bin/pip ]]; then
    PIP=.venv/bin/pip
else
    PIP=pip
fi

echo ">> openwakeword (sem deps: pula o tflite-runtime)"
"$PIP" install --no-deps 'openwakeword>=0.6'

echo ">> dependencias reais do openwakeword"
"$PIP" install 'onnxruntime>=1.10,<2' 'tqdm>=4,<5' 'scipy>=1.3,<2' \
               'scikit-learn>=1,<2' 'requests>=2,<3'

echo ">> baixando os modelos base (melspectrogram + embedding)"
python - <<'PY'
import openwakeword.utils
openwakeword.utils.download_models()
print("modelos base ok")
PY

echo
echo "Pronto. Agora treine 'Aristoteles' (veja README, fase 5) e ponha o .onnx em"
echo "modelos/wake/aristoteles.onnx, ou teste o caminho com um modelo pre-treinado."
