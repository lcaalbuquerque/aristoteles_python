#!/usr/bin/env bash
# Baixa os dados de treino da wake word (fase 5).
#
#   ./scripts/06_baixar_dados_wake.sh          # so o essencial (~200 MB)
#   ./scripts/06_baixar_dados_wake.sh --tudo   # + 17,3 GB de negativos
#
# O que e cada coisa:
#
#  * RIRs (8 MB)   -- 271 respostas impulsivas de salas reais (MIT). A convolucao
#                     com elas e o que faz o modelo funcionar a 2 m do microfone
#                     em vez de so colado nele. Barato e de alto retorno.
#  * fp_val (180 MB) -- ~11,3 h de features de audio sem a palavra. E a regua de
#                     "falsos positivos por hora"; sem ela nao ha como calibrar o
#                     limiar com honestidade.
#  * ACAV100M (17,3 GB) -- ~2000 h de features negativas, em float16. E o que
#                     ensina o modelo a NAO disparar em fala qualquer. Sem isso o
#                     modelo dispara demais no uso real. Baixe quando puder.
set -euo pipefail

RAIZ="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DESTINO="$RAIZ/dados_wake"
HF="https://huggingface.co"
mkdir -p "$DESTINO"

baixar() {  # baixar <url> <arquivo>
    if [[ -f "$2" ]]; then
        echo "ja existe: $(basename "$2")"
        return
    fi
    echo "baixando $(basename "$2") ..."
    # Retomavel e teimoso: o arquivo do ACAV100M tem 17,3 GB, e o HuggingFace
    # derruba o fluxo no meio ("HTTP/2 stream was not closed cleanly: CANCEL").
    # `-C -` retoma de onde parou; --http1.1 evita justamente esse erro de HTTP/2.
    local tentativa=0
    until curl -fL --http1.1 --progress-bar -C - \
               --retry 5 --retry-delay 5 --retry-all-errors \
               -o "$2.parcial" "$1"; do
        tentativa=$((tentativa + 1))
        if (( tentativa >= 10 )); then
            echo "desisti depois de $tentativa tentativas: $1" >&2
            return 1
        fi
        echo "  caiu; retomando (tentativa $tentativa)..." >&2
        sleep 5
    done
    mv "$2.parcial" "$2"   # so renomeia se completou, para poder repetir o script
}

# --- respostas impulsivas de sala ---
if [[ -d "$DESTINO/rir" ]] && [[ -n "$(ls -A "$DESTINO/rir" 2>/dev/null)" ]]; then
    echo "ja existe: rir/ ($(ls "$DESTINO/rir" | wc -l) arquivos)"
else
    mkdir -p "$DESTINO/rir"
    echo "baixando as respostas impulsivas (MIT, 271 arquivos)..."
    "$RAIZ/.venv/bin/python" - "$DESTINO/rir" <<'PY'
import sys, requests
from pathlib import Path
destino = Path(sys.argv[1])
repo = "davidscripka/MIT_environmental_impulse_responses"
arvore = requests.get(
    f"https://huggingface.co/api/datasets/{repo}/tree/main?recursive=true",
    timeout=60).json()
wavs = [f["path"] for f in arvore if f["type"] == "file" and f["path"].endswith(".wav")]
for i, caminho in enumerate(wavs, 1):
    alvo = destino / Path(caminho).name
    if alvo.exists():
        continue
    url = f"https://huggingface.co/datasets/{repo}/resolve/main/{caminho}"
    alvo.write_bytes(requests.get(url, timeout=120).content)
    print(f"\r  {i}/{len(wavs)}", end="", flush=True)
print()
PY
fi

# --- features de validacao de falso positivo ---
FEAT="$HF/datasets/davidscripka/openwakeword_features/resolve/main"
baixar "$FEAT/validation_set_features.npy" "$DESTINO/validation_set_features.npy"

# --- negativos em escala (opcional, grande) ---
if [[ "${1:-}" == "--tudo" ]]; then
    baixar "$FEAT/openwakeword_features_ACAV100M_2000_hrs_16bit.npy" \
           "$DESTINO/acav100m_2000h.npy"
else
    echo
    echo "Pulei os 17,3 GB de negativos do ACAV100M."
    echo "Sem eles o modelo dispara demais em fala qualquer. Rode com --tudo"
    echo "quando tiver banda; o treino funciona sem, mas so para experimentar."
fi

echo
du -sh "$DESTINO"/* 2>/dev/null || true
