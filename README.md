# Aristóteles

Assistente de voz **local** para Linux. Ouve o microfone, transcreve, pergunta ao
Claude e responde em voz. Só a chamada ao LLM sai da máquina.

```
microfone → wake word → VAD → STT → Claude → TTS → alto-falante
             (local)   (local) (local) (nuvem) (local)
```

## Hardware alvo

Ubuntu 24.04, 16 núcleos, 31 GB RAM, **Radeon RX 470/480/570/580/590 (Polaris, gfx803)**.

> **Sobre a GPU AMD:** `faster-whisper` (CTranslate2) é CUDA-only, e o ROCm removeu
> o suporte a gfx803 na série 5.x — então PyTorch+ROCm não é opção nessa placa.
> O caminho de GPU que funciona é **Vulkan** via `whisper.cpp` (driver Mesa RADV).
> O backend `cpu` funciona sem nada disso e é o padrão.

## Instalação

```bash
./scripts/01_deps_sistema.sh          # apt: portaudio, vulkan, build tools
python3 -m venv .venv && source .venv/bin/activate
pip install -e '.[cpu,dev]'
./scripts/02_baixar_modelos.sh        # voz pt-BR do Piper (~60 MB)
export ANTHROPIC_API_KEY=sk-ant-...   # ou: ant auth login
```

O modelo do Whisper (`small`, ~460 MB) é baixado sozinho no primeiro uso.

## Uso

```bash
python -m aristoteles                 # push-to-talk: Enter, fale, escute
python -m aristoteles --texto         # sem microfone (só LLM + TTS), para depurar
python -m aristoteles --listar-audio  # descobrir índices de dispositivo
pytest                                # testes
```

## Estrutura

| Arquivo | Papel |
|---|---|
| `aristoteles/__main__.py` | máquina de estados: OCIOSO → DESPERTO → OUVINDO → … |
| `aristoteles/audio/entrada.py` | captura + **buffer de pré-gravação** (1,5 s antes do gatilho) |
| `aristoteles/audio/saida.py` | fila de reprodução com thread própria (permite barge-in) |
| `aristoteles/vad.py` | webrtcvad + endpointing (quando o usuário parou de falar) |
| `aristoteles/stt/` | dois backends atrás de um `Protocol`: `cpu` e `vulkan` |
| `aristoteles/cerebro.py` | Claude em streaming, histórico com janela |
| `aristoteles/frases.py` | **quebra o stream em frases** → começa a falar antes de a resposta acabar |
| `aristoteles/tts.py` | Piper (CPU, mais rápido que tempo real) |
| `aristoteles/wake.py` | push-to-talk hoje, openWakeWord na fase 5 |

Cada estágio é uma função pura e testável — dá para trocar o STT sem tocar no resto.

## Ligando a GPU (opcional)

```bash
./scripts/03_build_whisper_vulkan.sh small   # compila whisper.cpp + baixa modelo
./scripts/04_servidor_whisper.sh small       # deixa rodando num terminal
# config.yaml → stt.backend: vulkan
```

Confira no log do build/servidor se aparece `AMD Radeon` — se disser `CPU`,
o Vulkan não pegou (rode `vulkaninfo --summary`).

## Roteiro

- [x] **0** — Estrutura, config, testes
- [x] **1** — STT (CPU) → texto
- [x] **2** — TTS → fala
- [x] **3** — Claude em streaming, frase a frase
- [x] **4** — **Pipeline completo com push-to-talk** ← você está aqui
- [ ] **5** — Wake word "Aristóteles" (openWakeWord)
- [ ] **6** — Polimento: barge-in, systemd, reconexão

`SaidaAudio.interromper()` já descarta a fila, mas o bloco que está dentro de
`stream.write()` toca até o fim — para barge-in de verdade falta `stream.abort()` ou
escrever em fatias menores checando a flag entre elas.

### Fase 5 — treinar a wake word

`openWakeWord` treina com áudio sintético — sem gravar centenas de amostras:

1. `pip install -e '.[wake]'`
2. Gere ~2.000 amostras de "Aristóteles" com o
   [piper-sample-generator](https://github.com/rhasspy/piper-sample-generator),
   variando voz, velocidade e ruído de fundo.
3. Treine pelo notebook oficial do openWakeWord → sai um `.onnx`.
4. Salve em `modelos/wake/aristoteles.onnx` e ponha `wake.modo: openwakeword`.

Calibre `wake.limiar` na mão: comece em 0.5 e suba se disparar sozinho.

## Latência medida (nesta máquina, backend `cpu`, modelo `small`)

| Etapa | Medido | Observação |
|---|---|---|
| Silêncio até fim da gravação | 0,70 s | `vad.silencio_final_ms` |
| Whisper `small` (CPU, int8, 8 threads) | **0,85–0,92 s** | RTF 0,30–0,67; carga+aquecimento 2,0 s |
| Claude — 1º token (effort `low`) | **1,36 s** | mediana de 3 perguntas; 1,13–1,50 s |
| Claude — 1ª frase completa | **2,06 s** | 1,92–2,94 s — **é este que conta** |
| Piper (1ª frase) | **~0,05 s** | RTF 0,04 — 25× mais rápido que tempo real; carga 0,9 s |
| **Até ouvir a resposta** | **~3,7 s** | 0,70 + 0,90 + 2,06 + 0,05 |

Atenção à diferença entre as duas linhas do Claude: o áudio não começa no primeiro
token, começa quando [frases.py](aristoteles/frases.py) fecha a **primeira frase** —
são 0,7 s a mais, e é esse o número do orçamento. O `[N.Ns]` que o loop principal
imprime é o tempo até a primeira frase, não até o primeiro token.

O TTS é irrelevante no orçamento. O gargalo é Claude (2,06 s) seguido de Whisper
(0,90 s). Duas alavancas, em ordem de retorno:

1. **Encurtar a primeira frase.** É o caminho mais barato — 0,7 s estão em gerar o
   resto da frase depois do primeiro token. Pedir no `prompt_sistema` que a primeira
   frase seja curta corta isso sem trocar nada de infraestrutura.
2. **Backend Vulkan** para o Whisper, que ganha ~0,9 s e libera CPU (ou permite
   `medium` pelo mesmo custo de latência do `small` em CPU).

## Diagnóstico: o VAD e o ruído de fundo

`webrtcvad` classifica **ruído estacionário de banda larga** (ventoinha,
ar-condicionado, ganho alto de mic USB) como fala. Medido nesta máquina, com
ambiente a RMS 0,028:

| Agressividade | Blocos de ambiente vistos como "fala" |
|---|---|
| 0 | 100/100 |
| 1 | 100/100 |
| 2 | 98/100 |
| 3 | 98/100 |

Subir a agressividade **não resolve**. Sem tratamento, `gravar_ate_silencio`
nunca vê silêncio e grava os 20 s de `duracao_maxima_s` a cada pergunta.

Por isso o [vad.py](aristoteles/vad.py) tem duas camadas: um **gate de energia
auto-calibrado** (mede o piso de ruído na inicialização; exige que a fala esteja
`fator_acima_do_piso` × acima dele) seguido do webrtcvad. Depois do gate:
**0/60 blocos** de ambiente como fala, e o endpointing desiste em 4,1 s quando
ninguém fala.

Se o assistente **cortar suas frases**, baixe `vad.fator_acima_do_piso` para 2,0.
Se **gravar sem parar**, suba para 4,0 ou baixe o ganho do microfone. O gate pode
comer o início de uma fala muito suave — é o que o buffer de pré-gravação compensa
no modo wake word.

## Notas de configuração

- **`llm.effort`**: `low` para conversa, `medium`/`high` para perguntas difíceis.
- **`llm.pensar`**: `false` reduz latência, e é o padrão. Tem dois efeitos colaterais
  documentados do Opus 5 com thinking desligado:
  1. Às vezes ele escreve a chamada de *ferramenta* como texto visível em vez de emitir
     o bloco `tool_use`, e a chamada silenciosamente não executa. Ative `pensar` se
     adicionar ferramentas.
  2. Às vezes vaza `<thinking>` no texto visível — que o Piper leria em voz alta. Daí
     a regra de tags no `prompt_sistema` **e** o filtro `sem_tags()` em `frases.py`.

  (A API também rejeita thinking desligado com `effort` acima de `high` — o
  `config.py` valida isso.) Se preferir eliminar os dois na raiz, ponha `pensar: true`
  com `effort: low` e pague a latência do primeiro token.
- **`vad.agressividade`**: suba para 3 em ambiente ruidoso; desça para 1 se cortar
  suas frases no meio.
- **`llm.prompt_sistema`**: é o que impede o TTS de ler "asterisco asterisco" em voz
  alta. Não remova as regras de formatação.
- **`audio.fila_maxima_s`**: teto da fila de captura, em segundos de áudio. Precisa ser
  maior que `vad.duracao_maxima_s`, senão uma pergunta longa perde o começo. Existe
  porque o microfone captura sempre, mas só a gravação consome: sem teto, ficar parado
  no gatilho acumula ~32 KB/s (2,7 GB/dia como serviço). Blocos descartados no ocioso
  são normais e aparecem no encerramento.
- **`llm.turnos_historico`**: pares pergunta/resposta em contexto. O corte descarta
  turnos inteiros — a API exige `role: "user"` na primeira mensagem, então cortar
  mensagem a mensagem deixaria uma *resposta* na frente e daria 400.
