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
python -m aristoteles                 # diga "Aristóteles", fale, escute
python -m aristoteles --texto         # sem microfone (só LLM + TTS), para depurar
python -m aristoteles --listar-audio  # descobrir índices de dispositivo
pytest                                # testes
```

Para voltar ao push-to-talk (Enter em vez da palavra), ponha
`wake.modo: push_to_talk` no `config.yaml`.

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
| `aristoteles/wake.py` | push-to-talk e openWakeWord atrás de um `Protocol` |
| `aristoteles/barge_in.py` | vigia o mic durante a fala; a palavra de ativação interrompe |
| `treino/` | treino da wake word (fase 5). Fora do pacote instalado; só isto usa `torch` |

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
- [x] **4** — Pipeline completo com push-to-talk
- [x] **5** — **Wake word "Aristóteles"** — modelo treinado e ativo
  (`wake.modo: openwakeword`)
- [x] **6** — Polimento: barge-in, systemd, reconexão

### Fase 6 — barge-in, serviço e reconexão

**Barge-in: diga "Aristóteles" para interromper a fala.** Medido, o corte sai em
**0 ms** — contra os 11,9 s que a resposta inteira levaria. Depois de interromper,
ele já vai gravar sua próxima pergunta, sem passar pelo ocioso.

O gatilho da interrupção é a **própria palavra de ativação**, não energia de voz.
Sem cancelamento de eco o microfone ouve o alto-falante, e qualquer detector de
energia faria o assistente se interromper na primeira sílaba que ele mesmo
pronuncia. O modelo da wake word é discriminativo e atravessa o próprio áudio do
assistente sem se confundir. Resta um caso: se ele *disser* "Aristóteles" na
resposta, se interrompe — daí a regra no `llm.prompt_sistema` e o `wake.cooldown_s`.

Três decisões em [audio/saida.py](aristoteles/audio/saida.py) que parecem detalhe e
não são, todas descobertas medindo:

1. **Escrever em fatias de 50 ms.** `stream.write()` bloqueia até consumir, e o
   Piper entrega frases inteiras — um bloco de 2 s significava 2 s entre pedir
   silêncio e obtê-lo.
2. **Toda chamada ao stream mora na thread do worker.** A primeira versão chamava
   `abort()` da thread que interrompia; com o worker dentro de `write()`, o
   PortAudio travava — a fala seguinte levava **21 s para tocar 2 s de áudio**, a
   thread emperrava e o processo despejava núcleo. Quem interrompe só marca.
3. **Contador de geração, não uma flag.** Com flag simples havia corrida: o worker
   podia não vê-la antes do `retomar()` e voltava a tocar o bloco descartado — a
   fala interrompida ressuscitava. A geração é monotônica.

**Reconexão.** O SDK já retenta, mas só antes do stream começar; se a conexão cai no
meio, ele não remonta a chamada. O [cerebro.py](aristoteles/cerebro.py) reemite o
turno inteiro com espera dobrando (`llm.reconexoes`, `llm.espera_reconexao_s`) —
**mas apenas enquanto nada tiver sido falado**. Depois da primeira frase no
alto-falante não há desfazer, e repetir a resposta do zero seria pior que admitir o
erro.

**Serviço.**

```bash
./scripts/07_instalar_servico.sh            # instala e valida, sem iniciar
systemctl --user enable --now aristoteles   # ligar agora e no boot
journalctl --user -u aristoteles -f         # acompanhar
systemctl --user disable --now aristoteles   # desfazer
```

Serviço de **usuário**, não de sistema: precisa da sessão de áudio (PipeWire) do seu
login — um serviço de sistema não alcança o microfone. O script recusa instalar se
`wake.modo` não for `openwakeword` (sem terminal, o push-to-talk espera um Enter que
nunca vem) ou se o modelo da wake word não existir, e cria
`~/.config/aristoteles/env` com modo 600 aparando espaços da chave — o systemd não
apara por você, e um `\n` ali reapareceria como falso erro de rede.

### Fase 5 — treinar a wake word

```bash
./scripts/05_instalar_wake.sh                # openWakeWord (veja a ressalva abaixo)
./scripts/06_baixar_dados_wake.sh --tudo     # RIRs + negativos (~17,5 GB)
python -m treino sintetizar                  # ~1.300 clipes das 4 vozes pt-BR
python -m treino gravar                      # ~100 amostras SUAS (10 min no mic)
python -m treino features                    # aumentação + features (16, 96)
python -m treino treinar                     # -> modelos/wake/aristoteles.onnx
python -m treino avaliar                     # curva recall x fp/h, sugere o limiar
```

Depois ponha `wake.modo: openwakeword` no `config.yaml`. O código de treino vive em
[treino/](treino/), fora do pacote `aristoteles` — não vai para a instalação e é o
único lugar que depende de `torch`.

**O plano original desta fase não funcionou, e vale saber por quê.** A ideia era
gerar as amostras com o
[piper-sample-generator](https://github.com/rhasspy/piper-sample-generator) e
treinar pelo notebook oficial. Três paredes:

| Parede | Saída |
|---|---|
| `pip install -e '.[wake]'` falha: o `openwakeword` exige `tflite-runtime`, sem wheel para Python 3.12 | Instalar sem deps. Usamos `inference_framework="onnx"`, e o `tflite_runtime` só é importado no ramo tflite — é código morto aqui. É o que o `05_instalar_wake.sh` faz. |
| O extra `full` pede `tensorflow-cpu==2.8.1`, que não existe no 3.12 | Só serve para converter ONNX→tflite. O export ONNX do openWakeWord é `torch.onnx.export` puro, então nada de TensorFlow. |
| **O piper-sample-generator é só inglês** (checkpoint `en_US-libritts_r`, 904 falantes) | Aqui não há contorno: fonética inglesa numa palavra portuguesa não serve. Medido: `hey jarvis` pronunciado pela voz pt-BR do Piper marca 0.29 no modelo pré-treinado — reage, mas não passa do limiar 0.5. |

Em pt-BR existem só 4 vozes Piper (`faber`, `cadu`, `jeff`, `edresson`) contra os
904 falantes em inglês. Duas defesas contra essa falta de diversidade:

1. **As suas gravações.** Num assistente de um único usuário, generalizar entre
   falantes vale pouco e acertar *a sua* voz vale tudo. São ~10 minutos e é o dado
   de maior valor do treino — o `python -m treino features` avisa se estiverem
   faltando. As gravações levam 3× mais voltas de aumentação que o TTS, para
   compensar serem menos numerosas.
2. **Aumentação** ([treino/aumentar.py](treino/aumentar.py)): posição aleatória na
   janela de 2 s, convolução com 271 respostas impulsivas de salas reais (MIT), ruído
   de fundo em SNR sorteada e ganho aleatório. A reverberação é o que faz o modelo
   funcionar a 2 m do microfone em vez de só colado nele.

O `python -m treino avaliar` substitui o "calibre na mão começando em 0.5": para
cada limiar candidato ele mede o recall nos positivos de validação e os falsos
positivos por hora em ~10,7 h de áudio sem a palavra, e sugere o menor limiar que
fica abaixo de 1 fp/h. Se o recall vier baixo, **mais gravações suas** ajudam mais
que mais passos de treino.

#### Medido

12.000 passos sobre 4.488 positivos aumentados (360 clipes de TTS + 100 gravações
próprias), 6.528 negativos adversariais e os 5,6 milhões de negativos do ACAV100M.
Melhor checkpoint no passo 8.500.

| `wake.limiar` | recall | fp/h | espúrios/dia | adversariais |
|---|---|---|---|---|
| 0.10 | 88,0% | 0,93 | ~22 | 0,1% |
| 0.40 | 85,0% | 0,28 | ~7 | 0,1% |
| **0.50** | **84,3%** | **0,19** | **~4** | 0,1% |
| 0.80 | 81,6% | 0,09 | ~2 | 0,1% |
| 0.95 | 77,0% | 0,00 | 0 | 0,0% |

**As gravações próprias mudaram a coisa nos falsos positivos, não no recall.** No
mesmo limiar 0.50, o modelo só com TTS dava 5,33 fp/h (~128 despertares espúrios por
dia, inutilizável); com as gravações, 0,19. O recall *aparente* caiu de 92,4% para
84,3% porque o conjunto de validação passou a incluir gravações reais, que são mais
difíceis que TTS — teste mais honesto, não regressão.

Esses 84,3% são o pessimista: a aumentação degrada de propósito (reverberação, ruído
a 0–25 dB de SNR, ganho até 0,25×). Em áudio limpo, pelo caminho de streaming real
(`openwakeword.predict` em blocos de 480):

| | resultado |
|---|---|
| **15 gravações próprias reservadas** (nunca vistas no treino) | **15/15, pontuação mínima 0.999** |
| "Aristóteles" nas 4 vozes do Piper, 3 velocidades | 0.988 – 1.000 |
| "Aristides", "Sócrates", "aristocrata", "bom dia" | 0.000 |
| "tóteles" | 0.000 – 0.080 |
| silêncio, ruído branco | 0.000 |
| 90 s de escuta contínua no ambiente | nenhum disparo |

São só 15 clipes de validação — o intervalo de confiança de 15/15 vai até ~80% no
limite inferior. Mas a margem (pior caso 0.999, não 0.6) sugere folga real, e não um
acerto por sorte.

Detalhes que custaram tempo e estão no código:

- A divisão treino/validação é **por arquivo, antes da aumentação**. Se variantes do
  mesmo clipe caíssem nos dois lados, o recall de validação viria inflado.
- Os dois arquivos de negativos do openWakeWord têm **layouts diferentes**, e confundi-los
  quebra o treino: o do ACAV100M já vem janelado, `(5.625.000, 16, 96)` em float16 (5,6
  milhões de exemplos independentes = as 2000 h anunciadas), enquanto o de validação vem
  contínuo, `(481.345, 96)` em float32 — ali as janelas se recortam com passo 1, porque
  para medir falso positivo por hora interessa cada posição possível. Os 17,3 GB ficam em
  memmap.
- O `.onnx` sai com eixo de lote dinâmico, ao contrário dos oficiais (`[1, 16, 96]`
  fixo). O runtime não muda — o `openwakeword` só lê `shape[1]` — mas avaliar 481 mil
  janelas uma a uma seria inviável.
- 16 frames **não** são 16 × 80 ms. O embedding usa janela de 76 frames de
  melspectrograma (760 ms) com passo de 8, então `[1, 16, 96]` corresponde a 2,0 s de
  áudio, não 1,28 s.
- `audio.bloco_ms: 30` (480 amostras) serve, apesar de o openWakeWord recomendar 1280:
  o `predict()` acumula internamente e a pontuação sai **idêntica**.

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

### A janela de escuta parecia instantânea

Sintoma: você diz "Aristóteles", ouve o bipe, e antes de conseguir formular a
pergunta ele já voltou a esperar a palavra de ativação.

A culpa **não era** do `vad.espera_inicial_s`. Quando o gatilho dispara, a cauda da
própria palavra ainda está na fila de captura; ela era contada como fala do usuário e
**armava o endpointing**. Dali em diante valia o `silencio_final_ms` — ou seja, a
janela real para começar a falar era de **700 ms**, não os segundos configurados.

Foram **duas** causas, e a segunda só apareceu depois de consertar a primeira.

**1. A cauda da palavra de ativação.** O [vad.py](aristoteles/vad.py) agora absorve a
primeira sequência de fala como sendo o gatilho, sem armar o endpointing. A absorção
tem teto (`vad.absorver_max_ms`, 1300 ms contra os ~910 ms da palavra) para não
engolir a pergunta de quem fala tudo numa tirada só — "Aristóteles, que horas são?".
O áudio absorvido continua indo para o Whisper, justamente por causa desse caso.

**2. Qualquer ruído curto depois disso.** Um estalo de 90 ms — tosse, cadeira,
teclado — bastava para armar o endpointing e a janela caía de 6 s para 1,4 s. Agora
`houve_fala` exige `vad.fala_minima_ms` **acumulados**, não um bloco isolado, e um
som qualquer *reinicia* a paciência em vez de consumi-la. Medido, nos mesmos
cenários:

| Cenário | Antes | Agora |
|---|---|---|
| gatilho, depois silêncio | 6,3 s | 6,3 s |
| gatilho, pausa, **estalo de 90 ms** | **1,7 s** | 7,0 s |
| gatilho, pausa, pergunta | grava | grava |
| tirada única de 2,2 s | grava | grava |

Quando ele desiste, agora diz por quê: `[nao ouvi nada: ninguem falou em 6s
(vad.espera_inicial_s)]`. O `DetectorFala.ultimo_motivo` guarda o motivo — depois de
diagnosticar isto duas vezes, valia o próximo caso ser uma olhada em vez de uma
investigação.

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
  alta. Não remova as regras de formatação — nem a que proíbe dizer "Aristóteles",
  que existe para ele não se interromper pelo barge-in.
- **`wake.cooldown_s`**: janela de silêncio depois de um disparo. Uma pronúncia
  atravessa várias janelas do modelo e passa do limiar em mais de uma; sem cooldown,
  interromper com "Aristóteles" disparava de novo no bloco seguinte e o assistente
  entrava e saía da gravação. (Existia no config sem nenhum efeito até a fase 6.)
- **`audio.fila_maxima_s`**: teto da fila de captura, em segundos de áudio. Precisa ser
  maior que `vad.duracao_maxima_s`, senão uma pergunta longa perde o começo. Existe
  porque o microfone captura sempre, mas só a gravação consome: sem teto, ficar parado
  no gatilho acumula ~32 KB/s (2,7 GB/dia como serviço). Blocos descartados no ocioso
  são normais e aparecem no encerramento.
- **`llm.turnos_historico`**: pares pergunta/resposta em contexto. O corte descarta
  turnos inteiros — a API exige `role: "user"` na primeira mensagem, então cortar
  mensagem a mensagem deixaria uma *resposta* na frente e daria 400.
- **`llm.timeout_leitura_s`**: o default do SDK é 600 s. Num assistente de voz isso
  significa dez minutos de silêncio se a API travar no meio do stream — é melhor
  desistir e dizer que deu erro. Conta por leitura sem dados, não o total.

## Diagnóstico: "estou sem conexão com a internet" mentindo

Se o assistente disser que está sem internet, **desconfie antes de mexer na rede.**
`anthropic.APITimeoutError` é subclasse de `anthropic.APIConnectionError`, então um
`except APIConnectionError` captura os dois e um simples timeout virava a afirmação
falsa de que a rede caiu. Aconteceu aqui com a rede perfeitamente sã.

O [cerebro.py](aristoteles/cerebro.py) agora trata os dois casos separadamente — a
ordem dos `except` importa, o do timeout precisa vir primeiro — e **os dois imprimem
a causa real** (`__cause__`: DNS, TLS, conexão recusada). O que o assistente fala:

| Situação | Fala | No log |
|---|---|---|
| timeout | "Demorei demais para responder." | `[cerebro] timeout (...) causa: ...` |
| conexão falhou | "Não consegui falar com o servidor agora." | `[cerebro] falha de conexao (...): ...` |
| chave recusada | "Minha chave de acesso foi recusada." | `[cerebro] credencial invalida...` |

Para conferir a rede de fora do aplicativo:

```bash
curl -sS -o /dev/null -w 'http=%{http_code} tls=%{time_appconnect}s\n' \
  https://api.anthropic.com/v1/models \
  -H "x-api-key: $ANTHROPIC_API_KEY" -H 'anthropic-version: 2023-06-01'
```

Note que `api.anthropic.com` resolve para IPv6 nesta máquina. Se o seu IPv6 for
intermitente, o sintoma aparece como timeout na conexão — vale testar `curl -4` e
`curl -6` separadamente para isolar.

### O caso real: `\n` na chave da API

O primeiro erro de conexão de verdade não foi rede nenhuma:

```
[cerebro] falha de conexao (APIConnectionError):
  LocalProtocolError("Illegal header value b'sk-ant-api03-...\n'")
```

Cabeçalho HTTP não aceita quebra de linha, e o SDK repassa o valor cru — então uma
chave terminando em `\n` **não** dá erro de autenticação, dá erro de *protocolo*,
que o SDK embrulha em `APIConnectionError`. O assistente culpa a rede por um defeito
da variável de ambiente.

É fácil cair nisso: `export ANTHROPIC_API_KEY="$(cat arquivo)"` apara o newline
sozinho, mas colar a chave no terminal ou escrevê-la num `EnvironmentFile` do systemd
não apara. Duas defesas no [cerebro.py](aristoteles/cerebro.py):

1. **Apara as pontas** de `ANTHROPIC_API_KEY`/`ANTHROPIC_AUTH_TOKEN` e avisa no log,
   para você corrigir a origem — outros clientes não serão tão tolerantes.
2. **Recusa na inicialização** se sobrar qualquer caractere fora de ASCII imprimível
   (um `\n` no *meio* da chave, que `strip()` não resolveria), com o motivo certo em
   vez de um erro de rede na primeira pergunta.

Se a chave vazar em log, transcrição ou captura de tela, **revogue-a no Claude
Console** — a correção acima não desfaz a exposição.
