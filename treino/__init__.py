"""Ferramentaria de treino da wake word (fase 5). Nao faz parte do runtime.

Fica fora do pacote `aristoteles` de proposito: o `pyproject.toml` empacota so
`aristoteles*`, entao nada daqui vai para a instalacao. Depende de torch, que o
assistente em si nao usa.

Por que nao o pipeline oficial do openWakeWord:

* O `openwakeword.train` importa `openwakeword.data`, que arrasta speechbrain,
  audiomentations, torch-audiomentations, mutagen e acoustics -- e o extra `full`
  ainda pede `tensorflow-cpu==2.8.1`, que nao existe no Python 3.12. Esse
  TensorFlow serve apenas para converter o ONNX em tflite, que nao usamos.
* O gerador de amostras dele (`piper-sample-generator`) e **so ingles**. Para uma
  palavra portuguesa a fonetica sai errada, e era justamente ele quem dava os
  904 falantes de diversidade.

Entao geramos as amostras com as 4 vozes pt-BR do Piper mais gravacoes do dono da
maquina, e reimplementamos aumentacao/treino em numpy + torch puro. O formato de
saida e identico ao oficial -- (16, 96) features, ONNX opset 13 -- porque o
`aristoteles/wake.py` carrega o modelo pelo `openwakeword.model.Model`.
"""
