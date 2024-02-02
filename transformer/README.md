- [Materials](#materials)
- [Basic](#basic)
- [Advanced](#advanced)

----

# Materials

* [Transformer 모델 (Attention is all you need)](https://gaussian37.github.io/dl-concept-transformer/)
  * [Attention 메커니즘의 이해](https://gaussian37.github.io/dl-concept-attention/)
* [[딥러닝 기계 번역] Transformer: Attention Is All You Need (꼼꼼한 딥러닝 논문 리뷰와 코드 실습) | youtube](https://www.youtube.com/watch?v=AA621UofTUA)
  * [pdf](https://github.com/ndb796/Deep-Learning-Paper-Review-and-Practice/blob/master/lecture_notes/Transformer.pdf)
  * [src](https://github.com/ndb796/Deep-Learning-Paper-Review-and-Practice/blob/master/code_practices/Attention_is_All_You_Need_Tutorial_(German_English).ipynb)
* [The Annotated Transformer](https://nlp.seas.harvard.edu/2018/04/03/attention.html)
* [Pytorch Transformers from Scratch (Attention is all you need) | youtube](https://www.youtube.com/watch?v=U0s0f995w14)
  * [pytorch src](https://github.com/aladdinpersson/Machine-Learning-Collection/blob/master/ML/Pytorch/more_advanced/transformer_from_scratch/transformer_from_scratch.py)
* [Visual Guide to Transformer Neural Networks - (Episode 1) Position Embeddings | youtube](https://www.youtube.com/watch?v=dichIcUZfOw)
  * [Visual Guide to Transformer Neural Networks - (Episode 2) Multi-Head & Self-Attention | youtube](https://www.youtube.com/watch?v=mMa2PmYJlCo)
  * [Visual Guide to Transformer Neural Networks - (Episode 3) Decoder’s Masked Attention | youtube](https://www.youtube.com/watch?v=gJ9kaJsE78k)

# Basic

Transformer 모델은 자연어 처리(NLP) 분야에 혁신을 가져온 중요한 기술입니다. 이를 18살도 이해할 수 있게 설명하기 위해, 복잡한 수학이나 전문 용어 없이 기본 개념부터 차근차근 설명하겠습니다.

**Transformer 모델이란?**

Transformer는 단어나 문장을 컴퓨터가 이해하고 처리할 수 있도록 변환하는 모델입니다. 예를 들어, "I like apples"라는 문장을 컴퓨터가 이해할 수 있도록 수치화하고, 이를 바탕으로 문장을 번역하거나 요약하는 작업을 할 수 있습니다.

**핵심 개념**

- Self-Attention: Transformer의 핵심 기능 중 하나입니다. 이는 모델이 문장 내의 각 단어가 다른 단어들과 어떤 관계를 가지는지를 파악할 수 있게 해줍니다. 예를 들어, "The cat sat on the mat"이라는 문장에서 "cat"과 "sat"이 밀접한 관계가 있음을 인식할 수 있습니다.
- Positional Encoding: 문장 내 단어의 순서 정보를 모델에 제공합니다. Transformer는 기본적으로 단어의 순서를 고려하지 않기 때문에, 이를 통해 단어의 순서 정보를 모델에 추가로 알려줍니다.
- Encoder와 Decoder: Transformer 모델은 Encoder와 Decoder로 구성되어 있습니다. Encoder는 입력 문장을 처리하여 내부적인 표현으로 변환하고, Decoder는 이 내부 표현을 사용하여 출력 문장(예: 번역된 문장)을 생성합니다.

PyTorch는 머신러닝 모델을 구현하기 위한 인기 있는 라이브러리입니다. Transformer 모델을 구현하는 간단한 예제 코드를 살펴보겠습니다. 이 코드는 Transformer 모델의 기본적인 구조를 보여주며, 실제로 모델을 학습하기 위한 전체 코드는 아닙니다.

```py
import torch
import torch.nn as nn
import torch.nn.functional as F

class TransformerModel(nn.Module):
    def __init__(self, ntoken, ninp, nhead, nhid, nlayers, dropout=0.5):
        super(TransformerModel, self).__init__()
        from torch.nn import Transformer
        self.model_type = 'Transformer'
        self.pos_encoder = PositionalEncoding(ninp, dropout)
        self.encoder_layer = nn.TransformerEncoderLayer(d_model=ninp, nhead=nhead, dim_feedforward=nhid, dropout=dropout)
        self.transformer_encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=nlayers)
        self.encoder = nn.Embedding(ntoken, ninp)
        self.ninp = ninp
        self.decoder = nn.Linear(ninp, ntoken)

        self.init_weights()

    def generate_square_subsequent_mask(self, sz):
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask

    def init_weights(self):
        initrange = 0.1
        self.encoder.weight.data.uniform_(-initrange, initrange)
        self.decoder.weight.data.uniform_(-initrange, initrange)
        self.decoder.bias.data.zero_()

    def forward(self, src, src_mask):
        src = self.encoder(src) * math.sqrt(self.ninp)
        src = self.pos_encoder(src)
        output = self.transformer_encoder(src, src_mask)
        output = self.decoder(output)
        return F.log_softmax(output, dim=-1)

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * -(math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, d_model)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)
```

이 코드는 Transformer 모델의 기본 구조를 나타내며, 실제 모델을 구성하는 데 필요한 핵심 요소들을 포함하고 있습니다. Positional Encoding과 같은 중요한 개념들이 코드에 반영되어 있으며, 이를 통해 Transformer가 어떻게 작동하는지 이해할 수 있습니다.

위 예제는 시작점이며, Transformer 모델을 완전히 이해하고 활용하기 위해서는 더 많은 학습과 실습이 필요합니다. 각 구성 요소의 역할과 상호 작용을 이해하는 것이 중요하며, 이를 통해 다양한 NLP 문제에 Transformer를 적용할 수 있게 됩니다.

# Advanced


