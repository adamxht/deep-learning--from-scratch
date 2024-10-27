# GPT From Scratch

Tutorial to understand transformers architecture better.

Link: https://www.youtube.com/watch?v=kCc8FmEb1nY

Env Setup:

- `conda create -n transformers python=3.12`
- `conda activate transformers`
- `pip install -r requirements.txt`

Sequence:

1. Go through gpt_dev.ipynb
2. Go through bigram.py
3. Go through attention.py

Understand sequentially:

- tril trick
- self attention (decoder - masked, encoder - full)
- mutli head attention
- adding linear layer
- transformer blocks
- residual connection/blocks (Add)
- Layer Norm (Norm)

Stages of training an LLM like GPT:

1. Pretraining on massive internet dataset - self-supervised learninig of just predicting the next token.

- Model has knowledge but no intelligent. It is just a document completer.
- High volume low quality data.

2. Fine tuning with Supervised finetuning and Reinforcement Learning Human Feedback

- Model can learn to align response with human preference complete tasks (qna, summarization, coding, etc) based on knowledge from phase 1.
- Medium volume high quality data.
