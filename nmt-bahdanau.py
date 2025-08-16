"""
nmt_bahdanau.py
Minimal Neural Machine Translation (EN->FR) with Bahdanau Attention
- Encoder-Decoder LSTM
- Additive Attention
- BLEU evaluation on toy dataset
"""

import torch
import torch.nn as nn
import torch.optim as optim
import random
from torchtext.data.metrics import bleu_score

# -----------------------
# Toy Dataset
# -----------------------
pairs = [
    ("i am a student", "je suis un etudiant"),
    ("he is a teacher", "il est un professeur"),
    ("she is reading", "elle lit"),
    ("we are playing", "nous jouons"),
    ("they are eating", "ils mangent")
]

src_vocab = {"<pad>":0, "<sos>":1, "<eos>":2}
trg_vocab = {"<pad>":0, "<sos>":1, "<eos>":2}
for en, fr in pairs:
    for w in en.split(): src_vocab.setdefault(w, len(src_vocab))
    for w in fr.split(): trg_vocab.setdefault(w, len(trg_vocab))

inv_trg_vocab = {i:w for w,i in trg_vocab.items()}

def tensorize(sentence, vocab):
    ids = [vocab[w] for w in sentence.split()]
    return torch.tensor([ [vocab["<sos>"]] + ids + [vocab["<eos>"]] ])

# -----------------------
# Models
# -----------------------
class Encoder(nn.Module):
    def __init__(self, vocab_size, emb_dim, hid_dim):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, emb_dim)
        self.rnn = nn.LSTM(emb_dim, hid_dim, batch_first=True)
    def forward(self, src):
        emb = self.embedding(src)
        outputs, (hidden, cell) = self.rnn(emb)
        return outputs, hidden, cell

class BahdanauAttention(nn.Module):
    def __init__(self, hid_dim):
        super().__init__()
        self.W1 = nn.Linear(hid_dim, hid_dim)
        self.W2 = nn.Linear(hid_dim, hid_dim)
        self.V = nn.Linear(hid_dim, 1)
    def forward(self, hidden, encoder_outputs):
        hidden = hidden[-1].unsqueeze(1) # (batch,1,hid)
        score = self.V(torch.tanh(self.W1(encoder_outputs) + self.W2(hidden)))
        attn = torch.softmax(score, dim=1)
        context = (attn * encoder_outputs).sum(1)
        return context, attn

class Decoder(nn.Module):
    def __init__(self, vocab_size, emb_dim, hid_dim, attention):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, emb_dim)
        self.attention = attention
        self.rnn = nn.LSTM(hid_dim+emb_dim, hid_dim, batch_first=True)
        self.fc = nn.Linear(hid_dim, vocab_size)
    def forward(self, x, hidden, cell, enc_outputs):
        x = self.embedding(x).unsqueeze(1)
        context, attn = self.attention(hidden, enc_outputs)
        rnn_input = torch.cat((x, context.unsqueeze(1)), dim=2)
        output, (hidden, cell) = self.rnn(rnn_input, (hidden, cell))
        pred = self.fc(output.squeeze(1))
        return pred, hidden, cell, attn

# -----------------------
# Training Setup
# -----------------------
INPUT_DIM, OUTPUT_DIM = len(src_vocab), len(trg_vocab)
HID_DIM, EMB_DIM = 64, 32
enc = Encoder(INPUT_DIM, EMB_DIM, HID_DIM)
attn = BahdanauAttention(HID_DIM)
dec = Decoder(OUTPUT_DIM, EMB_DIM, HID_DIM, attn)

enc_opt = optim.Adam(enc.parameters(), lr=0.01)
dec_opt = optim.Adam(dec.parameters(), lr=0.01)
criterion = nn.CrossEntropyLoss(ignore_index=trg_vocab["<pad>"])

# -----------------------
# Training Loop
# -----------------------
for epoch in range(100):
    total_loss = 0
    for en, fr in pairs:
        src = tensorize(en, src_vocab)
        trg = tensorize(fr, trg_vocab)

        enc_opt.zero_grad(); dec_opt.zero_grad()
        enc_out, h, c = enc(src)

        loss = 0
        x = torch.tensor([ [trg_vocab["<sos>"]] ])
        for t in range(1, trg.size(1)):
            output, h, c, attn = dec(x, h, c, enc_out)
            loss += criterion(output, trg[:,t])
            x = trg[:,t].unsqueeze(0) # teacher forcing

        loss.backward()
        enc_opt.step(); dec_opt.step()
        total_loss += loss.item()
    if epoch % 20 == 0:
        print(f"Epoch {epoch}, Loss {total_loss:.3f}")

# -----------------------
# Evaluation (BLEU)
# -----------------------
def translate(sentence):
    src = tensorize(sentence, src_vocab)
    enc_out, h, c = enc(src)
    x = torch.tensor([[trg_vocab["<sos>"]]])
    outputs = []
    for _ in range(10):
        output, h, c, attn = dec(x, h, c, enc_out)
        token = output.argmax(1).item()
        if token == trg_vocab["<eos>"]: break
        outputs.append(inv_trg_vocab[token])
        x = torch.tensor([[token]])
    return " ".join(outputs)

refs, hyps = [], []
for en, fr in pairs:
    pred = translate(en)
    refs.append([fr.split()])
    hyps.append(pred.split())
    print(f"{en} -> {pred} (ref: {fr})")

print("BLEU:", bleu_score(hyps, refs))
