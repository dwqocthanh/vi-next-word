import torch.nn as nn
 
 
class LSTMLM(nn.Module):
    """LSTM 2 tầng: Embedding -> LSTM -> Linear. Trả về logits [B, T, V]."""
 
    def __init__(self, vocab_size, emb=256, hid=512, layers=2, dropout=0.2):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, emb)
        self.lstm = nn.LSTM(emb, hid, layers, batch_first=True,
                            dropout=dropout)
        self.drop = nn.Dropout(dropout)
        self.head = nn.Linear(hid, vocab_size)
 
    def forward(self, x):
        h, _ = self.lstm(self.drop(self.emb(x)))
        return self.head(self.drop(h))
