"""DKT model definition in MindSpore."""

from mindspore import nn


class DKTModel(nn.Cell):
    """Deep Knowledge Tracing model."""

    def __init__(self, num_skills: int, emb_dim: int = 64, hidden_size: int = 128, rnn_type: str = "gru"):
        super().__init__()
        if num_skills <= 0:
            raise ValueError("num_skills must be > 0")
        vocab_size = 2 * num_skills
        self.embedding = nn.Embedding(vocab_size, emb_dim)
        rnn_type = (rnn_type or "gru").lower()
        if rnn_type == "lstm":
            self.rnn = nn.LSTM(emb_dim, hidden_size, batch_first=True)
        else:
            self.rnn = nn.GRU(emb_dim, hidden_size, batch_first=True)
        self.fc = nn.Dense(hidden_size, num_skills)
        self.sigmoid = nn.Sigmoid()

    def construct(self, x):
        emb = self.embedding(x)
        rnn_out, _ = self.rnn(emb)
        logits = self.fc(rnn_out)
        probs = self.sigmoid(logits)
        return probs
