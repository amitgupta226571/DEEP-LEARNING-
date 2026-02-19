import torch
import torch.nn as nn
import torch.nn.functional as F

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class Encoder(nn.Module):
    def __init__(self, vocab_size, embedding_dim, enc_units, batch_sz):
        super(Encoder, self).__init__()
        self.batch_sz = batch_sz
        self.enc_units = enc_units
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, enc_units, batch_first=True)

    def forward(self, x, hidden):
        x = self.embedding(x)
        output, hidden_state = self.lstm(x, hidden)
        return output, hidden_state

    def initialize_hidden_state(self):
        return (torch.zeros(1, self.batch_sz, self.enc_units, device=device),
                torch.zeros(1, self.batch_sz, self.enc_units, device=device))

class BahdanauAttention(nn.Module):
    def __init__(self, units):
        super(BahdanauAttention, self).__init__()
        self.W1 = nn.Linear(units, units)
        self.W2 = nn.Linear(units, units)
        self.V = nn.Linear(units, 1)

    def forward(self, query, values):
        # query shape: (1, batch, hidden) -> (batch, 1, hidden)
        query_with_time_axis = query.permute(1, 0, 2)
        score = self.V(torch.tanh(self.W1(query_with_time_axis) + self.W2(values)))
        attention_weights = F.softmax(score, dim=1)
        context_vector = attention_weights * values
        context_vector = torch.sum(context_vector, dim=1)
        return context_vector, attention_weights

class Decoder(nn.Module):
    def __init__(self, vocab_size, embedding_dim, dec_units, batch_sz, use_attention=True):
        super(Decoder, self).__init__()
        self.batch_sz = batch_sz
        self.dec_units = dec_units
        self.use_attention = use_attention
        self.embedding = nn.Embedding(vocab_size, embedding_dim)

        if self.use_attention:
            self.lstm = nn.LSTM(embedding_dim + dec_units, dec_units, batch_first=True)
            self.attention = BahdanauAttention(self.dec_units)
        else:
            self.lstm = nn.LSTM(embedding_dim, dec_units, batch_first=True)

        self.fc = nn.Linear(dec_units, vocab_size)

    def forward(self, x, hidden, enc_output):
        x = self.embedding(x) # x shape: (batch, 1, embedding_dim)

        if self.use_attention:
            # hidden[0] is the hidden state (state_h)
            context_vector, attention_weights = self.attention(hidden[0], enc_output)
            x = torch.cat((context_vector.unsqueeze(1), x), -1)
        else:
            attention_weights = None

        output, hidden_state = self.lstm(x, hidden)
        output = output.reshape(-1, output.shape[2])
        x = self.fc(output)

        return x, hidden_state, attention_weights
