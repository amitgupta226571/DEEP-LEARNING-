import torch
import torch.nn as nn
import torch.nn.functional as F

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ... [Keep your existing Encoder class here] ...

class LuongAttention(nn.Module):
    def __init__(self, hidden_dim):
        super(LuongAttention, self).__init__()
        # The learnable weight matrix Wa for the "general" scoring method
        self.Wa = nn.Linear(hidden_dim, hidden_dim, bias=False)

    def forward(self, query, keys):
        # query shape: (1, batch, hidden_dim) -> (batch, 1, hidden_dim)
        query = query.permute(1, 0, 2)

        # 1. Apply linear transformation to the query
        # query_transformed shape: (batch, 1, hidden_dim)
        query_transformed = self.Wa(query)

        # 2. Calculate alignment scores using dot product (Batch Matrix Multiplication)
        # keys shape: (batch, seq_len, hidden_dim) -> transposed to (batch, hidden_dim, seq_len)
        # scores shape: (batch, 1, seq_len)
        scores = torch.bmm(query_transformed, keys.transpose(1, 2))

        # 3. Apply softmax to get attention weights
        # attention_weights shape: (batch, 1, seq_len)
        attention_weights = F.softmax(scores, dim=-1)

        # 4. Multiply weights by encoder outputs (keys) to get the context vector
        # context_vector shape: (batch, 1, hidden_dim)
        context_vector = torch.bmm(attention_weights, keys)

        # Reshape for the decoder:
        # Context vector: (batch, hidden_dim)
        # Attention weights: (batch, seq_len, 1) for plotting
        return context_vector.squeeze(1), attention_weights.transpose(1, 2)

class Decoder(nn.Module):
    def __init__(self, vocab_size, embedding_dim, dec_units, batch_sz, use_attention=True):
        super(Decoder, self).__init__()
        self.batch_sz = batch_sz
        self.dec_units = dec_units
        self.use_attention = use_attention

        self.embedding = nn.Embedding(vocab_size, embedding_dim)

        if self.use_attention:
            # We are now using Luong Attention instead of Bahdanau
            self.attention = LuongAttention(self.dec_units)
            # The input to LSTM will be the concatenated context vector and embedding
            self.lstm = nn.LSTM(embedding_dim + dec_units, dec_units, batch_first=True)
        else:
            self.lstm = nn.LSTM(embedding_dim, dec_units, batch_first=True)

        self.fc = nn.Linear(dec_units, vocab_size)

    def forward(self, x, hidden, enc_output):
        # x shape: (batch, 1)
        x = self.embedding(x) # x shape: (batch, 1, embedding_dim)

        if self.use_attention:
            # Calculate Luong attention using the hidden state (hidden[0]) and encoder outputs
            context_vector, attention_weights = self.attention(hidden[0], enc_output)

            # Concatenate context vector and embedded input
            # x shape after concat: (batch, 1, embedding_dim + dec_units)
            x = torch.cat((context_vector.unsqueeze(1), x), -1)
        else:
            attention_weights = None

        # Pass through LSTM
        output, hidden_state = self.lstm(x, hidden)

        # Project to vocabulary size
        output = output.reshape(-1, output.shape[2])
        x = self.fc(output)

        return x, hidden_state, attention_weights
