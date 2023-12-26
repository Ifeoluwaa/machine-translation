import torch
import torch.nn as nn

class EmbeddingLayer(nn.Module):
    def __init__(self, vocab_size, emb_dim, seq_len):
        super(EmbeddingLayer, self).__init__()

        self.embedding = nn.Embedding(vocab_size, emb_dim)
        self.positional_encoding = self._generate_positional_encoding(emb_dim, seq_len)
        self.emb_dim = emb_dim
    
    def forward(self, input_ids):
        device = input_ids.device
        positional_encoding = self.positional_encoding.to(device)
        
        input_ids = self.embedding(input_ids) * math.sqrt(self.emb_dim)
        input_ids = input_ids.to(device) + positional_encoding[:, :input_ids.size(1)].detach()
        return input_ids

    def _generate_positional_encoding(self, emb_dim, seq_len):
        pe = torch.zeros(seq_len, emb_dim)
        position = torch.arange(0, seq_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, emb_dim, 2) * -(math.log(10000.0) / emb_dim))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        return pe


class PositionWiseFeedForwardNet(nn.Module):
    def __init__(self, model_dim, width_factor=4):
        super(PositionWiseFeedForwardNet, self).__init__()
        self.linear_1 = nn.Linear(model_dim, width_factor * model_dim)
        self.linear_2 = nn.Linear(width_factor * model_dim, model_dim)
        self.relu = nn.ReLU()
    
    def forward(self, input_tensors):
        output = self.linear_1(input_tensors)
        output = self.relu(output)
        output = self.linear_2(output)
        
        return output