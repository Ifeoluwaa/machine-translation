import torch
import torch.nn as nn
import math

class EmbeddingLayer(nn.Module):
    def __init__(self, vocab_size, emb_dim, seq_len, dropout_probability=0.1):
        super(EmbeddingLayer, self).__init__()

        self.embedding = nn.Embedding(vocab_size, emb_dim)
        self.positional_encoding = self._generate_positional_encoding(emb_dim, seq_len)
        self.emb_dim = emb_dim
        self.dropout = nn.Dropout(dropout_probability)
    
    def forward(self, input_ids):
        device = input_ids.device
        positional_encoding = self.positional_encoding.to(device)
        
        input_ids = self.embedding(input_ids) * math.sqrt(self.emb_dim)
        input_ids = input_ids.to(device) + positional_encoding[:, :input_ids.size(1)].detach()
        return self.dropout(input_ids)

    def _generate_positional_encoding(self, emb_dim, seq_len):
        pe = torch.zeros(40, 164)
        position = torch.arange(0, 40).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, 164, 2) * -(math.log(10000.0) / 164))
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