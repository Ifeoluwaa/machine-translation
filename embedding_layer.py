import torch
import torch.nn as nn

class EmbeddingLayer(nn.Module):
    def __init__(self, vocab_size, emb_dim, seq_len,):
        super(EmbeddingLayer, self).__init__()
        self.word_embedding = nn.Embedding(vocab_size, emb_dim)
        self.positional_embedding = nn.Embedding(seq_len, emb_dim)
        
        
    def forward(self, input_ids):
        _, seq_len = input_ids.shape
        word_embeddings = self.word_embedding(input_ids)
        
        positional_embeddings = self.positional_embedding(torch.arange(seq_len))
        embeddings = word_embeddings + positional_embeddings
        return embeddings
    
class PositionWiseFeedForwardNet(nn.Module):
    def __init__(self, model_dim, width_factor=4):
        super(PositionWiseFeedForwardNet, self).__init__()
        self.linear_1 = nn.Linear(model_dim, width_factor*model_dim)
        self.linear_2 = nn.Linear(width_factor*model_dim, model_dim)
        self.relu = nn.ReLU()
    
    def forward(self, input_tensors):
        output = self.linear_1(input_tensors)
        output = self.relu(output)
        output = self.linear_2(output)
        
        return output