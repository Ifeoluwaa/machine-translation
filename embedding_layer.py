import torch
import torch.nn as nn

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class EmbeddingLayer(nn.Module):
    def __init__(self, emb_dim, vocab_size, seq_len):
        super (EmbeddingLayer, self).__init__()
        self.emb_dim = emb_dim
        self.vocab_size = vocab_size
        self.seq_len = seq_len

        self.word_embedding = nn.Embedding(vocab_size, emb_dim)
        self.positional_embedding = nn.Embedding(seq_len, emb_dim)


    def forward(self, input_idx):

        _, seq_len = input_idx.shape
        print("Input Indices:", input_idx)
        
        # Check for invalid indices in word_embedding
        invalid_indices = (input_idx < 0) | (input_idx >= self.vocab_size)
        if torch.any(invalid_indices):
            print("Invalid indices detected in word_embedding:", input_idx[invalid_indices])
            raise ValueError("Invalid input indices detected in word_embedding.")

        word_embed = self.word_embedding(input_idx)
        
        if torch.any(input_idx >= seq_len):
            print("Invalid indices detected. Indices should be within the sequence length:", input_idx)
            raise ValueError("Invalid input indices detected. Indices should be within the sequence length.")

   # Check for invalid positional indices
        if torch.any(torch.arange(seq_len).to(device) >= self.seq_len):
            print("Invalid positional indices detected in positional_embedding.")
            raise ValueError("Invalid positional indices detected in positional_embedding.")
        position_embed = self.positional_embedding(torch.arange(seq_len).to(device))
        print(word_embed.shape)
        #print(position_embed.shape)
        return word_embed + position_embed




class PositionwiseFeedforward(nn.Module):
    def __init__(self, input_dimension, width_factor=4):
        super(PositionwiseFeedforward, self).__init__()
        self.input_dimension = input_dimension
        self.linear1 = nn.Linear(input_dimension, input_dimension * width_factor)
        self.linear2 = nn.Linear(input_dimension * width_factor, input_dimension)
        self.gelu = nn.GELU()


    def forward(self, input_tensors):
        #x = input_tensors
        output = self.linear1(input_tensors)
        output = self.gelu(output)
        output = self.linear2(output)

        return output
