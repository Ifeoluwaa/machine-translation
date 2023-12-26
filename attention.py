import torch
import torch.nn as nn


class Attention(nn.Module):
    def __init__(self, seq_len, masked=True):
        super(Attention, self).__init__()
        self.masked = masked
        
        if self.masked:
            self.register_buffer("tril", torch.tril(torch.ones(seq_len, seq_len)))
        
        self.softmax = nn.Softmax(dim=-1)
        
    def forward(self, keys, queries, values, head_dim):
        scores = torch.matmul(queries, keys.transpose(-2, -1)) / torch.sqrt(torch.tensor(head_dim).float())
        
        if self.masked:
            scores = scores.masked_fill(self.tril==0, float("-inf"))
            
        attention_weights = self.softmax(scores)
        attention_vectors = torch.matmul(attention_weights, values)
        return attention_vectors
    
class MultiHeadAttention(nn.Module):
    def __init__(self, model_dim, head_dim, num_of_heads, seq_len, masked=True):
        super(MultiHeadAttention, self).__init__()
        self.head_dim = head_dim
        self.W_q = nn.Linear(model_dim, self.head_dim, bias=False)
        self.W_k = nn.Linear(model_dim, self.head_dim, bias=False)
        self.W_v = nn.Linear(model_dim, self.head_dim, bias=False)
    
        self.attention_heads = nn.ModuleList(Attention(seq_len, masked) 
                                             for _ in range(num_of_heads))
        self.W_o = nn.Linear(num_of_heads*self.head_dim, model_dim, bias=False)
        
    def _linear_projection(self, X):
        W = nn.Linear(X.size(-1), self.head_dim, bias=False).to(X.device)
        return W(X)
        
    def forward(self, keys, queries, values):
        heads = [attention_head(self._linear_projection(keys), 
                                    self._linear_projection(queries), 
                                    self._linear_projection(values), 
                                    self.head_dim
                                )
                    for attention_head in self.attention_heads
                ]
            
        concatenated_heads = torch.cat(heads, dim=-1)
        attention_vectors = self.W_o(concatenated_heads)
        
        return attention_vectors