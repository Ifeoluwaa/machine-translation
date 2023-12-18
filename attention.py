import torch
import torch.nn as nn


class SelfAttention(nn.Module):
    def __init__(self, input_dimension, head_dim, seq_len, mask):
        super(SelfAttention, self).__init__()
        self.input_dimension = input_dimension
        self.head_dim = head_dim
        self.mask = mask

        self.W_q = nn.Linear(input_dimension, head_dim, bias=False)
        self.W_k = nn.Linear(input_dimension, head_dim, bias=False)
        self.W_v = nn.Linear(input_dimension, head_dim, bias=False)

        self.softmax = nn.Softmax(dim=-1)
        self.register_buffer('tril', torch.tril(torch.ones(seq_len, seq_len)))


    def forward(self, query, key, value):
        Q = self.W_q(query)
        K = self.W_k(key)
        V = self.W_v(value)
        scaled_attention = torch.matmul(Q, K.transpose(-2, -1)) / self.head_dim ** 0.5
        #where Q, K qnd V are queries, keys and values respectively

        if self.mask:
            scaled_attention = scaled_attention.masked_fill(self.tril==0, float("-inf"))

        attention_weights = self.softmax(scaled_attention)
        #return attention_weights

        weighted_value_vectors = torch.matmul(attention_weights, V)
        return weighted_value_vectors





class MultiheadAttention(nn.Module):
    def __init__(self, input_dimension, num_heads, seq_len, mask):
        super(MultiheadAttention, self).__init__()
        assert input_dimension % num_heads == 0

        self.input_dimension = input_dimension
        self.num_heads = num_heads
        self.seq_len = seq_len
        self.mask = mask

        self.head_dim = int(input_dimension / num_heads)

        self.attention_heads = nn.ModuleList(SelfAttention(self.input_dimension, self.head_dim, self.seq_len, self.mask)
                                             for _ in range(num_heads))

        self.W_o = nn.Linear(self.num_heads * self.head_dim, self.input_dimension, bias=False)


    def forward(self, query, key, value):
        heads = [attention_head(query, key, value) for attention_head in self.attention_heads]
        heads_contatenation = torch.cat(heads, dim=-1)

        weighted_value_vectors = self.W_o(heads_contatenation)

        return weighted_value_vectors

