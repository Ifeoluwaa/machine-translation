from attention import MultiHeadAttention
from embedding_layer import EmbeddingLayer, PositionWiseFeedForwardNet
import torch
import torch.nn as nn


class EncoderLayer(nn.Module):
    def __init__(self, model_dim, num_of_heads, seq_len):
        super(EncoderLayer, self).__init__()
        assert model_dim % num_of_heads == 0, f"model_dim {model_dim} is not divisible by num_of_heads {num_of_heads}"
        self.head_dim = int(model_dim / num_of_heads)

        self.multi_head_attention = MultiHeadAttention(model_dim, self.head_dim, num_of_heads, seq_len, masked=False)
        self.layer_norm_1 = nn.LayerNorm(model_dim)
        self.layer_norm_2 = nn.LayerNorm(model_dim)
        self.position_wise_feed_forward = PositionWiseFeedForwardNet(model_dim)

        
        # Linear projection layers
        self.linear_projection_q = nn.Linear(model_dim, self.head_dim * num_of_heads, bias=False)
        self.linear_projection_k = nn.Linear(model_dim, self.head_dim * num_of_heads, bias=False)
        self.linear_projection_v = nn.Linear(model_dim, self.head_dim * num_of_heads, bias=False)

    def forward(self, input_tensors):
        queries = self._linear_projection(self.linear_projection_q, input_tensors)
        keys = self._linear_projection(self.linear_projection_k, input_tensors)
        values = self._linear_projection(self.linear_projection_v, input_tensors)

        attention_vectors = self.multi_head_attention(queries, keys, values)
        attention_vectors = self.layer_norm_1(attention_vectors + input_tensors)

        feed_forward_output = self.position_wise_feed_forward(attention_vectors)
        output = self.layer_norm_2(feed_forward_output + attention_vectors)

        return output

    def _linear_projection(self, linear_layer, X):
        return linear_layer(X)

    
class DecoderLayer(nn.Module):
    def __init__(self, model_dim, num_of_heads, seq_len):
        super(DecoderLayer, self).__init__()
        assert model_dim % num_of_heads == 0, f"model_dim {model_dim} is not divisible by num_of_heads {num_of_heads}"
        self.head_dim = int(model_dim / num_of_heads)

        self.masked_multi_head_attention = MultiHeadAttention(model_dim, self.head_dim, num_of_heads, seq_len, masked=True)
        self.multi_head_attention = MultiHeadAttention(model_dim, self.head_dim, num_of_heads, seq_len, masked=False)
        self.layer_norm_1 = nn.LayerNorm(model_dim)
        self.layer_norm_2 = nn.LayerNorm(model_dim)
        self.layer_norm_3 = nn.LayerNorm(model_dim)
        self.position_wise_feed_forward = PositionWiseFeedForwardNet(model_dim)
        
        # Linear projection layers
        self.linear_projection_q = nn.Linear(model_dim, self.head_dim * num_of_heads, bias=False)
        self.linear_projection_k = nn.Linear(model_dim, self.head_dim * num_of_heads, bias=False)
        self.linear_projection_v = nn.Linear(model_dim, self.head_dim * num_of_heads, bias=False)

    def forward(self, encoder_output, decoder_input):
        keys, queries, values = self._linear_projection(self.linear_projection_k, decoder_input), \
                               self._linear_projection(self.linear_projection_q, decoder_input), \
                               self._linear_projection(self.linear_projection_v, decoder_input)

        masked_attention_vectors = self.masked_multi_head_attention(keys, queries, values)
        masked_attention_vectors = self.layer_norm_1(masked_attention_vectors + decoder_input)

        attention_vectors = self.multi_head_attention(
            self._linear_projection(self.linear_projection_k, encoder_output),
            queries,
            self._linear_projection(self.linear_projection_v, encoder_output)
        )
        attention_vectors = self.layer_norm_2(attention_vectors + masked_attention_vectors)

        feed_forward_output = self.position_wise_feed_forward(attention_vectors)
        output = self.layer_norm_3(feed_forward_output + attention_vectors)

        return output

    def _linear_projection(self, linear_layer, X):
        return linear_layer(X)


class Encoder(nn.Module):
    def __init__(self, embed_dim, num_of_layers, seq_len, num_of_heads):
        super(Encoder, self).__init__()
        self.num_of_layers = num_of_layers
        self.encoder_layers = nn.ModuleList([EncoderLayer(embed_dim, num_of_heads, seq_len) for _ in range(num_of_layers)])
    
    def forward(self, source_embeddings):
        output = source_embeddings
        for encoder_layer in self.encoder_layers:
            output = encoder_layer(output)
        return output
    
class Decoder(nn.Module):
    def __init__(self, vocab_size, embed_dim, num_of_layers, seq_len, num_of_heads):
        super(Decoder, self).__init__()
        self.num_of_layers = num_of_layers
        self.decoder_layers = nn.ModuleList([DecoderLayer(embed_dim, num_of_heads, seq_len) for _ in range(num_of_layers)])
        self.linear = nn.Linear(embed_dim, vocab_size)

    def forward(self, encoder_output, target_embeddings):
        output = target_embeddings
        for decoder_layer in self.decoder_layers:
            output = decoder_layer(encoder_output, output)
        return nn.functional.log_softmax(self.linear(output), dim=-1)
    
class Transformer(nn.Module):
    def __init__(self, vocab_size, emb_dim, num_of_layers, seq_len, num_of_heads):
        super(Transformer, self).__init__()
        self.embeddings = EmbeddingLayer(vocab_size, emb_dim, seq_len)
        self.encoder = Encoder(emb_dim, num_of_layers, seq_len, num_of_heads)
        self.decoder = Decoder(vocab_size, emb_dim, num_of_layers, seq_len, num_of_heads)
        self.linear = nn.Linear(emb_dim, vocab_size)

    def forward(self, source, target):
        source_embeddings = self.embeddings(source)
        target_embeddings = self.embeddings(target)

        encoder_output = self.encoder(source_embeddings)
        decoder_output = self.decoder(encoder_output, target_embeddings)
        output = self.linear(decoder_output)