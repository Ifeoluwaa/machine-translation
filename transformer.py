import torch
import torch.nn as nn
from attention import MultiheadAttention
from embedding_layer import EmbeddingLayer, PositionwiseFeedforward


class EncoderLayer(nn.Module):
    def __init__(self, input_dimension, num_heads, seq_len, width_factor=4):
        super(EncoderLayer, self).__init__()
        self.input_dimension = input_dimension
        self.num_heads = num_heads
        self.seq_len = seq_len
        self.width_factor = width_factor

        self.multiheadattn = MultiheadAttention(input_dimension, num_heads, seq_len, mask=False)
        self.positionwise_ff = PositionwiseFeedforward(input_dimension, width_factor=4)
        self.layer_norm1 = nn.LayerNorm(input_dimension)
        self.layer_norm2 = nn.LayerNorm(input_dimension)


    def _linear_projection(self, X):
        W = nn.Linear(X.size(-1), self.head_dim, bias=False)
        return W(X)


    def forward(self, encoder_input):
        encoder_output = self.multiheadattn(encoder_input, encoder_input, encoder_input)
        encoder_output = encoder_output + encoder_input
        encoder_output = self.layer_norm1(encoder_output)

        encoder_output_ff = self.positionwise_ff(encoder_output)
        encoder_output = encoder_output_ff + encoder_output
        encoder_output = self.layer_norm2(encoder_output)

        return encoder_output



class DecoderLayer(nn.Module):
    def __init__(self, input_dimension, num_heads, seq_len, width_factor=4):
        super(DecoderLayer, self).__init__()
        self.multiheadattn = MultiheadAttention(input_dimension, num_heads, seq_len, mask=True)
        self.cross_attn = MultiheadAttention(input_dimension, num_heads, seq_len, mask =False)
        self.positionwise_ff = PositionwiseFeedforward(input_dimension, width_factor)

        self.layer_norm1 = nn.LayerNorm(input_dimension)
        self.layer_norm2 = nn.LayerNorm(input_dimension)
        self.layer_norm3 = nn.LayerNorm(input_dimension)


    def _linear_projection(self, X):
        W = nn.Linear(X.size(-1), self.head_dim, bias=False)
        return W(X)


    def forward(self, decoder_input, encoder_output):
        decoder_output = self.multiheadattn(decoder_input, decoder_input, decoder_input)
        decoder_output = decoder_output + decoder_input
        decoder_output = self.layer_norm1(decoder_output)

        decoder_output = self.cross_attn(decoder_output, encoder_output, encoder_output)
        print(decoder_output)
        decoder_output = self.layer_norm2(decoder_output)
        print(decoder_output)

        decoder_output_ff = self.positionwise_ff(decoder_output)
        decoder_output = decoder_output_ff + decoder_output
        decoder_output = self.layer_norm3(decoder_output)

        return decoder_output




class Encoder(nn.Module):
    def __init__(self, emb_dim, vocab_size, seq_len, input_dimension, num_layers, num_heads, width_factor=4):
        super(Encoder, self).__init__()

        self.emb_dim = emb_dim

        self.vocab_size = vocab_size
        self.seq_len = seq_len
        self.input_dimension = input_dimension
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.width_factor = width_factor

        self.embeddinglayer = EmbeddingLayer(emb_dim, vocab_size,seq_len)
        self.layers = nn.ModuleList(EncoderLayer(input_dimension, num_heads, seq_len, width_factor=4) for i in range(num_layers))


    def forward(self, encoder_input):

        encoder_output = self.embeddinglayer(encoder_input)
        for layer in self.layers:
            encoder_output = layer(encoder_output)

        return encoder_output




class Decoder(nn.Module):
    def __init__(self, emb_dim, vocab_size, seq_len, input_dimension, num_layers, num_heads, width_factor=4):
        super(Decoder, self).__init__()
        self.input_dimension = input_dimension
        self.num_layers = num_layers
        self.emb_dim = emb_dim
        self.vocab_size = vocab_size
        self.num_heads = num_heads
        self.seq_len = seq_len
        self.width_factor = width_factor
        self.embeddinglayer = EmbeddingLayer(emb_dim, vocab_size,seq_len)
        self.layers = nn.ModuleList([DecoderLayer(input_dimension, num_heads, seq_len, width_factor=4) for i in range(num_layers)])


    def forward(self, decoder_input, encoder_output):
        decoder_output = self.embeddinglayer(decoder_input)

        for layer in self.layers:
            decoder_output = layer(decoder_output, encoder_output)

        return decoder_output



class Transformer(nn.Module):
    def __init__(self, emb_dim, source_vocab_size, target_vocab_size, source_seq_len, target_seq_len,  input_dimension, num_heads, num_layers, width_factor=4):
        super(Transformer, self).__init__()
        self.source_vocab_size = source_vocab_size
        self.target_vocab_size = target_vocab_size
        self.source_seq_len = source_seq_len
        self.target_seq_len = target_seq_len
        self.emb_dim = emb_dim
        self.num_heads = num_heads
        self.num_layers = num_layers
        
        
        self.encoder = Encoder(emb_dim, source_vocab_size, source_seq_len, input_dimension, num_layers, num_heads, width_factor)
        
        self.decoder = Decoder(emb_dim, target_vocab_size, target_seq_len, input_dimension, num_layers, num_heads, width_factor)
        self.positionwise_feedforward = PositionwiseFeedforward(emb_dim, target_vocab_size)
        
        
    def forward(self, source, target):
        #encoder forward pass
        encoder_output = self.encoder(source)
        #decoder forward pass
        output = self.decoder(target, encoder_output)

        output = self.positionwise_feedforward(output)
    
        return output

