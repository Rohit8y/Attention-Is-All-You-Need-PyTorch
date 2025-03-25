import math
import torch
import torch.nn as nn

class InputEmbeddings(nn.Module):

    def __init__(self, d_model: int, vocab_size: int):
        super().__init__()
        self.d_model = d_model
        self.vocab_size = vocab_size
        self.embedding = nn.Embedding(vocab_size, d_model)

    def forward(self, x):
        # As suggested in the paper to multiply the embeddings by sqrt(d_model)
        return self.embedding(x) * math.sqrt(self.d_model)


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, seq_len: int, dropout: float) -> None:
        super().__init__()
        self.d_model = d_model
        self.seq_len = seq_len
        self.dropout = nn.Dropout(dropout)
        pe = self.generate_sin_cosine_encodings()

        # Register buffer since the encodings are not going to change
        self.register_buffer('pos_encoding', pe)

    def generate_sin_cosine_encodings(self):
        pe = torch.zeros(self.seq_len, self.d_model)
        for pos in range(self.seq_len):
            for i in range(self.d_model):
                if i % 2 == 0:
                    pe[pos][i] = math.sin(pos / (torch.pow(torch.tensor(10000.0), ((2 * i) / self.d_model))))
                else:
                    pe[pos][i] = math.cos(pos / (torch.pow(torch.tensor(10000.0), ((2 * i) / self.d_model))))

        # Accommodate for batch size (1, seq_len, d_model)
        pe = pe.unsqueeze(0)
        return pe

    def forward(self, x):
        x = x + (self.pos_encoding[:, :x.shape[1], :]).requires_grad_(False)
        return self.dropout(x)


class LayerNormalization(nn.Module):
    def __init__(self, d_model: int):
        super().__init__()
        self.layer_norm = nn.LayerNorm(d_model, eps=1e-6)

    def forward(self, x):
        return self.layer_norm(x)


class FeedForwardBlock(nn.Module):
    def __init__(self, d_model: int, d_ff: int, dropout: float):
        super().__init__()
        self.linear_1 = nn.Linear(d_model, d_ff)
        self.relu = nn.ReLU()
        self.linear_2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.linear_1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.linear_2(x)
        return x


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model: int, heads: int, dropout: float):
        super().__init__()
        assert d_model % heads == 0, "number of heads should perfectly divide the d_model"

        self.d_model = d_model
        self.heads = heads

        # Define weights for Q, K, V
        self.w_q = nn.Linear(d_model, d_model)  # Wq
        self.w_k = nn.Linear(d_model, d_model)  # Wk
        self.w_v = nn.Linear(d_model, d_model)  # Wv

        # Dimension of split part depending on the heads
        self.d_k = d_model // heads

        # Weights for final part when heads are concatenated
        self.w_o = nn.Linear(d_model, d_model)  # Wo

        self.dropout = nn.Dropout(dropout)

    @staticmethod
    def attention(query, key, value, mask, dropout: nn.Dropout):
        d_k = query.shape[-1]

        # For multiplication of Query * Key.transpose(), here is how the shape looks
        # (batch, heads, seq_len, d_k) * (batch, heads, d_k, seq_len) -> (batch, heads, seq_len, seq_len)
        attention_scores = (query @ key.transpose(-2, -1)) / math.sqrt(d_k)
        if mask is not None:
            attention_scores.masked_fill_(mask == 0, -1e9)

        # The last dim is chosen for softmax because we want horizontally
        # the relation between each word in a sentence and the sum prob. is 1.0
        attention_scores = attention_scores.softmax(dim=-1)

        if dropout is not None:
            attention_scores = dropout(attention_scores)

        final_matrix = attention_scores @ value  # (batch, heads, seq_len, d_k)

        return final_matrix, attention_scores

    def forward(self, q, k, v, mask):

        # As per paper the q,k,v are multiplied by the weight vector(w_*) which has dimension (d_model, d_model)
        # But nn.Linear() is also essentially matrix multiplication and takes care of weight and biases
        q_prime = self.w_q(q)
        k_prime = self.w_k(k)
        v_prime = self.w_v(v)

        # The transpose at the end is important to switch seq_len <-> heads so that heads focus on the
        # whole sequence but part of d_model i.e d_k
        # (batch, seq_len, d_model) -> (batch, seq_len, heads, d_k) -> (batch, heads, seq_len, d_k)
        q_split = q_prime.view(q_prime.shape[0], q_prime.shape[1], self.heads, self.d_k).transpose(1, 2)
        k_split = k_prime.view(k_prime.shape[0], k_prime.shape[1], self.heads, self.d_k).transpose(1, 2)
        v_split = v_prime.view(v_prime.shape[0], v_prime.shape[1], self.heads, self.d_k).transpose(1, 2)

        x, self.attention_scores = MultiHeadAttention.attention(q_split, k_split, v_split, mask, self.dropout)

        # Combine all the heads together
        # (batch, heads, seq_len, d_k) --> (batch, seq_len, heads, d_k) --> (batch, seq_len, d_model)
        x = x.transpose(1, 2).contiguous().view(x.shape[0], -1, self.heads * self.d_k)

        # Multiply by Wo
        # (batch, seq_len, d_model) --> (batch, seq_len, d_model)
        return self.w_o(x)


class ResidualConnection(nn.Module):
    def __init__(self, d_model: int, dropout: float):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.norm = LayerNormalization(d_model)

    def forward(self, x, sublayer):
        return x + self.dropout(sublayer(self.norm(x)))


class EncoderBlock(nn.Module):

    def __init__(self, d_model: int, self_attention_block: MultiHeadAttention, feed_forward_block: FeedForwardBlock,
                 dropout: float):
        super().__init__()
        self.self_attention_block = self_attention_block
        self.feed_forward_block = feed_forward_block
        self.dropout = dropout
        self.residual_connections = nn.ModuleList([ResidualConnection(d_model, dropout) for _ in range(2)])

    def forward(self, x, src_mask):
        x = self.residual_connections[0](x, lambda x: self.self_attention_block(x, x, x, src_mask))
        x = self.residual_connections[1](x, self.feed_forward_block)
        return x


class Encoder(nn.Module):

    def __init__(self, d_model: int, layers: nn.ModuleList) -> None:
        super().__init__()
        self.layers = layers
        self.norm = LayerNormalization(d_model)

    def forward(self, x, mask):
        for layer in self.layers:
            x = layer(x, mask)
        return self.norm(x)


class DecoderBlock(nn.Module):
    def __init__(self, d_model: int, mask_attention_block: MultiHeadAttention,
                 cross_attention_block: MultiHeadAttention,
                 feed_forward_block: FeedForwardBlock, dropout: float):
        super().__init__()
        self.mask_attention_block = mask_attention_block
        self.cross_attention_block = cross_attention_block
        self.feed_forward_block = feed_forward_block
        self.dropout = dropout
        self.residual_connections = nn.ModuleList([ResidualConnection(d_model, dropout) for _ in range(3)])

    def forward(self, x, encoder_output, src_mask, tgt_mask):
        x = self.residual_connections[0](x, lambda x: self.mask_attention_block(x, x, x, tgt_mask))
        x = self.residual_connections[1](x, lambda x: self.cross_attention_block(x, encoder_output, encoder_output,
                                                                                 src_mask))
        x = self.residual_connections[2](x, self.feed_forward_block)
        return x


class Decoder(nn.Module):
    def __init__(self, d_model: int, layers: nn.ModuleList) -> None:
        super().__init__()
        self.layers = layers
        self.norm = LayerNormalization(d_model)

    def forward(self, x, encoder_output, src_mask, tgt_mask):
        for layer in self.layers:
            x = layer(x, encoder_output, src_mask, tgt_mask)
        return self.norm(x)


class ProjectionLayer(nn.Module):
    def __init__(self, d_model: int, vocab_size: int):
        super().__init__()
        self.projection = nn.Linear(d_model, vocab_size)

    def forward(self, x):
        x = self.projection(x)
        return x


class Transformer(nn.Module):
    def __init__(self, encoder: Encoder, decoder: Decoder, src_embed: InputEmbeddings, tgt_embed: InputEmbeddings,
                 src_pos: PositionalEncoding, tgt_pos: PositionalEncoding, projection_layer: ProjectionLayer):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.src_embed = src_embed
        self.tgt_embed = tgt_embed
        self.src_pos = src_pos
        self.tgt_pos = tgt_pos
        self.projection_layer = projection_layer

    def encode(self, src, src_mask):
        output = self.src_embed(src)
        output = self.src_pos(output)
        output = self.encoder(output, src_mask)
        return output

    def decode(self, tgt, encoder_output, src_mask, tgt_mask):
        output = self.tgt_embed(tgt)
        output = self.tgt_pos(output)
        output = self.decoder(output, encoder_output, src_mask, tgt_mask)
        return output

    def project(self, decoder_output):
        return self.projection_layer(decoder_output)

    def forward(self, src, src_mask, tgt, tgt_mask):
        x = self.encode(src, src_mask)
        x = self.decode(tgt, x, src_mask, tgt_mask)
        x = self.project(x)
        return x