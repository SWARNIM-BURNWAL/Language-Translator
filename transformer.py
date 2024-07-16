import math
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor, matmul, ones, zeros, sqrt, device, cuda, pow, cos, sin, stack, arange, flatten, float, tensor, stack, backends


def get_device():

    return device('mps' or 'cuda') if backends.mps.is_available() or cuda.is_available() else device('cpu')


def scaled_dot_product_attention(Q: Tensor, K: Tensor, V: Tensor, mask=None):
    d_k = Q.size(-1)
    scaled = matmul(Q, K.transpose(-1, -2)) / math.sqrt(d_k)
    if mask is not None:
        scaled = scaled.permute(1, 0, 2, 3) + mask
        scaled = scaled.permute(1, 0, 2, 3)
    attention = F.softmax(scaled, dim=-1)
    values = matmul(attention, V)
    return values, attention


class PositionalEncoding(nn.Module):
    "Implement the positional encoding function"

    def __init__(self, d_model, sequence_length) -> None:
        super().__init__()
        self.d_model = d_model
        self.sequence_length = sequence_length

    def forward(self):
        index = arange(0, self.d_model, 2).float()  # random index
        denominator = pow(10000, index/self.d_model)
        position = arange(self.sequence_length, dtype=float).unsqueeze(1)
        even_position = sin(position/denominator)
        odd_position = cos(position/denominator)
        stacked = stack((even_position, odd_position), dim=2)
        positional_encoding = flatten(stacked, start_dim=1, end_dim=2)
        return positional_encoding


class SentenceEmbedding(nn.Module):
    "For a given sentence, create an embedding"

    def __init__(self, max_sequence_length, d_model, language_to_index, START_TOKEN, END_TOKEN, PADDING_TOKEN):
        super().__init__()
        self.vocab_size = len(language_to_index)
        self.max_sequence_length = max_sequence_length
        self.embedding = nn.Embedding(self.vocab_size, d_model)
        self.language_to_index = language_to_index
        self.position_encoder = PositionalEncoding(
            d_model, max_sequence_length)
        self.dropout = nn.Dropout(p=0.1)
        self.START_TOKEN = START_TOKEN
        self.END_TOKEN = END_TOKEN
        self.PADDING_TOKEN = PADDING_TOKEN

    def batch_tokenize(self, batch, start_token, end_token):

        def tokenize(sentence, start_token=True, end_token=True):
            print(f"Sentence: {sentence}")
            print("Sentence Type: ", type(sentence))
            sentence_word_indices = [self.language_to_index[token]for token in list(sentence)]
            if start_token:
                sentence_word_indices.insert(0, self.language_to_index[self.START_TOKEN])
            if end_token:
                sentence_word_indices.append(self.language_to_indexx[self.END_TOKEN])

            for _ in range(len(sentence_word_indices), self.max_sequence_length):
                sentence_word_indices.append(self.language_to_index[self.PADDING_TOKEN])
            return tensor(sentence_word_indices)

        tokenized = []
        for sentence_num in range(len(batch)):
            tokenized.append(
                tokenize(batch[sentence_num], start_token, end_token))
        tokenized = stack(tokenized)
        return tokenized.to(get_device())

    def forward(self, x, start_token, end_token):  # sentence
        x = self.batch_tokenize(x, start_token, end_token)
        x = self.embedding(x)
        pos = self.position_encoder().to(get_device())
        x = self.dropout(x + pos)
        return x


class MultiHeadAttention(nn.Module):
    "   MultiHeadAttention: Perform scaled dot-product attention on the input tensor    "

    def __init__(self, d_model: int, heads: int):
        super(MultiHeadAttention, self).__init__()
        self.d_model = d_model  # 512
        self.heads = heads  # 8
        self.head_dim = self.d_model // self.heads  # 64
        # 512*3 = 1536 this is done to perform the operation parallely we are stacking up the vectors
        self.qkv_layer = nn.Linear(d_model, 3*d_model)
        print(self.qkv_layer.weight.data.shape, self.qkv_layer.bias.data.shape)
        self.linear_layer = nn.Linear(d_model, d_model)
        print(self.linear_layer.weight.data.shape,
              self.linear_layer.bias.data.shape)

    def forward(self, y: Tensor, mask: Tensor = None):
        batch_size, sequence_length, d_model = y.size()  # 30, 200, 512
        print(f"Batch Size:  {batch_size}")
        print(f"Sequence Length:  {sequence_length}")
        print(f"Model Size:  {d_model}")
        qkv: Tensor = self.qkv_layer(y)  # 30, 200, 1536
        print(f"qkv.size(): {qkv.size()}")
        qkv = qkv.reshape(batch_size, sequence_length,
                          self.heads, 3*self.head_dim)  # 30, 200, 8, 192
        #  this is done in order to rearrange the dimensions of the tensor to ensure the correct alignment for splitting the tensor into queries, keys, and values for each head of the multi-head attention
        qkv = qkv.permute(0, 2, 1, 3)  # 30, 8, 200, 192
        print(f"qkv.size() after permute: {qkv.size()}")
        q, k, v = qkv.chunk(3, dim=-1)  # 30, 8, 200, 64
        print(f"Q Size:  {q.size()}")
        print(f"K Size:  {k.size()}")
        print(f"V Size:  {v.size()}")
        values, attention = scaled_dot_product_attention(q, k, v, mask)
        values = values.permute(0, 2, 1, 3).reshape(
            batch_size, sequence_length, self.heads*self.head_dim)
        out = self.linear_layer(values)
        return out


class MultiHeadCrossAttention(nn.Module):
    "   MultiHeadCrossAttention: Perform scaled dot-product attention on the input tensor    "

    def __init__(self, d_model, num_heads) -> None:
        super(MultiHeadCrossAttention, self).__init__()
        self.d_model = d_model  # 512
        self.heads = num_heads  # 8
        self.head_dim = self.d_model // self.heads  # 64
        self.kv_layer = nn.Linear(d_model, 2*d_model)  # 512, 1024
        self.q_layer = nn.Linear(d_model, d_model)
        self.linear_layer = nn.Linear(d_model, d_model)

    def forward(self, x: Tensor, y: Tensor, mask: Tensor = None):
        batch_size, sequence_length, d_model = x.size()  # 30, 200, 512
        print(f"Batch Size:  {batch_size}")
        print(f"Sequence Length:  {sequence_length}")
        print(f"Model Size:  {d_model}")
        kv: Tensor = self.kv_layer(x)  # 30, 200, 1024
        print(f"kv.size(): {kv.size()}")
        q: Tensor = self.q_layer(y)  # 30, 200, 512
        print(f"Q Size:  {q.size()}")
        kv = kv.reshape(batch_size, sequence_length,
                        self.heads, 2*self.head_dim)  # 30, 200, 8, 128
        q = q.reshape(batch_size, sequence_length, self.heads,
                      self.head_dim)  # 30, 200, 8, 64
        q = q.permute(0, 2, 1, 3)  # 30, 8, 200, 64
        print(f"q.size() after permute: {q.size()}")
        kv = kv.permute(0, 2, 1, 3)  # 30, 8, 200, 128
        print(f"kv.size() after permute: {kv.size()}")
        # chunking the tensor into keys and values 30, 8, 200, 64
        k, v = kv.chunk(2, dim=-1)
        print(f"K Size:  {k.size()}")
        print(f"V Size:  {v.size()}")
        values, attention = scaled_dot_product_attention(q, k, v, mask)
        values = values.permute(0, 2, 1, 3).reshape(
            batch_size, sequence_length, self.heads*self.head_dim)
        out = self.linear_layer(values)
        print(f"Output Size after applying linear layer:  {out.size()}")

        return out


class LayerNormalisation(nn.Module):
    "   LayerNormalisation: Perform layer normalisation on the input tensor    "

    def __init__(self, parameter_shape, epsilon=1e-5) -> None:
        super(LayerNormalisation, self).__init__()
        # standard deviation for values
        self.Gamma = nn.Parameter(ones(parameter_shape))
        self.Beta = nn.Parameter(zeros(parameter_shape))  # mean for values
        self.parameter_shape = parameter_shape  # 512
        self.epsilon = epsilon  # epsilon value to avoid division by zero or very small number

    def forward(self, inputs: Tensor) -> Tensor:
        # [-1] perform layer normalisation only at last dimension
        dimensions = [-(i+1) for i in range(len(self.parameter_shape))]
        mean = inputs.mean(dim=dimensions, keepdim=True)
        print(f"Mean Size:  {mean.size()}")
        variance = (inputs-mean).pow(2).mean(dim=dimensions, keepdim=True)
        print(f"Variance Size:  {variance.size()}")
        std = sqrt(variance+self.epsilon)
        print(f" Standard Deviation Size:  {std.size()}")
        y = (inputs-mean)/std
        print(f"Y Size:  {y.size()}")
        output = self.Gamma*y+self.Beta
        print(f"Output Size:  {output.size()}")
        return output

class PointFeedForward(nn.Module):
    "   PointFeedForward: Perform feed forward network on the input tensor    "

    def __init__(self, d_model, hidden_ffn, dropout=0.1) -> None:
        super(PointFeedForward, self).__init__()
        self.linear1 = nn.Linear(d_model, hidden_ffn)
        self.linear2 = nn.Linear(hidden_ffn, d_model)
        self.activation = nn.ReLU()
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x: Tensor) -> Tensor:
        x = self.linear1(x)
        print(f"y after first linear layer: {x.size()}")

        x = self.activation(x)
        print(f"y after activation: {x.size()}")

        x = self.dropout(x)
        print(f"y after dropout: {x.size()}")

        x = self.linear2(x)
        print(f"y after 2nd linear layer: {x.size()}")

        return x
class EncoderLayer(nn.Module):
    "  EncoderLayer: Perform encoder layer operations on the input tensor    "

    def __init__(self, d_model, ffn_hidden, num_heads, drop_prob):
        super(EncoderLayer, self).__init__()
        self.attention = MultiHeadAttention(d_model=d_model, heads=num_heads)
        self.dropout1 = nn.Dropout(p=drop_prob)
        self.dropout2 = nn.Dropout(p=drop_prob)
        self.normalisation1 = LayerNormalisation(parameter_shape=[d_model])
        self.normalisation2 = LayerNormalisation(parameter_shape=[d_model])
        self.feed_forward_network = PointFeedForward(
            d_model=d_model, hidden_ffn=ffn_hidden, dropout=drop_prob)

    def forward(self, y, self_attetion_mask):
        residual_x = y
        print("------- ATTENTION 1 ------")
        y = self.attention(y, mask=self_attetion_mask)
        print(f"X after attention:  {y}")
        print("------- DROPOUT 1 ------")
        y = self.dropout1(y)
        print(f"X after first dropout:  {y}")
        print("------- ADD AND LAYER NORMALIZATION 1 ------")
        y = self.normalisation1(y + residual_x)
        print(f"X after layer normalisation:  {y}")
        residual_x = y
        print("------- ATTENTION 2 ------")
        y = self.feed_forward_network(y)
        print(f"X after feed forward network:  {y}")
        print("------- DROPOUT 2 ------")
        y = self.dropout2(y)
        print(f"X after second dropout:  {y}")
        print("------- ADD AND LAYER NORMALIZATION 2 ------")
        y = self.normalisation2(y + residual_x)
        print(f"X after layer normalisation:  {y}")
        return y

class SequentialEncoder(nn.Sequential):
    def forward(self, *inputs):
        x, self_attention_mask = inputs
        for module in self._modules.values():
            x = module(x, self_attention_mask)  # 30 x 200 x 512
        return x


class Encoder(nn.Module):
    def __init__(self,
                 d_model,
                 ffn_hidden,
                 num_heads,
                 drop_prob,
                 num_layers,
                 max_sequence_length,
                 language_to_index,
                 START_TOKEN,
                 END_TOKEN,
                 PADDING_TOKEN):
        super().__init__()
        self.sentence_embedding = SentenceEmbedding(
            max_sequence_length, d_model, language_to_index, START_TOKEN, END_TOKEN, PADDING_TOKEN)
        self.layers = SequentialEncoder(*[EncoderLayer(d_model, ffn_hidden, num_heads, drop_prob)
                                          for _ in range(num_layers)])

    def forward(self, x, self_attention_mask, START_TOKEN, END_TOKEN):
        x = self.sentence_embedding(x, START_TOKEN, END_TOKEN)
        x = self.layers(x, self_attention_mask)
        return x    # 30 x 200 x 512

class DecoderLayer(nn.Module):
    "  DecoderLayer: Perform decoder layer operations on the input tensor    "

    def __init__(self, d_model, ffn_hidden, num_heads, drop_prob) -> None:
        super(DecoderLayer, self).__init__()
        self.attention = MultiHeadAttention(d_model=d_model, heads=num_heads)
        self.cross_attention = MultiHeadCrossAttention(d_model=d_model, num_heads=num_heads)
        self.dropout1 = nn.Dropout(p=drop_prob)
        self.dropout2 = nn.Dropout(p=drop_prob)
        self.dropout3 = nn.Dropout(p=drop_prob)
        self.normalisation1 = LayerNormalisation(parameter_shape=[d_model])
        self.normalisation2 = LayerNormalisation(parameter_shape=[d_model])
        self.normalisation3 = LayerNormalisation(parameter_shape=[d_model])
        self.feed_forward_network = PointFeedForward(
            d_model=d_model, hidden_ffn=ffn_hidden, dropout=drop_prob)

    def forward(self, x, y,  self_attention_mask, cross_attention_mask):
        residual_y = y
        print("------- MASKED SELF ATTENTION  ------")
        y = self.attention(y, mask=self_attention_mask)  # 30, 200, 512
        print(f"Y after attention:  {y}")
        print("------- DROPOUT 1 ------")
        y = self.dropout1(y)  # 30, 200, 512
        print(f"Y after first dropout:  {y}")
        print("------- ADD AND LAYER NORMALIZATION 1 ------")
        y = self.normalisation1(y + residual_y)  # 30, 200, 512
        print(f"Y after layer normalisation:  {y}")
        residual_y = y
        print("------- CROSS ATTENTION ------")
        y = self.cross_attention(x, y, mask=cross_attention_mask)
        print(f"Y after cross attention:  {y}")
        print("------- DROPOUT 2 ------")
        y = self.dropout2(y)
        print(f"Y after second dropout:  {y}")
        print("------- ADD AND LAYER NORMALIZATION 2 ------")
        y = self.normalisation2(y + residual_y)
        print(f"Y after layer normalisation:  {y}")
        residual_y = y
        print("------- ATTENTION 2 ------")
        y = self.feed_forward_network(y)
        print(f"Y after feed forward network:  {y}")
        print("------- DROPOUT 3 ------")
        y = self.dropout3(y)
        print(f"Y after third dropout:  {y}")
        print("------- ADD AND LAYER NORMALIZATION 3 ------")
        y = self.normalisation3(y + residual_y)
        print(f"Y after layer normalisation:  {y}")
        return y

class SequentialDecoder(nn.Sequential):
    def forward(self, *inputs):
        x, y, self_attention,cross_attention = inputs
        for module in self._modules.values():
            y = module(x, y,self_attention,cross_attention)  # 30 x 200 x 512
        return y


class Decoder(nn.Module):
    def __init__(self, d_model, ffn_hidden, num_heads, drop_prob, num_layers, max_sequence_length, language_to_index, START_TOKEN, END_TOKEN, PADDING_TOKEN):
        super().__init__()
        self.sentence_embedding = SentenceEmbedding(max_sequence_length, d_model, language_to_index, START_TOKEN, END_TOKEN, PADDING_TOKEN)
        self.layers = SequentialDecoder(*[DecoderLayer(d_model, ffn_hidden, num_heads, drop_prob) for _ in range(num_layers)])

    def forward(self, x, y, self_attention_mask, cross_attention_mask, START_TOKEN, END_TOKEN):
        y = self.sentence_embedding(y, START_TOKEN, END_TOKEN)
        y = self.layers(x, y, self_attention_mask, cross_attention_mask)
        return y   # 30 x 200 x 512


class Transformer(nn.Module):
    " Transformer: Perform transformer operations on the input tensor    "

    def __init__(self,
                 d_model,  # 512
                 ffn_hidden,  # 2048
                 num_heads,  # 8
                 drop_prob,  # 0.1
                 num_layers,  # 5
                 max_sequence_length,  # 200
                 language_vocab_size,
                 current_language_to_index,
                 target_language_to_index,
                 START_TOKEN,
                 END_TOKEN,
                 PADDING_TOKEN):
        super().__init__()
        self.encoder = Encoder(d_model, ffn_hidden,num_heads, drop_prob, num_layers, max_sequence_length,  current_language_to_index, START_TOKEN, END_TOKEN, PADDING_TOKEN)
        self.decoder = Decoder(d_model, ffn_hidden,num_heads, drop_prob, num_layers, max_sequence_length, target_language_to_index, START_TOKEN,END_TOKEN, PADDING_TOKEN)
        self.linear = nn.Linear(d_model, language_vocab_size)
        self.device = get_device()

    def forward(self,
                x,
                y,
                encoder_self_attention_mask,
                decoder_self_attention_mask,
                decoder_cross_attention_mask,
                encoder_start_token=False,
                encoder_end_token=False,
                decoder_start_token=False, 
                decoder_end_token=False
                ):
        x = self.encoder(x, encoder_self_attention_mask,encoder_start_token, encoder_end_token)
        out = self.decoder(x, y, decoder_self_attention_mask,decoder_cross_attention_mask,decoder_start_token, decoder_end_token)
        out = self.linear(out)
        return out
