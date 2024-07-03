import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor, split, matmul, ones, zeros, sqrt

# scaled dot-product attention


def scaled_dot_product_attention(Q, K, V, mask=None):
    d_k = Q.size(-1)
    scores = matmul(Q, K.transpose(-2, -1)) / math.sqrt(d_k)
    if mask is not None:
        scores = scores.masked_fill(mask == 0, float('-inf'))
    attention = F.softmax(scores, dim=-1)
    values = matmul(attention, V)
    return values, attention


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model: int, heads: int):
        super().__init__()
        self.d_model = d_model  # 512
        self.heads = heads  # 8
        self.head_dim = d_model//heads
        # to perform the operation parallely we are stacking up the vectors
        self.qkv_layer = nn.Linear(d_model, 3*d_model)
        print(self.qkv_layer.weight.data.shape, self.qkv_layer.bias.data.shape)
        self.linear_layer = nn.Linear(d_model, d_model)
        print(self.linear_layer.weight.data.shape,
              self.linear_layer.bias.data.shape)

    def forward(self, x: Tensor, mask: Tensor = None):
        batch_size, sequence_length, d_model = x.size()
        print(f"Batch Size:  {batch_size}")
        print(f"Sequence Length:  {sequence_length}")
        print(f"Model Size:  {d_model}")
        qkv: Tensor = self.qkv_layer(x)
        print(f"qkv.size(): {qkv.size()}")
        qkv = qkv.reshape(batch_size, sequence_length,
                          self.heads, 3*self.head_dim)
        print(f"qkv.size(): {qkv.size()}")
        qkv = qkv.permute(0, 2, 1, 3)
        print(f"qkv.size() after permute: {qkv.size()}")
        q, k, v = qkv.chunk(3, dim=-1)
        print(f"Q Size:  {q.size()}")
        print(f"K Size:  {k.size()}")
        print(f"V Size:  {v.size()}")
        values, attention = scaled_dot_product_attention(q, k, v, mask)
        values = values.reshape(
            batch_size, sequence_length, self.heads*self.head_dim)
        out = self.linear_layer(values)
        return out


class LayerNormalisation(nn.Module):
    def __init__(self, parameter_shape, epsilon=1e-5) -> None:
        super(LayerNormalisation, self).__init__()
        # standard deviation for values
        self.Gamma = nn.Parameter(ones(parameter_shape))
        self.Beta = nn.Parameter(zeros(parameter_shape))  # mean for values
        self.parameter_shape = parameter_shape  # 512
        self.epsilon = epsilon

    def forward(self, inputs: Tensor) -> Tensor:
        # [-1] perform layer normalisationonly at last dimension
        dimensions = [-(i+1) for i in range(len(self.parameter_shape))]
        mean = inputs.mean(dimensions, keepdim=True)
        print(f"Mean Size:  {mean.size()}")
        variance = (inputs-mean).pow(2).mean(dimensions, keepdim=True)
        print(f"Variance Size:  {variance.size()}")
        std = sqrt(variance+self.epsilon)
        print(f" Standard Deviation Size:  {std.size()}")
        y = (inputs-mean)/std
        print(f"Y Size:  {y.size()}")
        output = self.Gamma*y+self.Beta
        print(f"Output Size:  {output.size()}")
        return output


class PointFeedForward(nn.Module):
    def __init__(self, d_model, hidden_ffn, dropout=0.1) -> None:
        super(PointFeedForward, self).__init__()
        self.linear1 = nn.Linear(d_model, hidden_ffn)
        self.linear2 = nn.Linear(hidden_ffn, d_model)
        self.activation = nn.ReLU()
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x):
        x = self.linear1(x)
        print(f"x after first linear layer: {x.size()}")

        x = self.activation(x)
        print(f"x after activation: {x.size()}")

        x = self.dropout(x)
        print(f"x after dropout: {x.size()}")

        x = self.linear2(x)
        print(f"x after 2nd linear layer: {x.size()}")

        return x


class EncoderLayer(nn.Module):

    def __init__(self, d_model, ffn_hidden, num_heads, drop_prob):
        super(EncoderLayer, self).__init__()
        self.attention = MultiHeadAttention(d_model=d_model, heads=num_heads)
        self.dropout1 = nn.Dropout(p=drop_prob)
        self.dropout2 = nn.Dropout(p=drop_prob)
        self.normalisation1 = LayerNormalisation(parameter_shape=[d_model])
        self.normalisation2 = LayerNormalisation(parameter_shape=[d_model])
        self.feed_forward_network = PointFeedForward(
            d_model=d_model, hidden_ffn=ffn_hidden, dropout=drop_prob)

    def forward(self, x):
        residual_x = x
        print("------- ATTENTION 1 ------")
        x = self.attention(x, mask=None)
        print(f"X after attention:  {x}")
        print("------- DROPOUT 1 ------")
        x = self.dropout1(x)
        print(f"X after first dropout:  {x}")
        print("------- ADD AND LAYER NORMALIZATION 1 ------")
        x = self.normalisation1(x + residual_x)
        print(f"X after layer normalisation:  {x}")
        residual_x = x
        print("------- ATTENTION 2 ------")
        x = self.feed_forward_network(x)
        print(f"X after feed forward network:  {x}")
        print("------- DROPOUT 2 ------")
        x = self.dropout2(x)
        print(f"X after second dropout:  {x}")
        print("------- ADD AND LAYER NORMALIZATION 2 ------")
        x = self.normalisation2(x + residual_x)
        print(f"X after layer normalisation:  {x}")
        return x


class Encoder(nn.Module):
    def __init__(self, d_model, ffn_hidden, num_heads, drop_prob, num_layers):
        super().__init__()
        self.layers = nn.Sequential(*[EncoderLayer(d_model, ffn_hidden, num_heads, drop_prob)
                                      for _ in range(num_layers)])

    def forward(self, x):
        x = self.layers(x)
        return x


d_model = 512
num_heads = 8
drop_prob = 0.1
batch_size = 30
max_sequence_length = 200
ffn_hidden = 2048
num_layers = 5

encoder = Encoder(d_model, ffn_hidden, num_heads, drop_prob, num_layers)


# includes positional encoding
x = torch.randn((batch_size, max_sequence_length, d_model))
out = encoder(x)
