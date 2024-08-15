import torch
import torch.nn as nn
import torch.nn.functional as F

class TemporalAttention(nn.Module):
    def __init__(self, in_dim, num_nodes=None, cut_size=0):
        super(TemporalAttention, self).__init__()
        self.K = 8

        if in_dim % self.K != 0:
            raise Exception('Hidden size is not divisible by the number of attention heads')

        self.head_size = int(in_dim // self.K)
        self.key_proj = LinearCustom()
        self.value_proj = LinearCustom()

        self.projection1 = nn.Linear(in_dim,in_dim)
        self.projection2 = nn.Linear(in_dim,in_dim)

    def forward(self, query, key, value, parameters):
        batch_size = query.shape[0]

        # [batch_size, num_step, N, K * head_size]
        key = self.key_proj(key, parameters[0])
        value = self.value_proj(value, parameters[1])

        # [K * batch_size, num_step, N, head_size]
        query = torch.cat(torch.split(query, self.head_size, dim=-1), dim=0)
        key = torch.cat(torch.split(key, self.head_size, dim=-1), dim=0)
        value = torch.cat(torch.split(value, self.head_size, dim=-1), dim=0)

        # query: [K * batch_size, N, 1, head_size]
        # key:   [K * batch_size, N, head_size, num_step]
        # value: [K * batch_size, N, num_step, head_size]
        query = query.permute((0, 2, 1, 3))
        key = key.permute((0, 2, 3, 1))
        value = value.permute((0, 2, 1, 3))

        attention = torch.matmul(query, key)  # [K * batch_size, N, num_step, num_step]
        attention /= (self.head_size ** 0.5)

        # normalize the attention scores
        # attention = self.mask * attention
        attention = F.softmax(attention, dim=-1)

        x = torch.matmul(attention, value)  # [batch_size * head_size, num_step, N, K]
        x = x.permute((0, 2, 1, 3))
        x = torch.cat(torch.split(x, batch_size, dim=0), dim=-1)

        # projection
        x = self.projection1(x)
        x = F.tanh(x)
        x = self.projection2(x)
        return x


class SpatialAttention(nn.Module):
    def __init__(self, in_dim, support=None, num_nodes=None):
        super(SpatialAttention, self).__init__()
        self.support = support
        self.K = 8

        if in_dim % self.K != 0:
            raise Exception('Hidden size is not divisible by the number of attention heads')

        self.head_size = int(in_dim // self.K)
        self.linear = LinearCustom()
        self.projection1 = nn.Linear(in_dim, in_dim)
        self.projection2 = nn.Linear(in_dim, in_dim)

    def forward(self, x, parameters):
        batch_size = x.shape[0]
        # [batch_size, 1, N, K * head_size]
        # query = self.linear(x, parameters[2])
        key = self.linear(x, parameters[0])
        value = self.linear(x, parameters[1])

        # [K * batch_size, num_step, N, head_size]
        query = torch.cat(torch.split(x, self.head_size, dim=-1), dim=0)
        key = torch.cat(torch.split(key, self.head_size, dim=-1), dim=0)
        value = torch.cat(torch.split(value, self.head_size, dim=-1), dim=0)

        attention = torch.matmul(query, key.transpose(2, 3))  # [K * batch_size, N, num_step, num_step]
        attention /= (self.head_size ** 0.5)

        attention = F.softmax(attention, dim=-1)
        x = torch.matmul(attention, value)  # [batch_size * head_size, num_step, N, K]
        x = torch.cat(torch.split(x, batch_size, dim=0), dim=-1)

        # projection
        x = self.projection1(x)
        x = F.relu(x)
        x = self.projection2(x)
        return x


class LinearCustom(nn.Module):

    def __init__(self):
        super(LinearCustom, self).__init__()

    def forward(self, inputs, parameters):
        weights, biases = parameters[0], parameters[1]
        if len(weights.shape) > 3:
            return torch.matmul(inputs.unsqueeze(-2), weights.unsqueeze(1).repeat(1, inputs.shape[1], 1, 1, 1)).squeeze(
                -2) + biases.unsqueeze(1).repeat(1, inputs.shape[1], 1, 1)
        return torch.matmul(inputs, weights) + biases
