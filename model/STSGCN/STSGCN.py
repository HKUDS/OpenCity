import torch
import torch.nn as nn
# from torch.nn.init import xavier_uniform
import numpy as np
from logging import getLogger
import torch.nn.functional as F
import math

class PositionEmbedding(nn.Module):
    def __init__(self, input_length, num_of_vertices, embedding_size, temporal=True, spatial=True):
        super(PositionEmbedding, self).__init__()
        self.input_length = input_length
        self.num_of_vertices = num_of_vertices
        self.embedding_size = embedding_size
        self.temporal = temporal
        self.spatial = spatial
        self.temporal_emb = torch.nn.Parameter(torch.zeros((1, input_length, 1, embedding_size)).to('cuda:0'))
        # shape is (1, T, 1, C)
        # xavier_uniform(self.temporal_emb)
        self.spatial_emb = torch.nn.Parameter(torch.zeros((1, 1, num_of_vertices, embedding_size)).to('cuda:0'))
        # shape is (1, 1, N, C)
        # xavier_uniform(self.spatial_emb)

    def forward(self, data):
        if self.temporal:
            data += self.temporal_emb
        if self.spatial:
            data += self.spatial_emb
        return data

class GcnOperation(nn.Module):
    def __init__(self, num_of_filter, num_of_features, num_of_vertices, activation):
        super().__init__()
        assert activation in {'GLU', 'relu'}

        self.num_of_filter = num_of_filter
        self.num_of_features = num_of_features
        self.num_of_vertices = num_of_vertices
        self.activation = activation
        if activation == "GLU":
            self.layer = nn.Linear(num_of_features, 2 * num_of_filter)
        elif activation == "relu":
            self.layer = nn.Linear(num_of_features, num_of_filter)

    def forward(self, data, adj):
        data = torch.matmul(adj, data)

        if self.activation == "GLU":
            data = self.layer(data)
            lhs, rhs = data.split(self.num_of_filter, -1)
            data = lhs * torch.sigmoid(rhs)

        elif self.activation == "relu":
            data = torch.relu(self.layer(data))

        return data


class Stsgcm(nn.Module):
    def __init__(self, filters, num_of_features, num_of_vertices, activation):
        super().__init__()
        self.filters = filters
        self.num_of_features = num_of_features
        self.num_of_vertices = num_of_vertices
        self.activation = activation
        self.layers = nn.ModuleList()
        for i in range(len(filters)):
            self.layers.append(GcnOperation(filters[i], num_of_features, num_of_vertices, activation))
            num_of_features = filters[i]

    def forward(self, data, adj):
        need_concat = []
        for i in range(len(self.layers)):
            data = self.layers[i](data, adj)
            need_concat.append(torch.transpose(data, 1, 0))

        need_concat = [
            torch.unsqueeze(
                i[self.num_of_vertices:2 * self.num_of_vertices, :, :],
                dim=0
            ) for i in need_concat
        ]

        return torch.max(torch.cat(need_concat, dim=0), dim=0)[0]


class STSGCL(nn.Module):
    def __init__(self, t, num_of_vertices, num_of_features, filters, module_type, activation, temporal_emb=True,
                 spatial_emb=True):
        super().__init__()
        assert module_type in {'sharing', 'individual'}
        self.T = t
        self.num_of_vertices = num_of_vertices
        self.num_of_features = num_of_features
        self.filters = filters
        self.module_type = module_type
        self.activation = activation
        self.temporal_emb = temporal_emb
        self.spatial_emb = spatial_emb

        if module_type == 'individual':
            self.layer = STSGCNLayerIndividual(
                t, num_of_vertices, num_of_features, filters,
                activation, temporal_emb, spatial_emb
            )
        else:
            self.layer = STSGCNLayerSharing(
                t, num_of_vertices, num_of_features, filters,
                activation, temporal_emb, spatial_emb
            )

    def forward(self, data, adj):
        return self.layer(data, adj)


class STSGCNLayerIndividual(nn.Module):
    def __init__(self, t, num_of_vertices, num_of_features, filters,
                 activation, temporal_emb=True, spatial_emb=True):
        super().__init__()
        self.T = t
        self.num_of_vertices = num_of_vertices
        self.num_of_features = num_of_features
        self.filters = filters
        self.activation = activation
        self.temporal_emb = temporal_emb
        self.spatial_emb = spatial_emb
        self.position_embedding = PositionEmbedding(t, num_of_vertices, num_of_features,
                                                    temporal_emb, spatial_emb)

        self.gcms = nn.ModuleList()
        for i in range(self.T - 2):
            self.gcms.append(Stsgcm(self.filters, self.num_of_features, self.num_of_vertices,
                                    activation=self.activation))

    def forward(self, data, adj):
        data = self.position_embedding(data)
        need_concat = []

        for i in range(self.T - 2):
            t = data[:, i:i + 3, :, :]
            # shape is (B, 3, N, C)

            t = torch.reshape(t, (-1, 3 * self.num_of_vertices, self.num_of_features))
            # shape is (B, 3N, C)

            t = self.gcms[i](t, adj)
            # shape is (N, B, C')

            t = torch.transpose(t, 0, 1)
            # shape is (B, N, C')

            need_concat.append(torch.unsqueeze(t, dim=1))
            # shape is (B, 1, N, C')

        return torch.cat(need_concat, dim=1)
        # shape is (B, T-2, N, C')


class STSGCNLayerSharing(nn.Module):
    def __init__(self, t, num_of_vertices, num_of_features, filters,
                 activation, temporal_emb=True, spatial_emb=True):
        super().__init__()
        self.T = t
        self.num_of_vertices = num_of_vertices
        self.num_of_features = num_of_features
        self.filters = filters
        self.activation = activation
        self.temporal_emb = temporal_emb
        self.spatial_emb = spatial_emb
        self.position_embedding = PositionEmbedding(t, num_of_vertices, num_of_features,
                                                    temporal_emb, spatial_emb)
        self.gcm = Stsgcm(self.filters, self.num_of_features, self.num_of_vertices,
                          activation=self.activation)

    def forward(self, data, adj):
        data = self.position_embedding(data)

        need_concat = []
        for i in range(self.T - 2):
            t = data[:, i:i + 3, :, :]
            # shape is (B, 3, N, C)

            t = torch.reshape(t, (-1, 3 * self.num_of_vertices, self.num_of_features))
            # shape is (B, 3N, C)

            need_concat.append(t)
            # shape is (B, 3N, C)

        t = torch.cat(need_concat, dim=0)
        # shape is ((T-2)*B, 3N, C)

        t = self.gcm(t, adj)
        # shape is (N, (T-2)*B, C')

        t = t.reshape((self.num_of_vertices, self.T - 2, -1, self.filters[-1]))
        # shape is (N, T - 2, B, C)

        return torch.transpose(t, 0, 2)
        # shape is (B, T - 2, N, C)


class OutputLayer(nn.Module):
    def __init__(self, num_of_vertices, input_length, num_of_features,
                 num_of_filters=128, predict_length=12, output_dim=1):
        super().__init__()
        self.num_of_vertices = num_of_vertices
        self.input_length = input_length
        self.num_of_features = num_of_features
        self.num_of_filters = num_of_filters
        self.predict_length = predict_length
        self.output_dim = output_dim
        self.hidden_layer = nn.Linear(self.input_length * self.num_of_features, self.num_of_filters)
        self.ouput_layer = nn.Linear(self.num_of_filters, self.predict_length * self.output_dim)

    def forward(self, data):
        data = torch.transpose(data, 1, 2)

        # (B, N, T * C)
        data = torch.reshape(
            data, (-1, self.num_of_vertices, self.input_length * self.num_of_features)
        )

        # (B, N, C')
        # data = self.hidden_layer(data)
        # data = torch.relu(self.hidden_layer(data))
        data = self.hidden_layer(data)

        # (B, N, T' * C_out) -> (B, N, T', C_out)
        data = self.ouput_layer(data).reshape(
            (-1, self.num_of_vertices, self.predict_length, self.output_dim)
        )

        # (B, T', N, C_out)
        data = data.permute(0, 2, 1, 3)

        return data

class PositionalEncoding(nn.Module):
    def __init__(self, embed_dim, max_len=100):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, embed_dim).float()
        pe.require_grad = False

        position = torch.arange(0, max_len).float().unsqueeze(1)
        div_term = (torch.arange(0, embed_dim, 2).float() * -(math.log(10000.0) / embed_dim)).exp()

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return self.pe[:, :x.size(2)].unsqueeze(1).expand_as(x)

class PatchEmbedding_flow(nn.Module):
    def __init__(self, d_model, patch_len, stride, padding):
        super(PatchEmbedding_flow, self).__init__()
        # Patching
        self.patch_len = patch_len
        self.stride = stride
        self.value_embedding = nn.Linear(patch_len, d_model, bias=False)
        self.position_encoding = PositionalEncoding(d_model)

    def forward(self, x):
        x = x.squeeze(-1).permute(0, 2, 1)
        if x.shape[-1] == 288:
            x = x.unfold(dimension=-1, size=self.patch_len, step=self.stride)
        else:
            gap = 288 // x.shape[-1]
            x = x.unfold(dimension=-1, size=self.patch_len//gap, step=self.stride//gap)
            x = F.pad(x, (0, (self.patch_len - self.patch_len//gap)))
        x = self.value_embedding(x)
        x = x + self.position_encoding(x)
        x = x.permute(0, 2, 1, 3)
        return x

def construct_adj(a, steps):
    n = len(a)
    adj = np.zeros([n * steps] * 2)

    for i in range(steps):
        adj[i * n: (i + 1) * n, i * n: (i + 1) * n] = a

    # 实际就是加了相邻两个时间步节点到自身的边
    for i in range(n):
        for k in range(steps - 1):
            adj[k * n + i, (k + 1) * n + i] = 1
            adj[(k + 1) * n + i, k * n + i] = 1

    for i in range(len(adj)):
        adj[i, i] = 1

    return adj


class STSGCN(nn.Module):
    def __init__(self, args_predcitor, device, dim_in, dim_out):
        super(STSGCN, self).__init__()
        self.num_nodes = args_predcitor.num_nodes
        self.feature_dim = args_predcitor.feature_dim
        self.output_dim = dim_out
        self._logger = getLogger()

        self.module_type = args_predcitor.module_type
        self.activation = args_predcitor.activation
        self.temporal_emb = args_predcitor.temporal_emb
        self.spatial_emb = args_predcitor.spatial_emb
        self.use_mask = args_predcitor.use_mask
        self.device = device
        self.input_window = args_predcitor.input_window
        self.output_window = args_predcitor.output_window
        self.rho = args_predcitor.rho
        self.num_of_vertices = self.num_nodes
        self.input_length = self.input_window
        self.predict_length = self.output_window

        self.adj = args_predcitor.A
        self.adj = construct_adj(self.adj, args_predcitor.steps)
        self.adj = torch.tensor(self.adj, requires_grad=False, dtype=torch.float32).to(self.device)

        self.patch_embedding_flow = PatchEmbedding_flow(
            self.feature_dim, patch_len=12, stride=12, padding=0)

        if self.use_mask:
            self._logger.warning('You use mask matrix, please make sure you set '
                                 '`loss.backward(retain_graph=True)` to replace the '
                                 '`loss.backward()` in traffic_state_executor.py!')
            self.mask = torch.nn.Parameter(torch.tensor((self.adj != 0) + 0.0).to(self.device))
            self.adj = self.mask * self.adj

        self.embedding_dim = self.feature_dim
        self.num_of_features = dim_in
        first_layer_embedding_size = args_predcitor.first_layer_embedding_size
        if first_layer_embedding_size:
            self.first_layer_embedding = nn.Linear(self.num_of_features, first_layer_embedding_size)
            self.num_of_features = first_layer_embedding_size
        else:
            self.first_layer_embedding = None

        self.filter_list = args_predcitor.filter_list
        self.stsgcl_layers = nn.ModuleList()
        for idx, filters in enumerate(self.filter_list):
            # if self.input_length <= 0:
                # break
            self.stsgcl_layers.append(STSGCL(self.input_length, self.num_of_vertices,
                                             self.num_of_features, filters, self.module_type,
                                             activation=self.activation,
                                             temporal_emb=self.temporal_emb,
                                             spatial_emb=self.spatial_emb))
            self.input_length -= 2
            self.num_of_features = filters[-1]

        # self.outputs = nn.ModuleList()
        # for i in range(self.predict_length):
        #     self.outputs.append(OutputLayer(self.num_of_vertices, self.input_length, self.num_of_features,
        #                                     num_of_filters=128, predict_length=1, output_dim=self.output_dim))
        self.outputs_layer = OutputLayer(self.num_of_vertices, self.input_length, self.num_of_features,
                                        num_of_filters=128, predict_length=self.predict_length, output_dim=self.output_dim)

    def forward(self, source, select_dataset):
        data = source
        # data.shape = (B:batch_size, T:input_length, N:vertical_num, C:feature_num)
        if data.shape[-1] > self.embedding_dim:
            data = data[:, :, :, 0:self.embedding_dim]

        data = torch.relu(self.patch_embedding_flow(data))

        # data.shape = (B, T, N, C:embedding_feature_num)
        # if self.first_layer_embedding:
        #     data = torch.relu(self.first_layer_embedding(data))

        for stsgcl_layer in self.stsgcl_layers:
            data = stsgcl_layer(data.clone(), self.adj)

        # need_concat = []
        # for output_layer in self.outputs:
        #     need_concat.append(output_layer(data))
        #     print('output_layer', output_layer(data).shape)
        # data = torch.cat(need_concat, dim=1)

        data = self.outputs_layer(data)
        # print('output', data.shape)

        return data
