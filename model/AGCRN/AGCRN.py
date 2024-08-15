import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from model.AGCRN.AGCRNCell import AGCRNCell

class AVWDCRNN(nn.Module):
    def __init__(self, node_num, dim_in, dim_out, cheb_k, embed_dim, num_layers=1):
        super(AVWDCRNN, self).__init__()
        assert num_layers >= 1, 'At least one DCRNN layer in the Encoder.'
        self.node_num = node_num
        self.input_dim = dim_in
        self.out_dim = dim_out
        self.num_layers = num_layers
        self.dcrnn_cells = nn.ModuleList()
        self.dcrnn_cells.append(AGCRNCell(node_num, dim_in, dim_out, cheb_k, embed_dim))
        for _ in range(1, num_layers):
            self.dcrnn_cells.append(AGCRNCell(node_num, dim_out, dim_out, cheb_k, embed_dim))

    def forward(self, x, init_state, node_embeddings):
        #shape of x: (B, T, N, D)
        #shape of init_state: (num_layers, B, N, hidden_dim)
        # assert x.shape[2] == self.node_num and x.shape[3] == self.input_dim
        seq_length = x.shape[1]
        current_inputs = x
        output_hidden = []
        for i in range(self.num_layers):
            state = init_state[i]
            inner_states = []
            for t in range(seq_length):
                state = self.dcrnn_cells[i](current_inputs[:, t, :, :], state, node_embeddings)
                inner_states.append(state)
            output_hidden.append(state)
            current_inputs = torch.stack(inner_states, dim=1)
        #current_inputs: the outputs of last layer: (B, T, N, hidden_dim)
        #output_hidden: the last state for each layer: (num_layers, B, N, hidden_dim)
        #last_state: (B, N, hidden_dim)
        return current_inputs, output_hidden

    def init_hidden(self, batch_size, node_dataset):
        init_states = []
        for i in range(self.num_layers):
            init_states.append(self.dcrnn_cells[i].init_hidden_state(batch_size, node_dataset))
        return torch.stack(init_states, dim=0)      #(num_layers, B, N, hidden_dim)

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
        # print(self.pe.shape)
        return self.pe[:, :x.size(2)].unsqueeze(1).expand_as(x)
        # return self.pe[:, :x.size(1)].unsqueeze(2).expand_as(x)

class PatchEmbedding_flow(nn.Module):
    def __init__(self, d_model, patch_len, stride, padding):
        super(PatchEmbedding_flow, self).__init__()
        # Patching
        self.patch_len = patch_len
        self.stride = stride
        self.value_embedding = nn.Linear(patch_len, d_model, bias=False)
        self.position_encoding = PositionalEncoding(d_model)

    def forward(self, x):
        # do patching
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

class AGCRN(nn.Module):
    def __init__(self, args, device, dim_in, dim_out):
        super(AGCRN, self).__init__()
        self.num_node = args.num_nodes
        self.input_dim = dim_in
        self.hidden_dim = args.rnn_units
        self.output_dim = dim_out
        self.horizon = args.horizon
        self.num_layers = args.num_layers

        self.patch_embedding_flow = PatchEmbedding_flow(
            self.hidden_dim, patch_len=12, stride=12, padding=0)

        self.node_embeddings = nn.Parameter(torch.randn(self.num_node, args.embed_dim), requires_grad=True)

        self.Lin_input = nn.Linear(self.hidden_dim, 1)

        # self.encoder = AVWDCRNN(args.num_nodes, self.input_dim, args.rnn_units, args.cheb_k,
        #                         args.embed_dim, args.num_layers)
        self.encoder = AVWDCRNN(args.num_nodes, self.hidden_dim, args.rnn_units, args.cheb_k,
                                args.embed_dim, args.num_layers)

        #predictor
        self.end_conv = nn.Conv2d(1, args.horizon * self.output_dim, kernel_size=(1, self.hidden_dim), bias=True)

    def forward(self, source, select_dataset):
        #source: B, T_1, N, D
        #target: B, T_2, N, D
        #supports = F.softmax(F.relu(torch.mm(self.nodevec1, self.nodevec1.transpose(0,1))), dim=1)

        # print(source.shape)
        init_state = self.encoder.init_hidden(source.shape[0], self.node_embeddings.shape[0])
        # source = self.Lin_input(source)
        # print('init_state', init_state.shape)

        source = self.patch_embedding_flow(source)

        output, _ = self.encoder(source, init_state, self.node_embeddings)  # B, T, N, hidden
        output = output[:, -1:, :, :]                                   #B, 1, N, hidden

        #CNN based predictor
        output = self.end_conv((output))                         #B, T*C, N, 1
        output = output.squeeze(-1).reshape(-1, self.horizon, self.output_dim, self.node_embeddings.shape[0])
        output = output.permute(0, 1, 3, 2)                             #B, T, N, C
        # print(output.shape)
        return output