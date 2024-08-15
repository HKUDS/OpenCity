import numpy as np
import torch
import torch.nn as nn
import math
import torch.nn.functional as F
from .gmsdr_cell import GMSDRCell

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


class Seq2SeqAttrs:
    def __init__(self, adj_mx, args):
        self.adj_mx = adj_mx
        self.max_diffusion_step = args.max_diffusion_step
        self.cl_decay_steps = args.cl_decay_steps
        self.filter_type = args.filter_type
        self.num_nodes = args.num_nodes
        self.num_rnn_layers = args.num_rnn_layers
        self.rnn_units = args.rnn_units
        self.hidden_state_size = self.num_nodes * self.rnn_units
        self.pre_k = args.pre_k
        self.pre_v = args.pre_v
        self.input_dim = args.input_dim
        self.output_dim = args.output_dim


class EncoderModel(nn.Module, Seq2SeqAttrs):
    def __init__(self, adj_mx, args, device):
        nn.Module.__init__(self)
        Seq2SeqAttrs.__init__(self, adj_mx, args)
        self.input_dim = args.input_dim
        self.seq_len = args.seq_len  # for the encoder
        self.mlp = nn.Linear(self.input_dim, self.rnn_units)
        self.gmsdr_layers = nn.ModuleList(
            [GMSDRCell(device, self.rnn_units, self.input_dim, adj_mx.numpy(), self.max_diffusion_step, self.num_nodes, self.pre_k, self.pre_v,
                       filter_type=self.filter_type) for _ in range(self.num_rnn_layers)])

    def forward(self, inputs, hx_k):
        """
        Encoder forward pass.

        :param inputs: shape (batch_size, self.num_nodes * self.input_dim)
        :param hx_k: (num_layers, batch_size, pre_k, self.num_nodes, self.rnn_units)
               optional, zeros if not provided
        :return: output: # shape (batch_size, self.hidden_state_size)
                 hx_k # shape (num_layers, batch_size, pre_k, self.num_nodes, self.rnn_units)
                 (lower indices mean lower layers)
        """
        hx_ks = []
        # batch = inputs.shape[0]
        # print(batch, self.num_nodes, self.input_dim, inputs.shape)
        # x = inputs.reshape(batch, self.num_nodes, self.input_dim)
        # output = self.mlp(x).view(batch, -1)
        output = inputs
        for layer_num, dcgru_layer in enumerate(self.gmsdr_layers):
            next_hidden_state, new_hx_k = dcgru_layer(output, hx_k[layer_num])
            hx_ks.append(new_hx_k)
            output = next_hidden_state
        return output, torch.stack(hx_ks)


class DecoderModel(nn.Module, Seq2SeqAttrs):
    def __init__(self, adj_mx, args, device):
        nn.Module.__init__(self)
        Seq2SeqAttrs.__init__(self, adj_mx, args)
        self.output_dim = args.output_dim
        self.horizon = args.horizon  # for the decoder
        self.projection_layer = nn.Linear(self.rnn_units, self.output_dim)
        self.gmsdr_layers = nn.ModuleList(
            [GMSDRCell(device, self.rnn_units, self.rnn_units, adj_mx.numpy(), self.max_diffusion_step, self.num_nodes, self.pre_k, self.pre_v,
                       filter_type=self.filter_type) for _ in range(self.num_rnn_layers)])

    def forward(self, inputs, hx_k):
        """
        Decoder forward pass.

        :param inputs: shape (batch_size, self.num_nodes * self.output_dim)
        :param hx_k: (num_layers, batch_size, pre_k, num_nodes, rnn_units)
               optional, zeros if not provided
        :return: output: # shape (batch_size, self.num_nodes * self.output_dim)
                 hidden_state # shape (num_layers, batch_size, self.hidden_state_size)
                 (lower indices mean lower layers)
        """
        hx_ks = []
        output = inputs
        for layer_num, dcgru_layer in enumerate(self.gmsdr_layers):
            next_hidden_state, new_hx_k = dcgru_layer(output, hx_k[layer_num])
            hx_ks.append(new_hx_k)
            output = next_hidden_state

        # projected = self.projection_layer(output.view(-1, self.rnn_units))
        # output = projected.view(-1, self.num_nodes * self.output_dim)

        return output, torch.stack(hx_ks)


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
    def __init__(self, d_model, patch_len, stride, padding, num_for_predict=None):
        super(PatchEmbedding_flow, self).__init__()
        # Patching
        self.patch_len = patch_len
        self.stride = stride
        self.num_for_predict = num_for_predict
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

class GMSDRModel(nn.Module, Seq2SeqAttrs):
    def __init__(self, args, device):
        super().__init__()
        Seq2SeqAttrs.__init__(self, args.adj_mx, args)
        self.device = device
        self.patch_embedding_flow = PatchEmbedding_flow(
            args.rnn_units, patch_len=12, stride=12, padding=0)
        self.encoder_model = EncoderModel(args.adj_mx, args, self.device)
        self.decoder_model = DecoderModel(args.adj_mx, args, self.device)
        self.cl_decay_steps = args.cl_decay_steps
        self.use_curriculum_learning = args.use_curriculum_learning
        self.output_len = args.output_len
        # self._logger = logger
        self.out = nn.Linear(self.rnn_units*args.seq_len, self.decoder_model.output_dim*args.output_len)

    def _compute_sampling_threshold(self, batches_seen):
        return self.cl_decay_steps / (
                self.cl_decay_steps + np.exp(batches_seen / self.cl_decay_steps))

    def encoder(self, inputs):
        """
        encoder forward pass on t time steps
        :param inputs: shape (seq_len, batch_size, num_sensor * input_dim)
        :return: hx_k: (num_layers, batch_size, pre_k, num_sensor, rnn_units)
        """
        hx_k = torch.zeros(self.num_rnn_layers, inputs.shape[1], self.pre_k, self.num_nodes, self.rnn_units,
                           device=self.device)
        outputs = []
        for t in range(self.encoder_model.seq_len):
            output, hx_k = self.encoder_model(inputs[t], hx_k)
            outputs.append(output)
        return torch.stack(outputs), hx_k

    def decoder(self, inputs, hx_k, labels=None, batches_seen=None):
        """
        Decoder forward pass
        :param inputs: (seq_len, batch_size, num_sensor * rnn_units)
        :param hx_k: (num_layers, batch_size, pre_k, num_sensor, rnn_units)
        :param labels: (self.horizon, batch_size, self.num_nodes * self.output_dim) [optional, not exist for inference]
        :param batches_seen: global step [optional, not exist for inference]
        :return: output: (self.horizon, batch_size, self.num_nodes * self.output_dim)
        """
        decoder_hx_k = hx_k
        decoder_input = inputs

        outputs = []
        for t in range(self.decoder_model.horizon):
            decoder_output, decoder_hx_k = self.decoder_model(decoder_input[t],
                                                              decoder_hx_k)
            outputs.append(decoder_output)
        outputs = torch.stack(outputs)
        return outputs


    def forward(self, inputs, select_dataset, labels=None, batches_seen=None):
        """
        seq2seq forward pass
        :param inputs: shape (seq_len, batch_size, num_sensor * input_dim)
        :param labels: shape (horizon, batch_size, num_sensor * output)
        :param batches_seen: batches seen till now
        :return: output: (self.horizon, batch_size, self.num_nodes * self.output_dim)
        """
        # print('input', inputs.shape)
        inputs = self.patch_embedding_flow(inputs)
        # print('input', inputs.shape)
        inputs = inputs.transpose(0, 1)
        encoder_outputs, hx_k = self.encoder(inputs)
        # self._logger.debug("Encoder complete, starting decoder")
        # print(encoder_outputs.shape, hx_k.shape)
        outputs = self.decoder(encoder_outputs, hx_k, labels, batches_seen=batches_seen)
        # print('out', outputs.shape)
        bs, ts = outputs.shape[1], outputs.shape[0]
        out = outputs.transpose(1, 0).reshape(bs, ts, -1, self.rnn_units)
        out = out.transpose(1, 2).reshape(bs, -1, ts*self.rnn_units)

        out = self.out(out)
        outputs = out.reshape(bs, -1, self.output_len, self.output_dim).transpose(1, 2)
        # print(outputs.shape)
        # self._logger.debug("Decoder complete")
        # if batches_seen == 0:
        #     self._logger.info(
        #         "Total trainable parameters {}".format(count_parameters(self))
        #     )

        # if self.decoder_model.output_dim == 1:
        #     outputs = outputs.transpose(1, 0).unsqueeze(-1)
        # else:
        #     time_step, batch_size = outputs.shape[0], outputs.shape[1]
        #     outputs = outputs.transpose(1, 0).unsqueeze(-1).reshape(batch_size, time_step, -1, self.decoder_model.output_dim)
        return outputs
