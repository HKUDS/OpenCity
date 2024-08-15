import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
from functools import partial



def drop_path(x, drop_prob=0., training=False):
    if drop_prob == 0. or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)
    random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
    random_tensor.floor_()
    output = x.div(keep_prob) * random_tensor
    return output

class LlamaRMSNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-6):
        """
        LlamaRMSNorm is equivalent to T5LayerNorm
        """
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states):
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        return self.weight * hidden_states

class TokenEmbedding(nn.Module):
    def __init__(self, input_dim, embed_dim, norm_layer=None):
        super().__init__()
        self.token_embed = nn.Linear(input_dim, embed_dim, bias=True)
        self.norm = norm_layer(embed_dim) if norm_layer is not None else nn.Identity()

    def forward(self, x):
        x = self.token_embed(x)
        x = self.norm(x)
        return x

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

class LaplacianPE(nn.Module):
    def __init__(self, lape_dim, embed_dim):
        super().__init__()
        self.embedding_lap_pos_enc = nn.Linear(lape_dim, embed_dim)

    def forward(self, lap_mx):
        lap_pos_enc = self.embedding_lap_pos_enc(lap_mx).unsqueeze(0).unsqueeze(0)
        return lap_pos_enc


class DataEmbedding(nn.Module):
    def __init__(
        self, feature_dim, embed_dim, lape_dim, adj_mx, drop=0.,
        add_time_in_day=False, add_day_in_week=False, device=torch.device('cpu'),
    ):
        super().__init__()

        self.add_time_in_day = add_time_in_day
        self.add_day_in_week = add_day_in_week

        self.device = device
        self.embed_dim = embed_dim
        self.feature_dim = feature_dim
        self.value_embedding = TokenEmbedding(feature_dim, embed_dim)
        self.position_encoding = PositionalEncoding(embed_dim)
        self.spatial_embedding = LaplacianPE(lape_dim, embed_dim)
        self.dropout = nn.Dropout(drop)
        self.act = nn.LeakyReLU()

    def forward(self, x, lap_mx, tdh, dwh):
        # origin_x = x.transpose(1, 3)
        x = self.value_embedding(x[:, :, :, :self.feature_dim])
        x += self.position_encoding(x)
        x = x + tdh + dwh
        # print(origin_x.shape, lap_mx.shape)
        x += self.spatial_embedding(lap_mx.to(self.device))
        x = self.dropout(x)
        return x


class DropPath(nn.Module):
    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training)


class Chomp2d(nn.Module):
    def __init__(self, chomp_size):
        super(Chomp2d, self).__init__()
        self.chomp_size = chomp_size

    def forward(self, x):
        return x[:, :, :x.shape[2] - self.chomp_size, :].contiguous()


class STSelfAttention(nn.Module):
    def __init__(
        self, dim, s_attn_size, t_attn_size, geo_num_heads=4, sem_num_heads=4, tc_num_heads=4, t_num_heads=4, qkv_bias=False,
        attn_drop=0., proj_drop=0., device=torch.device('cpu'), output_dim=1,
    ):
        super().__init__()
        assert dim % (geo_num_heads + sem_num_heads + t_num_heads + tc_num_heads) == 0
        self.geo_num_heads = geo_num_heads
        self.sem_num_heads = sem_num_heads
        self.t_num_heads = t_num_heads
        self.head_dim = dim // (geo_num_heads + sem_num_heads + t_num_heads + tc_num_heads)
        self.scale = self.head_dim ** -0.5
        self.device = device
        self.s_attn_size = s_attn_size
        self.t_attn_size = t_attn_size
        self.geo_ratio = geo_num_heads / (geo_num_heads + sem_num_heads + tc_num_heads + t_num_heads)
        self.sem_ratio = sem_num_heads / (geo_num_heads + sem_num_heads + tc_num_heads + t_num_heads)
        self.tc_ratio = tc_num_heads / (geo_num_heads + sem_num_heads + tc_num_heads + t_num_heads)
        self.t_ratio = 1 - self.geo_ratio - self.sem_ratio - self.tc_ratio
        self.output_dim = output_dim

        # self.pattern_q_linears = nn.ModuleList([
        #     nn.Linear(dim, int(dim * self.geo_ratio)) for _ in range(output_dim)
        # ])
        # self.pattern_k_linears = nn.ModuleList([
        #     nn.Linear(dim, int(dim * self.geo_ratio)) for _ in range(output_dim)
        # ])
        # self.pattern_v_linears = nn.ModuleList([
        #     nn.Linear(dim, int(dim * self.geo_ratio)) for _ in range(output_dim)
        # ])

        self.geo_q_conv = nn.Linear(dim, int(dim * self.geo_ratio), bias=qkv_bias)
        self.geo_k_conv = nn.Linear(dim, int(dim * self.geo_ratio), bias=qkv_bias)
        self.geo_v_conv = nn.Linear(dim, int(dim * self.geo_ratio), bias=qkv_bias)
        self.geo_attn_drop = nn.Dropout(attn_drop)

        self.sem_q_conv = nn.Linear(dim, int(dim * self.sem_ratio), bias=qkv_bias)
        self.sem_k_conv = nn.Linear(dim, int(dim * self.sem_ratio), bias=qkv_bias)
        self.sem_v_conv = nn.Linear(dim, int(dim * self.sem_ratio), bias=qkv_bias)
        self.sem_attn_drop = nn.Dropout(attn_drop)

        self.t_q_conv = nn.Linear(dim, int(dim * self.t_ratio), bias=qkv_bias)
        self.t_k_conv = nn.Linear(dim, int(dim * self.t_ratio), bias=qkv_bias)
        self.t_v_conv = nn.Linear(dim, int(dim * self.t_ratio), bias=qkv_bias)
        self.t_attn_drop = nn.Dropout(attn_drop)


        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x, TH, geo_mask=None, sem_mask=None):
        B, T, N, D = x.shape
        t_q = self.t_q_conv(x)
        t_k = self.t_k_conv(x)
        t_v = self.t_v_conv(x)
        t_q = t_q.reshape(B, N, T, self.t_num_heads, self.head_dim).permute(0, 1, 3, 2, 4)
        t_k = t_k.reshape(B, N, T, self.t_num_heads, self.head_dim).permute(0, 1, 3, 2, 4)
        t_v = t_v.reshape(B, N, T, self.t_num_heads, self.head_dim).permute(0, 1, 3, 2, 4)
        t_attn = (t_q @ t_k.transpose(-2, -1)) * self.scale
        t_attn = t_attn.softmax(dim=-1)
        t_attn = self.t_attn_drop(t_attn)
        t_x = (t_attn @ t_v).transpose(2, 3).reshape(B, N, T, int(D * self.t_ratio)).transpose(1, 2)

        geo_q = self.geo_q_conv(x)
        geo_k = self.geo_k_conv(x)
        # for i in range(self.output_dim):
        #     pattern_q = self.pattern_q_linears[i](x_patterns[..., i])
        #     pattern_k = self.pattern_k_linears[i](pattern_keys[..., i])
        #     pattern_v = self.pattern_v_linears[i](pattern_keys[..., i])
        #     pattern_attn = (pattern_q @ pattern_k.transpose(-2, -1)) * self.scale
        #     pattern_attn = pattern_attn.softmax(dim=-1)
        #     geo_k += pattern_attn @ pattern_v
        geo_v = self.geo_v_conv(x)
        geo_q = geo_q.reshape(B, T, N, self.geo_num_heads, self.head_dim).permute(0, 1, 3, 2, 4)
        geo_k = geo_k.reshape(B, T, N, self.geo_num_heads, self.head_dim).permute(0, 1, 3, 2, 4)
        geo_v = geo_v.reshape(B, T, N, self.geo_num_heads, self.head_dim).permute(0, 1, 3, 2, 4)
        geo_attn = (geo_q @ geo_k.transpose(-2, -1)) * self.scale
        if geo_mask is not None:
            geo_attn.masked_fill_(geo_mask, float('-inf'))
        geo_attn = geo_attn.softmax(dim=-1)
        geo_attn = self.geo_attn_drop(geo_attn)
        geo_x = (geo_attn @ geo_v).transpose(2, 3).reshape(B, T, N, int(D * self.geo_ratio))

        sem_q = self.sem_q_conv(x)
        sem_k = self.sem_k_conv(x)
        sem_v = self.sem_v_conv(x)
        sem_q = sem_q.reshape(B, T, N, self.sem_num_heads, self.head_dim).permute(0, 1, 3, 2, 4)
        sem_k = sem_k.reshape(B, T, N, self.sem_num_heads, self.head_dim).permute(0, 1, 3, 2, 4)
        sem_v = sem_v.reshape(B, T, N, self.sem_num_heads, self.head_dim).permute(0, 1, 3, 2, 4)
        sem_attn = (sem_q @ sem_k.transpose(-2, -1)) * self.scale
        if sem_mask is not None:
            sem_attn.masked_fill_(sem_mask, float('-inf'))
        sem_attn = sem_attn.softmax(dim=-1)
        sem_attn = self.sem_attn_drop(sem_attn)
        sem_x = (sem_attn @ sem_v).transpose(2, 3).reshape(B, T, N, int(D * self.sem_ratio))

        x = self.proj(torch.cat([t_x, geo_x, sem_x], dim=-1))
        x = self.proj_drop(x)
        return x


class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class TemporalSelfAttention(nn.Module):
    def __init__(
        self, dim, dim_out, t_attn_size, t_num_heads=6, qkv_bias=False,
        attn_drop=0., proj_drop=0., device=torch.device('cpu'),
    ):
        super().__init__()
        assert dim % t_num_heads == 0
        self.t_num_heads = t_num_heads
        self.head_dim = dim // t_num_heads
        self.scale = self.head_dim ** -0.5
        self.device = device
        self.t_attn_size = t_attn_size

        self.t_q_conv = nn.Linear(dim, dim, bias=qkv_bias)
        self.t_k_conv = nn.Linear(dim, dim, bias=qkv_bias)
        self.t_v_conv = nn.Linear(dim, dim, bias=qkv_bias)
        self.t_attn_drop = nn.Dropout(attn_drop)


        self.proj = nn.Linear(dim, dim_out)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, T, N, D = x.shape
        t_q = self.t_q_conv(x.permute(0, 3, 1, 2)).permute(0, 3, 2, 1)
        t_k = self.t_k_conv(x.permute(0, 3, 1, 2)).permute(0, 3, 2, 1)
        t_v = self.t_v_conv(x.permute(0, 3, 1, 2)).permute(0, 3, 2, 1)
        t_q = t_q.reshape(B, N, T, self.t_num_heads, self.head_dim).permute(0, 1, 3, 2, 4)
        t_k = t_k.reshape(B, N, T, self.t_num_heads, self.head_dim).permute(0, 1, 3, 2, 4)
        t_v = t_v.reshape(B, N, T, self.t_num_heads, self.head_dim).permute(0, 1, 3, 2, 4)

        t_attn = (t_q @ t_k.transpose(-2, -1)) * self.scale

        t_attn = t_attn.softmax(dim=-1)
        t_attn = self.t_attn_drop(t_attn)

        t_x = (t_attn @ t_v).transpose(2, 3).reshape(B, N, T, D).transpose(1, 2)

        x = self.proj(t_x)
        x = self.proj_drop(x)
        return x


class STEncoderBlock(nn.Module):

    def __init__(
        self, dim, s_attn_size, t_attn_size, geo_num_heads=4, sem_num_heads=4, tc_num_heads=4, t_num_heads=4, mlp_ratio=4., qkv_bias=True, drop=0., attn_drop=0.,
        drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm, device=torch.device('cpu'), type_ln="pre", output_dim=1,
    ):
        super().__init__()
        self.type_ln = type_ln
        self.norm1 = norm_layer(dim)
        self.norm2 = norm_layer(dim)
        # self.norm1 = LlamaRMSNorm(dim)
        # self.norm2 = LlamaRMSNorm(dim)
        self.st_attn = STSelfAttention(
            dim, s_attn_size, t_attn_size, geo_num_heads=geo_num_heads, sem_num_heads=sem_num_heads, tc_num_heads=tc_num_heads, t_num_heads=t_num_heads, qkv_bias=qkv_bias,
            attn_drop=attn_drop, proj_drop=drop, device=device, output_dim=output_dim,
        )
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)
        self.drop1 = nn.Dropout(0.15)
        self.drop2 = nn.Dropout(0.15)

    def forward(self, x, TH, geo_mask=None, sem_mask=None):
        if self.type_ln == 'pre':
            x = x + self.drop_path(self.st_attn(self.norm1(x), TH, geo_mask=geo_mask, sem_mask=sem_mask))
            x = x + self.drop_path(self.mlp(self.norm2(x)))
        elif self.type_ln == 'post':
            x = self.norm1(x + self.drop_path(self.st_attn(x, TH, geo_mask=geo_mask, sem_mask=sem_mask)))
            x = self.norm2(x + self.drop_path(self.mlp(x)))
        elif self.type_ln == 'else':
            x = self.norm1(self.drop1(self.st_attn(self.norm1(x), TH, geo_mask=geo_mask, sem_mask=sem_mask)+x))
            x = self.norm2(x + self.drop2(self.mlp(x)))
        else:
            x = x + self.mlp(self.st_attn(self.norm1(x), TH, geo_mask=geo_mask, sem_mask=sem_mask))
        return x

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

class PDFormer(nn.Module):
    def __init__(self, args, device, dim_in):
        super(PDFormer, self).__init__()

        # self.num_nodes = args.num_nodes
        # self.feature_dim = dim_in
        self.feature_dim = dim_in
        self.ext_dim = 0
        # self.num_batches = args.num_batches
        # self.dtw_matrix = args.dtw_matrix
        self.adj_mx = args.adj_mx
        # self.sd_mx = args.sd_mx
        self.sh_mx = args.sh_mx
        self.lap_mx = args.lap_mx


        self.embed_dim = args.embed_dim
        self.skip_dim = args.skip_dim
        self.lape_dim = args.lape_dim
        self.geo_num_heads = args.geo_num_heads
        self.sem_num_heads = args.sem_num_heads
        self.tc_num_heads = args.tc_num_heads
        self.t_num_heads = args.t_num_heads
        self.mlp_ratio = args.mlp_ratio
        self.qkv_bias = args.qkv_bias
        self.drop = args.drop
        self.attn_drop = args.attn_drop
        self.drop_path = args.drop_path
        self.s_attn_size = args.s_attn_size
        self.t_attn_size = args.t_attn_size
        self.enc_depth = args.enc_depth
        self.type_ln = args.type_ln
        self.type_short_path = args.type_short_path

        self.output_dim = dim_in
        self.input_window = args.input_window
        self.output_window = args.output_window
        self.add_time_in_day = args.add_time_in_day
        self.add_day_in_week = args.add_day_in_week
        self.device = device
        # self.world_size = args.world_size
        # self.huber_delta = args.huber_delta
        # self.quan_delta = args.quan_delta
        self.far_mask_delta = args.far_mask_delta
        # self.dtw_delta = args.dtw_delta

        sh_mx = self.sh_mx.T
        self.geo_mask = torch.zeros_like(sh_mx)
        self.geo_mask[sh_mx >= self.far_mask_delta] = 1
        self.geo_mask = self.geo_mask.bool()
        self.sem_mask = None


        self.patch_embedding_flow = PatchEmbedding_flow(
            self.embed_dim, patch_len=12, stride=12, padding=0)

        enc_dpr = [x.item() for x in torch.linspace(0, self.drop_path, self.enc_depth)]
        self.encoder_blocks = nn.ModuleList([
            STEncoderBlock(
                dim=self.embed_dim, s_attn_size=self.s_attn_size, t_attn_size=self.t_attn_size,
                geo_num_heads=self.geo_num_heads, sem_num_heads=self.sem_num_heads, tc_num_heads=self.tc_num_heads, t_num_heads=self.t_num_heads,
                mlp_ratio=self.mlp_ratio, qkv_bias=self.qkv_bias, drop=self.drop, attn_drop=self.attn_drop, drop_path=enc_dpr[i], act_layer=nn.GELU,
                norm_layer=partial(nn.LayerNorm, eps=1e-6), device=self.device, type_ln=self.type_ln, output_dim=self.output_dim,
            ) for i in range(self.enc_depth)
        ])
        self.skip_convs = nn.ModuleList([nn.Linear(self.embed_dim, self.skip_dim) for _ in range(self.enc_depth)])
        self.end_conv1 = nn.Linear(self.input_window, self.output_window, bias=True)
        self.end_conv2 = nn.Linear(self.skip_dim, self.output_dim, bias=True)

    def forward(self, input, select_dataset):
        x = input
        TH = None
        enc = self.patch_embedding_flow(input[..., 0:1])
        skip = 0
        for i, encoder_block in enumerate(self.encoder_blocks):
            enc = encoder_block(enc, TH, self.geo_mask.to(self.device), self.sem_mask)
            skip += self.skip_convs[i](enc)
        skip = self.end_conv1(F.relu(skip.permute(0, 3, 2, 1)))
        skip = self.end_conv2(F.relu(skip.permute(0, 3, 2, 1)))
        return skip