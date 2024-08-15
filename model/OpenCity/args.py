import os
import numpy as np
from fastdtw import fastdtw
from tqdm import tqdm
from tslearn.clustering import TimeSeriesKMeans, KShape
import argparse
import configparser
from lib.predifineGraph import get_adjacency_matrix, cal_lape
import pandas as pd
import torch
import copy

def load_rel_2(adj_mx, args, dataset):
    directory = f'../data/OpenCity/{dataset}'
    if not os.path.exists(directory):
        os.makedirs(directory)
    filename = f'{dataset}_sh_mx.npy'
    cache_path = os.path.join(directory, filename)
    sh_mx = adj_mx.copy()
    if args.type_short_path == 'hop':
        if not os.path.exists(cache_path):
            sh_mx[sh_mx > 0] = 1
            sh_mx[sh_mx == 0] = 511
            np.save(cache_path, sh_mx)
        sh_mx = np.load(cache_path)
    return sh_mx

def parse_args(parser, args_base):
    # get configuration
    config_file = '../conf/OpenCity/OpenCity.conf'
    config = configparser.ConfigParser()
    config.read(config_file)

    # data
    parser.add_argument('--input_window', type=int, default=config['data']['input_window'])
    parser.add_argument('--output_window', type=int, default=config['data']['output_window'])
    # model
    parser.add_argument('--embed_dim', type=int, default=config['model']['embed_dim'])
    parser.add_argument('--skip_dim', type=int, default=config['model']['skip_dim'])
    parser.add_argument('--lape_dim', type=int, default=config['model']['lape_dim'])

    parser.add_argument('--geo_num_heads', type=int, default=config['model']['geo_num_heads'])
    parser.add_argument('--sem_num_heads', type=int, default=config['model']['sem_num_heads'])
    parser.add_argument('--tc_num_heads', type=int, default=config['model']['tc_num_heads'])
    parser.add_argument('--t_num_heads', type=int, default=config['model']['t_num_heads'])
    parser.add_argument('--mlp_ratio', type=int, default=config['model']['mlp_ratio'])
    parser.add_argument('--qkv_bias', type=eval, default=config['model']['qkv_bias'])
    parser.add_argument('--drop', type=float, default=config['model']['drop'])
    parser.add_argument('--attn_drop', type=float, default=config['model']['attn_drop'])
    parser.add_argument('--drop_path', type=float, default=config['model']['drop_path'])
    parser.add_argument('--s_attn_size', type=int, default=config['model']['s_attn_size'])
    parser.add_argument('--t_attn_size', type=int, default=config['model']['t_attn_size'])
    parser.add_argument('--enc_depth', type=int, default=config['model']['enc_depth'])
    parser.add_argument('--type_ln', type=str, default=config['model']['type_ln'])
    parser.add_argument('--type_short_path', type=str, default=config['model']['type_short_path'])
    parser.add_argument('--far_mask_delta', type=int, default=config['model']['far_mask_delta'])
    # train
    parser.add_argument('--seed', type=int, default=config['train']['seed'])
    parser.add_argument('--seed_mode', type=eval, default=config['train']['seed_mode'])
    parser.add_argument('--xavier', type=eval, default=config['train']['xavier'])
    parser.add_argument('--loss_func', type=str, default=config['train']['loss_func'])
    # parser.add_argument('--real_value', type=eval, default=config['train']['real_value'])

    args_predictor, _ = parser.parse_known_args()
    args_predictor.real_value = config['train']['real_value']
    sh_mx_dict = {}
    lpls_dict = {}
    adj_mx_dict = {}
    for dataset_select in (args_base.dataset_use):
        args_predictor.filepath = '../data/' + dataset_select +'/'
        args_predictor.filename = dataset_select
        if dataset_select == 'PEMS08' or dataset_select == 'PEMS04' or dataset_select == 'PEMS07':
            A, Distance = get_adjacency_matrix(
                distance_df_filename=args_predictor.filepath + dataset_select + '.csv',
                num_of_vertices=args_base.num_nodes_dict[dataset_select])
            A = A + np.eye(A.shape[0])
        else:
            A = np.load(args_predictor.filepath + f'{dataset_select}_rn_adj.npy')
        sh_mx_dict[dataset_select] = torch.FloatTensor(load_rel_2(A, args_predictor, dataset_select))
        lpls_dict[dataset_select] = torch.FloatTensor(cal_lape(copy.deepcopy(A)))
        d = np.sum(A, axis=1)
        sinvD = np.sqrt(np.mat(np.diag(d)).I)
        A_mx = np.mat(np.identity(A.shape[0]) + sinvD * A * sinvD)
        adj_mx_dict[dataset_select] = torch.FloatTensor(copy.deepcopy(A_mx))

    args_predictor.adj_mx_dict = adj_mx_dict
    args_predictor.sd_mx = None
    args_predictor.sh_mx_dict = sh_mx_dict
    args_predictor.lap_mx_dict = lpls_dict
    return args_predictor