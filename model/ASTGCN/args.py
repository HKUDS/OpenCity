import numpy as np
import configparser
from lib.predifineGraph import get_adjacency_matrix
import torch
import pandas as pd

def parse_args(dataset_use, num_nodes_dict, parser):
    # get configuration
    config_file = '../conf/ASTGCN/ASTGCN.conf'
    config = configparser.ConfigParser()
    config.read(config_file)

    # data
    # parser.add_argument('--num_nodes', type=int, default=config['data']['num_nodes'])
    parser.add_argument('--num_nodes', type=int, default=num_nodes_dict[dataset_use[0]])
    parser.add_argument('--len_input', type=int, default=config['data']['len_input'])
    parser.add_argument('--num_for_predict', type=int, default=config['data']['num_for_predict'])
    # model
    parser.add_argument('--nb_block', type=int, default=config['model']['nb_block'])
    parser.add_argument('--K', type=int, default=config['model']['K'])
    parser.add_argument('--nb_chev_filter', type=int, default=config['model']['nb_chev_filter'])
    parser.add_argument('--nb_time_filter', type=int, default=config['model']['nb_time_filter'])
    parser.add_argument('--time_strides', type=int, default=config['model']['time_strides'])
    # train
    parser.add_argument('--seed', type=int, default=config['train']['seed'])
    parser.add_argument('--seed_mode', type=eval, default=config['train']['seed_mode'])
    parser.add_argument('--xavier', type=eval, default=config['train']['xavier'])
    parser.add_argument('--loss_func', type=str, default=config['train']['loss_func'])
    args, _ = parser.parse_known_args()
    DATASET = dataset_use[0]
    args.filepath = '../data/' + DATASET +'/'
    args.filename = DATASET

    if DATASET == 'PEMS08' or DATASET == 'PEMS04' or DATASET == 'PEMS07':
        A, Distance = get_adjacency_matrix(
            distance_df_filename=args.filepath + args.filename + '.csv',
            num_of_vertices=num_nodes_dict[args.filename])
        A = A + np.eye(A.shape[0])
    elif DATASET == 'PEMS03':
        A, Distance = get_adjacency_matrix(
            distance_df_filename=args.filepath + DATASET + '.csv',
            num_of_vertices=args.num_nodes)
    else:
        A = np.load(args.filepath + f'{args.filename}_rn_adj.npy')
    args.A = A
    return args