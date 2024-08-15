import numpy as np
import configparser
import torch
from scipy.sparse.linalg import eigs
from lib.predifineGraph import get_adjacency_matrix
import pandas as pd


def scaled_Laplacian(W):
    '''
    compute \tilde{L}

    Parameters
    ----------
    W: np.ndarray, shape is (N, N), N is the num of vertices

    Returns
    ----------
    scaled_Laplacian: np.ndarray, shape (N, N)

    '''

    assert W.shape[0] == W.shape[1]

    D = np.diag(np.sum(W, axis=1))

    L = D - W

    lambda_max = eigs(L, k=1, which='LR')[0].real

    return (2 * L) / lambda_max - np.identity(W.shape[0])

def parse_args(dataset_use, num_nodes_dict, parser):
    # get configuration
    config_file = '../conf/STWA/STWA.conf'
    config = configparser.ConfigParser()
    config.read(config_file)

    # data
    # parser.add_argument('--data', default=DATASET, help='data path', type=str, )
    parser.add_argument('--num_nodes', type=int, default=num_nodes_dict[dataset_use[0]])
    parser.add_argument('--lag', type=int, default=config['data']['lag'])
    parser.add_argument('--horizon', type=int, default=config['data']['horizon'])
    # model
    parser.add_argument('--in_dim', type=int, default=config['model']['in_dim'])
    parser.add_argument('--out_dim', type=int, default=config['model']['out_dim'])
    parser.add_argument('--channels', type=int, default=config['model']['channels'])
    parser.add_argument('--dynamic', type=str, default=config['model']['dynamic'])
    parser.add_argument('--memory_size', type=int, default=config['model']['memory_size'])
    parser.add_argument('--column_wise', type=bool, default=False)
    # train
    parser.add_argument('--seed', type=int, default=config['train']['seed'])
    parser.add_argument('--seed_mode', type=eval, default=config['train']['seed_mode'])
    parser.add_argument('--xavier', type=eval, default=config['train']['xavier'])
    parser.add_argument('--loss_func', type=str, default=config['train']['loss_func'])
    args, _ = parser.parse_known_args()
    DATASET = dataset_use[0]
    args.filepath = '../data/' + DATASET + '/'
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

    adj_mx = scaled_Laplacian(A)
    adj_mx = [adj_mx]
    args.adj_mx = adj_mx

    return args