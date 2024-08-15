import numpy as np
import configparser
from lib.predifineGraph import get_adjacency_matrix
import torch
import pandas as pd

def scaled_laplacian(W):
    '''
    Normalized graph Laplacian function.
    :param W: np.ndarray, [n_route, n_route], weighted adjacency matrix of G.
    :return: np.matrix, [n_route, n_route].
    '''
    # d ->  diagonal degree matrix
    n, d = np.shape(W)[0], np.sum(W, axis=1)
    # L -> graph Laplacian
    L = -W
    L[np.diag_indices_from(L)] = d
    for i in range(n):
        for j in range(n):
            if (d[i] > 0) and (d[j] > 0):
                L[i, j] = L[i, j] / np.sqrt(d[i] * d[j])
    # lambda_max \approx 2.0, the largest eigenvalues of L.
    # lambda_max = eigs(L, k=1, which='LR')[0][0].real
    lambda_max = np.linalg.eigvals(L).max().real
    return np.mat(2 * L / lambda_max - np.identity(n))


def cheb_poly_approx(L, Ks, n):
    '''
    Chebyshev polynomials approximation function.
    :param L: np.matrix, [n_route, n_route], graph Laplacian.
    :param Ks: int, kernel size of spatial convolution.
    :param n: int, number of routes / size of graph.
    :return: np.ndarray, [n_route, Ks*n_route].
    '''
    L0, L1 = np.mat(np.identity(n)), np.mat(np.copy(L))

    if Ks > 1:
        L_list = [np.copy(L0), np.copy(L1)]
        for i in range(Ks - 2):
            Ln = np.mat(2 * L * L1 - L0)
            L_list.append(np.copy(Ln))
            L0, L1 = np.matrix(np.copy(L1)), np.matrix(np.copy(Ln))
        # L_lsit [Ks, n*n], Lk [n, Ks*n]
        return np.stack(L_list, axis=0)
    elif Ks == 1:
        return np.asarray(L0)
    else:
        raise ValueError(f'ERROR: the size of spatial kernel must be greater than 1, but received "{Ks}".')

def parse_args(dataset_use, num_nodes_dict, parser):
    # get configuration
    config_file = '../conf/STGCN/STGCN.conf'
    config = configparser.ConfigParser()
    config.read(config_file)

    blocks1_str = config.get('model', 'blocks1')
    blocks1 = eval(blocks1_str)
    # data
    parser.add_argument('--num_nodes', type=int, default=num_nodes_dict[dataset_use[0]])
    parser.add_argument('--input_window', type=int, default=config['data']['input_window'])
    parser.add_argument('--output_window', type=int, default=config['data']['output_window'])
    # model
    parser.add_argument('--Ks', type=int, default=config['model']['Ks'])
    parser.add_argument('--Kt', type=int, default=config['model']['Kt'])
    parser.add_argument('--blocks1', type=list, default=blocks1)
    parser.add_argument('--drop_prob', type=int, default=config['model']['drop_prob'])
    parser.add_argument('--outputl_ks', type=int, default=config['model']['outputl_ks'])
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
    L = scaled_laplacian(A)
    Lk = cheb_poly_approx(L, 3, args.num_nodes)
    args.G = torch.FloatTensor(Lk)
    return args