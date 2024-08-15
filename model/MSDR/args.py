import numpy as np
from scipy.sparse import linalg
import scipy.sparse as sp
import pandas as pd
import torch
import configparser
from lib.predifineGraph import get_adjacency_matrix

def calculate_normalized_laplacian(adj):
    """
    # L = D^-1/2 (D-A) D^-1/2 = I - D^-1/2 A D^-1/2
    # D = diag(A 1)
    :param adj:
    :return:
    """
    adj = sp.coo_matrix(adj)
    d = np.array(adj.sum(1))
    d_inv_sqrt = np.power(d, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
    normalized_laplacian = sp.eye(adj.shape[0]) - adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt).tocoo()
    return normalized_laplacian


def calculate_random_walk_matrix(adj_mx):
    adj_mx = sp.coo_matrix(adj_mx)
    d = np.array(adj_mx.sum(1))
    d_inv = np.power(d, -1).flatten()
    d_inv[np.isinf(d_inv)] = 0.
    d_mat_inv = sp.diags(d_inv)
    random_walk_mx = d_mat_inv.dot(adj_mx).tocoo()
    return random_walk_mx


def calculate_reverse_random_walk_matrix(adj_mx):
    return calculate_random_walk_matrix(np.transpose(adj_mx))


def calculate_scaled_laplacian(adj_mx, lambda_max=2, undirected=True):
    if undirected:
        adj_mx = np.maximum.reduce([adj_mx, adj_mx.T])
    L = calculate_normalized_laplacian(adj_mx)
    if lambda_max is None:
        lambda_max, _ = linalg.eigsh(L, 1, which='LM')
        lambda_max = lambda_max[0]
    L = sp.csr_matrix(L)
    M, _ = L.shape
    I = sp.identity(M, format='csr', dtype=L.dtype)
    L = (2 / lambda_max * L) - I
    return L.astype(np.float32)

def get_adjacency_matrix(distance_df_filename, num_of_vertices, type_='connectivity', id_filename=None):
    A = np.zeros((int(num_of_vertices), int(num_of_vertices)), dtype=np.float32)

    if id_filename:
        with open(id_filename, 'r') as f:
            id_dict = {int(i): idx for idx, i in enumerate(f.read().strip().split('\n'))}  # 建立映射列表
        df = pd.read_csv(distance_df_filename)
        for row in df.values:
            if len(row) != 3:
                continue
            i, j = int(row[0]), int(row[1])
            A[id_dict[i], id_dict[j]] = 1
            A[id_dict[j], id_dict[i]] = 1

        return A
    df = pd.read_csv(distance_df_filename)
    for row in df.values:
        if len(row) != 3:
            continue
        i, j, distance = int(row[0]), int(row[1]), float(row[2])
        if type_ == 'connectivity':
            A[i, j] = 1
            A[j, i] = 1
        elif type == 'distance':
            A[i, j] = 1 / distance
            A[j, i] = 1 / distance
        else:
            raise ValueError("type_ error, must be "
                             "connectivity or distance!")

    return A

def parse_args(dataset_use, num_nodes_dict, parser):
    # get configuration
    config_file = '../conf/MSDR/MSDR.conf'
    config = configparser.ConfigParser()
    config.read(config_file)

    parser.add_argument('--filter_type', default=config['model']['cl_decay_steps'], type=str)
    # parser.add_argument('--data', default=DATASET, help='data path', type=str, )
    parser.add_argument('--cl_decay_steps', type=int, default=config['model']['cl_decay_steps'])
    parser.add_argument('--num_nodes', type=int, default=num_nodes_dict[dataset_use[0]])
    parser.add_argument('--horizon', type=int, default=config['model']['horizon'])
    parser.add_argument('--seq_len', type=int, default=config['model']['seq_len'])
    parser.add_argument('--output_len', type=int, default=config['model']['output_len'])
    parser.add_argument('--max_diffusion_step', type=int, default=config['model']['max_diffusion_step'])
    parser.add_argument('--num_rnn_layers', type=int, default=config['model']['num_rnn_layers'])
    parser.add_argument('--output_dim', type=int, default=config['model']['output_dim'])
    parser.add_argument('--rnn_units', type=int, default=config['model']['rnn_units'])
    parser.add_argument('--pre_k', type=int, default=config['model']['pre_k'])
    parser.add_argument('--pre_v', type=int, default=config['model']['pre_v'])
    parser.add_argument('--use_curriculum_learning', type=eval, default=config['model']['use_curriculum_learning'])
    parser.add_argument('--construct_type', type=str, default=config['model']['construct_type'])
    parser.add_argument('--l2lambda', type=int, default=config['model']['l2lambda'])
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
    adj_mx = torch.FloatTensor(A)
    args.adj_mx = adj_mx
    print(adj_mx, adj_mx.shape)
    return args

