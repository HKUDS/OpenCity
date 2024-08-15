import numpy as np
import pandas as pd
import configparser
from lib.predifineGraph import get_adjacency_matrix, load_pickle, weight_matrix

def parse_args(dataset_use, num_nodes_dict, parser):
    # get configuration
    config_file = '../conf/MTGNN/MTGNN.conf'
    config = configparser.ConfigParser()
    config.read(config_file)

    # data
    parser.add_argument('--num_nodes', type=int, default=num_nodes_dict[dataset_use[0]])
    parser.add_argument('--input_window', type=int, default=config['data']['input_window'])
    parser.add_argument('--output_window', type=int, default=config['data']['output_window'])
    parser.add_argument('--output_dim', type=int, default=config['data']['output_dim'])
    # model
    parser.add_argument('--gcn_true', type=eval, default=config['model']['gcn_true'])
    parser.add_argument('--buildA_true', type=eval, default=config['model']['buildA_true'])
    parser.add_argument('--gcn_depth', type=int, default=config['model']['gcn_depth'])
    parser.add_argument('--dropout', type=float, default=config['model']['dropout'])
    parser.add_argument('--subgraph_size', type=int, default=config['model']['subgraph_size'])
    parser.add_argument('--node_dim', type=int, default=config['model']['node_dim'])
    parser.add_argument('--dilation_exponential', type=int, default=config['model']['dilation_exponential'])
    parser.add_argument('--conv_channels', type=int, default=config['model']['conv_channels'])
    parser.add_argument('--residual_channels', type=int, default=config['model']['residual_channels'])
    parser.add_argument('--skip_channels', type=int, default=config['model']['skip_channels'])
    parser.add_argument('--end_channels', type=int, default=config['model']['end_channels'])
    parser.add_argument('--layers', type=int, default=config['model']['layers'])
    parser.add_argument('--propalpha', type=float, default=config['model']['propalpha'])
    parser.add_argument('--tanhalpha', type=int, default=config['model']['tanhalpha'])
    parser.add_argument('--layer_norm_affline', type=eval, default=config['model']['layer_norm_affline'])
    parser.add_argument('--use_curriculum_learning', type=eval, default=config['model']['use_curriculum_learning'])
    parser.add_argument('--task_level', type=int, default=config['model']['task_level'])
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
    args.adj_mx = A
    return args