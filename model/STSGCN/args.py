import numpy as np
import configparser
from lib.predifineGraph import get_adjacency_matrix

def parse_args(dataset_use, num_nodes_dict, parser):
    # get configuration
    config_file = '../conf/STSGCN/STSGCN.conf'
    config = configparser.ConfigParser()
    config.read(config_file)

    filter_list_str = config.get('model', 'filter_list')
    filter_list = eval(filter_list_str)

    # data
    parser.add_argument('--num_nodes', type=int, default=num_nodes_dict[dataset_use[0]])
    parser.add_argument('--input_window', type=int, default=config['data']['input_window'])
    parser.add_argument('--output_window', type=int, default=config['data']['output_window'])

    # model
    parser.add_argument('--filter_list', type=list, default=config['model']['filter_list'])
    parser.add_argument('--rho', type=int, default=config['model']['rho'])
    parser.add_argument('--feature_dim', type=int, default=config['model']['feature_dim'])
    parser.add_argument('--module_type', type=str, default=config['model']['module_type'])
    parser.add_argument('--activation', type=str, default=config['model']['activation'])
    parser.add_argument('--temporal_emb', type=eval, default=config['model']['temporal_emb'])
    parser.add_argument('--spatial_emb', type=eval, default=config['model']['spatial_emb'])
    parser.add_argument('--use_mask', type=eval, default=config['model']['use_mask'])
    parser.add_argument('--steps', type=int, default=config['model']['steps'])
    parser.add_argument('--first_layer_embedding_size', type=int, default=config['model']['first_layer_embedding_size'])
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
    args.filter_list = filter_list
    args.A = A
    return args