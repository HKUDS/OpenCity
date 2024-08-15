import argparse
import configparser

def parse_args(dataset_use, num_nodes_dict, parser):
    config_file = '../conf/AGCRN/AGCRN.conf'
    config = configparser.ConfigParser()
    config.read(config_file)

    #parser
    #data
    parser.add_argument('--lag', default=config['data']['lag'], type=int)
    parser.add_argument('--horizon', default=config['data']['horizon'], type=int)
    parser.add_argument('--num_nodes', type=int, default=num_nodes_dict[dataset_use[0]])
    #model
    parser.add_argument('--input_dim', default=config['model']['input_dim'], type=int)
    parser.add_argument('--output_dim', default=config['model']['output_dim'], type=int)
    parser.add_argument('--embed_dim', default=config['model']['embed_dim'], type=int)
    parser.add_argument('--rnn_units', default=config['model']['rnn_units'], type=int)
    parser.add_argument('--num_layers', default=config['model']['num_layers'], type=int)
    parser.add_argument('--cheb_k', default=config['model']['cheb_order'], type=int)
    # train
    parser.add_argument('--seed', type=int, default=config['train']['seed'])
    parser.add_argument('--seed_mode', type=eval, default=config['train']['seed_mode'])
    parser.add_argument('--xavier', type=eval, default=config['train']['xavier'])
    parser.add_argument('--loss_func', type=str, default=config['train']['loss_func'])
    args, _ = parser.parse_known_args()
    return args