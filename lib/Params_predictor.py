import argparse
import configparser

def get_predictor_params(args):
    # get the based paras of predictors
    config_file = '../conf/general_conf/global_baselines.conf'
    config = configparser.ConfigParser()
    config.read(config_file)

    parser_pred = argparse.ArgumentParser(prefix_chars='--', description='predictor_based_arguments')
    # train
    parser_pred.add_argument('--batch_size', default=config['train']['batch_size'], type=int)
    parser_pred.add_argument('--epochs', default=config['train']['epochs'], type=int)
    parser_pred.add_argument('--val_ratio', default=config['train']['val_ratio'], type=float)
    parser_pred.add_argument('--test_ratio', default=config['train']['test_ratio'], type=float)
    parser_pred.add_argument('--lr_init', default=config['train']['lr_init'], type=float)
    parser_pred.add_argument('--lr_decay', default=config['train']['lr_decay'], type=eval)
    parser_pred.add_argument('--lr_decay_rate', default=config['train']['lr_decay_rate'], type=float)
    parser_pred.add_argument('--lr_decay_step', default=config['train']['lr_decay_step'], type=str)
    parser_pred.add_argument('--early_stop', default=config['train']['early_stop'], type=eval)
    parser_pred.add_argument('--early_stop_patience', default=config['train']['early_stop_patience'], type=int)
    parser_pred.add_argument('--grad_norm', default=config['train']['grad_norm'], type=eval)
    parser_pred.add_argument('--max_grad_norm', default=config['train']['max_grad_norm'], type=int)
    parser_pred.add_argument('--debug', default=config['train']['debug'], type=eval)
    parser_pred.add_argument('--real_value', default=config['train']['real_value'], type=eval, help='use real value for loss calculation')

    if args.model == 'MTGNN':
        from model.MTGNN.args import parse_args
        args_predictor = parse_args(args.dataset_use, args.num_nodes_dict, parser_pred)
    elif args.model == 'STGCN':
        from model.STGCN.args import parse_args
        args_predictor = parse_args(args.dataset_use, args.num_nodes_dict, parser_pred)
    elif args.model == 'STSGCN':
        from model.STSGCN.args import parse_args
        args_predictor = parse_args(args.dataset_use, args.num_nodes_dict, parser_pred)
    elif args.model == 'ASTGCN':
        from model.ASTGCN.args import parse_args
        args_predictor = parse_args(args.dataset_use, args.num_nodes_dict, parser_pred)
    elif args.model == 'AGCRN':
        from model.AGCRN.args import parse_args
        args_predictor = parse_args(args.dataset_use, args.num_nodes_dict, parser_pred)
    elif args.model == 'GWN':
        from model.GWN.args import parse_args
        args_predictor = parse_args(args.dataset_use, args.num_nodes_dict, parser_pred)
    elif args.model == 'TGCN':
        from model.TGCN.args import parse_args
        args_predictor = parse_args(args.dataset_use, args.num_nodes_dict, parser_pred)
    elif args.model == 'STWA':
        from model.ST_WA.args import parse_args
        args_predictor = parse_args(args.dataset_use, args.num_nodes_dict, parser_pred)
    elif args.model == 'MSDR':
        from model.MSDR.args import parse_args
        args_predictor = parse_args(args.dataset_use, args.num_nodes_dict, parser_pred)
    elif args.model == 'PDFormer':
        from model.PDFormer.args import parse_args
        args_predictor = parse_args(parser_pred, args)
    elif args.model == 'OpenCity':
        from model.OpenCity.args import parse_args
        args_predictor = parse_args(parser_pred, args)
    else:
        raise ValueError

    return args_predictor