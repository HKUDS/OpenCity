import torch
import torch.nn as nn

class Traffic_model(nn.Module):
    def __init__(self, args, args_predictor):
        super(Traffic_model, self).__init__()
        # self.num_node = args.num_nodes
        self.input_base_dim = args.input_base_dim
        self.input_extra_dim = args.input_extra_dim
        self.output_dim = args.output_dim
        self.mode = args.mode
        self.model = args.model
        self.load_pretrain_path = args.load_pretrain_path
        self.log_dir = args.log_dir

        dim_in = self.input_base_dim
        dim_out = self.output_dim

        if self.model == 'MTGNN':
            from MTGNN.MTGNN import MTGNN
            self.predictor = MTGNN(args_predictor, args.device, dim_in, dim_out)
        elif self.model == 'STGCN':
            from STGCN.stgcn import STGCN
            self.predictor = STGCN(args_predictor, args.device, dim_in, dim_out)
        elif self.model == 'STSGCN':
            from STSGCN.STSGCN import STSGCN
            self.predictor = STSGCN(args_predictor, args.device, dim_in, dim_out)
        elif self.model == 'ASTGCN':
            from ASTGCN.ASTGCN import ASTGCN
            self.predictor = ASTGCN(args_predictor, args.device, dim_in, dim_out)
        elif self.model == 'AGCRN':
            from AGCRN.AGCRN import AGCRN
            self.predictor = AGCRN(args_predictor, args.device, dim_in, dim_out)
        elif self.model == 'GWN':
            from GWN.GWN import GWNET
            self.predictor = GWNET(args_predictor, args.device, dim_in, dim_out)
        elif self.model == 'TGCN':
            from TGCN.TGCN import TGCN
            self.predictor = TGCN(args_predictor, args.device, dim_in)
        elif self.model == 'STWA':
            from ST_WA.ST_WA import STWA
            self.predictor = STWA(args_predictor, args.device, dim_in)
        elif self.model == 'MSDR':
            from MSDR.gmsdr_model import GMSDRModel
            args_predictor.input_dim = dim_in
            self.predictor = GMSDRModel(args_predictor, args.device)
        elif self.model == 'PDFormer':
            from PDFormer.PDFormer import PDFormer
            self.predictor = PDFormer(args_predictor, args.device, dim_in)
        elif self.model == 'OpenCity':
            from OpenCity.OpenCity import OpenCity
            self.predictor = OpenCity(args_predictor, args.dataset_use, args.device, dim_in)
        else:
            raise ValueError

    def forward(self, source, label, select_dataset, batch_seen=None):
        if self.model == 'OpenCity':
            x_predic = self.predictor(source, label, select_dataset)
        else:
            x_predic = self.predictor(source[..., 0:self.input_base_dim], select_dataset)
        return x_predic
