import torch
from model import AutoFed

class Server():
    def __init__(self, args):
        super().__init__()
        device_name = "cuda:"+args.cuda
        self.device = torch.device(device_name if torch.cuda.is_available() and not args.cpu else 'cpu')
        model = AutoFed(node_num=1, history=args.history, horizon=args.horizon, dim_in=args.input_dim, dim_in_dec=args.input_dec_dim, dim_out=args.output_dim, dim_hidden=args.hidden_dim, cheb_k=args.cheb_k, embed_dim=args.hidden_dim, layer=args.layer)
        self.model = model.to(self.device)
        self.W = {key: value for key, value in self.model.named_parameters()}

    def aggregate(self, client_list):
        weight_list = []
        weight_total = 0
        for k in self.W.keys():
            if "pattern_mlp" in k and "batch_norm" not in k:
                self.W[k].data = self.W[k].data - self.W[k].data
                for i in range(len(client_list)):
                    if len(client_list) != len(weight_list):
                        weight_list.append(client_list[i].num_nodes)
                        weight_total += client_list[i].num_nodes
                    self.W[k].data = self.W[k].data + client_list[i].W[k].data.clone() * weight_list[i]
                self.W[k].data = self.W[k].data / weight_total

