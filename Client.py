from model import GRU_AGCN_TP
import torch
import torch.nn as nn
from utils.dataset import prepare_x_y
import numpy as np

ae_rate = 0.1

# calculate MSE, RMSE, MAE, MAPE
def loss_func(y_hat, y):
    epsilon = 1e-8

    mask = ~torch.isnan(y)
    mask = mask.float()
    mask = mask / torch.sum(mask)

    mse = torch.sum(mask * (y_hat-y)**2)
    rmse = torch.sqrt(mse)
    mae = torch.sum(mask * torch.abs(y_hat-y))

    mask = (y>1e-4).float() # follow the set in https://github.com/lichuan210/FedTPS
    mask = mask / (epsilon+torch.sum(mask))
    mape = torch.sum(mask * torch.abs((y_hat-y)/(y+epsilon)))

    return mse, rmse, mae, mape

class Client():
    def __init__(self, dataset, client_id, args):
        super().__init__()

        self.client_id = client_id
        self.save_path = f"./model/{client_id}.pth"

        self.num_nodes = dataset["adj"].shape[1] # different clients have different node sets

        device_name = "cuda:"+args.cuda
        self.device = torch.device(device_name if torch.cuda.is_available() and not args.cpu else 'cpu')

        model = GRU_AGCN_TP(node_num=self.num_nodes, horizon=args.horizon, dim_in=args.input_dim, dim_in_dec=args.input_dec_dim, dim_out=args.output_dim, dim_hidden=args.hidden_dim, cheb_k=args.cheb_k, embed_dim=args.hidden_dim, pattern_num = args.pattern_num, pattern_dim = args.pattern_dim,layer=args.layer)
        self.model = model.to(self.device)
        self.W = {key: value for key, value in self.model.named_parameters()}

        self.dataset = dataset

        self.optim = torch.optim.Adam(self.model.parameters(), lr=args.lr, eps=args.epsilon)


    def train(self, epochs=1):
        self.model.train()
        mse_list, rmse_list, mae_list, mape_list, loss_list = [], [], [], [], []
        for _ in range(epochs):
            data_iter = self.dataset['train_loader'].get_iterator()
            for x, y in data_iter:
                x, y, x_dec = prepare_x_y(x,y,self.device)
                y_hat, loss_ae = self.model(x,x_dec,y)
                y = self.dataset["scaler"].inverse_transform(y)
                y_hat = self.dataset["scaler"].inverse_transform(y_hat)
                mse, rmse, mae, mape = loss_func(y_hat, y)
                # print(mse, rmse, mae, mape)
                loss = mae + ae_rate * loss_ae

                self.optim.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 5)
                self.optim.step()

                mse_list.append(mse.item())
                rmse_list.append(rmse.item())
                mae_list.append(mae.item())
                mape_list.append(mape.item())
                loss_list.append(loss.item())

        return np.mean(mse_list), np.mean(rmse_list), np.mean(mae_list), np.mean(mape_list), np.mean(loss_list)

    def eval(self, mode="val"):
        self.model.eval()
        mse_list, rmse_list, mae_list, mape_list, loss_list = [], [], [], [], []
        data_iter = self.dataset[f'{mode}_loader'].get_iterator()
        for x, y in data_iter:
            x, y, x_dec = prepare_x_y(x, y, self.device)
            y_hat, loss_ae = self.model(x, x_dec, y)

            y = self.dataset["scaler"].inverse_transform(y)
            y_hat = self.dataset["scaler"].inverse_transform(y_hat)

            mse, rmse, mae, mape = loss_func(y_hat, y)
            loss = mae + ae_rate * loss_ae

            mse_list.append(mse.item())
            rmse_list.append(rmse.item())
            mae_list.append(mae.item())
            mape_list.append(mape.item())
            loss_list.append(loss.item())

        return np.mean(mse_list), np.mean(rmse_list), np.mean(mae_list), np.mean(mape_list), np.mean(loss_list)

    def save(self, model_path=None):
        if model_path is None:
            model_path = self.save_path
        torch.save(self.model.state_dict(), model_path)

    def update_weight(self, W):
        for key in W:
            if "pattern_mlp" in key and "batch_norm" not in key:
                self.W[key].data = W[key].data.clone()