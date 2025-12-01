from model import AutoFed
import torch
import numpy as np
from dataset import Traffic_Dataset
from torch.utils.data import DataLoader

# calculate MSE, RMSE, MAE, MAPE
def loss_func(y_hat, y, epsilon = 1e-5):
    # mask = ~torch.isnan(y)
    # mask = mask.float()
    # mask = mask / torch.sum(mask)
    #
    # mse = torch.sum(mask * (y_hat-y)**2)
    # rmse = torch.sqrt(mse)
    # mae = torch.sum(mask * torch.abs(y_hat-y))
    #
    # mask = (y>1e-4).float()
    # mask = mask / (epsilon+torch.sum(mask))
    # mape = torch.sum(mask * torch.abs((y_hat-y)/(y+epsilon)))

    mse = torch.mean((y_hat-y)**2)
    rmse = torch.sqrt(mse)
    mae = torch.mean(torch.abs(y_hat-y))
    # mape = torch.mean(torch.abs((y_hat-y)/(y+epsilon)))
    mask = (y>1e-4).float()
    mask = mask / (epsilon+torch.sum(mask))
    mape = torch.sum(mask * torch.abs((y_hat-y)/(y+epsilon)))

    return mse, rmse, mae, mape

class Client():
    def __init__(self, dataset, client_id, args):
        super().__init__()

        self.client_id = client_id
        self.save_path = f"./{str(args.num_client)}/{client_id}.pth"
        self.max_norm = args.max_grad_norm

        self.num_nodes = dataset[0][0].shape[2]

        device_name = "cuda:"+args.cuda
        self.device = torch.device(device_name if torch.cuda.is_available() and not args.cpu else 'cpu')

        model = AutoFed(node_num=self.num_nodes, history=args.history, horizon=args.horizon, dim_in=args.input_dim, dim_in_dec=args.input_dec_dim, dim_out=args.output_dim, dim_hidden=args.hidden_dim, cheb_k=args.cheb_k, embed_dim=args.hidden_dim, layer=args.layer)
        self.model = model.to(self.device)
        self.W = {key: value for key, value in self.model.named_parameters()}

        train_data = Traffic_Dataset(dataset[0][0], dataset[0][1])
        valid_data = Traffic_Dataset(dataset[1][0], dataset[1][1])
        test_data = Traffic_Dataset(dataset[2][0], dataset[2][1])

        self.train_loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True)
        self.valid_loader = DataLoader(valid_data, batch_size=args.batch_size, shuffle=True)
        self.test_loader = DataLoader(test_data, batch_size=args.batch_size, shuffle=True)

        self.optim = torch.optim.Adam(self.model.parameters(), lr=args.lr, eps=args.epsilon)
        # self.scheduler = torch.optim.lr_scheduler.ExponentialLR(self.optim, gamma=args.gamma)


    def iteration(self, mode="train", epochs=1):
        mse_list, rmse_list, mae_list, mape_list, loss_list = [], [], [], [], []

        if mode == "train":
            self.model.train()
            torch.set_grad_enabled(True)
            data_iter = self.train_loader
        elif mode == "val":
            self.model.eval()
            torch.set_grad_enabled(False)
            data_iter = self.valid_loader
        elif mode == "test":
            self.model.eval()
            torch.set_grad_enabled(False)
            data_iter = self.test_loader
        else:
            print("Wrong Mode")
            exit()

        for _ in range(epochs):
            for x, y in data_iter:
                x, y = x.float().to(self.device), y.float().to(self.device)
                x_dec = torch.zeros_like(x)

                if mode == "train":
                    y_hat, loss_ae = self.model(x, x_dec, y)
                    mse, rmse, mae, mape = loss_func(y_hat, y)
                    alpha = loss_ae / mae
                    loss = mae + alpha * loss_ae
                    self.optim.zero_grad()
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_norm)
                    self.optim.step()
                    # self.scheduler.step()
                else:
                    y_hat, loss_ae = self.model(x, x_dec)
                    mse, rmse, mae, mape = loss_func(y_hat, y)
                    alpha = loss_ae / mae
                    loss = mae + alpha * loss_ae

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