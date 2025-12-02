import argparse
from dataset import load
from client import Client
from server import Server
import time
import numpy as np

def get_args():
    args = argparse.ArgumentParser(description='arguments')
    # -------------------------------data-------------------------------------------#
    args.add_argument('--dataset_path', type=str, default="../dataset/dataset.pkl", help='dataset path')
    args.add_argument('--history', type=int, default=12, help='input sequence length')
    args.add_argument('--horizon', type=int, default=12, help='output sequence length')
    args.add_argument('--slide', type=int, default=1, help='step of swing window')
    # -------------------------------model------------------------------------------#
    args.add_argument('--input_dim', type=int, default=9, help='number of input channel')
    args.add_argument('--input_dec_dim', type=int, default=9, help='number of input channel')
    args.add_argument('--output_dim', type=int, default=1, help='number of output channel')
    args.add_argument('--layer', type=int, default=1, help='number of rnn layers')
    args.add_argument('--hidden_dim', type=int, default=64, help='number of hidden units')
    args.add_argument('--cheb_k', type=int, default=3, help='max diffusion step or Cheb K')
    # -------------------------------train------------------------------------------#
    args.add_argument("--epochs", type=int, default=50, help="number of epochs of training")
    args.add_argument("--batch_size", type=int, default=128, help="size of the batches")
    args.add_argument("--lr", type=float, default=0.001, help="base learning rate")
    args.add_argument("--epsilon", type=float, default=1e-3, help="optimizer epsilon")
    args.add_argument("--gamma", type=float, default=0.98, help="scheduler epsilon")
    args.add_argument("--max_grad_norm", type=int, default=5, help="max_grad_norm")
    # -------------------------------static------------------------------------------#
    args.add_argument('--cuda', type=str, default="0", help='which gpu to use')
    args.add_argument("--cpu", action="store_true",default=False)
    args.add_argument('--num_client', type=int, default=5, help="number of clients")
    args.add_argument('--mode', type=int, default=0, help="0: uber, 1: lyft")
    args = args.parse_args()
    return args

def main():
    args = get_args()
    train_x, train_y, valid_x, valid_y, test_x, test_y = load(args.dataset_path, args.history, args.horizon, args.slide, args.num_client, args.mode)
    server = Server(args)
    clients_list = []
    for i in range(args.num_client):
        dataset = [[train_x[i], train_y[i]],[valid_x[i], valid_y[i]],[test_x[i], test_y[i]]]
        clients_list.append(Client(dataset, i, args))


    for i in range(1, 1 + args.epochs):
        server.aggregate(clients_list)


        start_time = time.time()
        mse_list, rmse_list, mae_list, mape_list, loss_list = [], [], [], [], []
        for j in range(len(clients_list)):
            client = clients_list[j]
            client.update_weight(server.W)
            mse, rmse, mae, mape, loss = client.iteration(mode="train")
            mse_list.append(mse)
            rmse_list.append(rmse)
            mae_list.append(mae)
            mape_list.append(mape)
            loss_list.append(loss)
            log = f"Round {i} (train) | MSE: {mse:.4f} , RMSE {rmse:.4f} , MAE {mae:.4f} , MAPE {mape:.4f} , Loss {loss:.4f}"
            with open(f"./{str(args.mode)}/{str(args.num_client)}/train_{j}.txt", 'a') as file:
                file.write(log + "\n")
            client.save()
        mse, rmse, mae, mape, loss = np.mean(mse_list), np.mean(rmse_list), np.mean(mae_list), np.mean(mape_list), np.mean(loss_list)
        # server.aggregate(clients_list)
        end_time = time.time()
        time_cost = end_time - start_time
        log = f"Round {i} (train) | MSE: {mse:.4f} , RMSE {rmse:.4f} , MAE {mae:.4f} , MAPE {mape:.4f} , Loss {loss:.4f} , Time {time_cost:.1f}"
        print(log)
        with open(f"./{str(args.mode)}/{str(args.num_client)}/train.txt", 'a') as file:
            file.write(log + "\n")


        mse_list, rmse_list, mae_list, mape_list, loss_list = [], [], [], [], []
        for j in range(len(clients_list)):
            client = clients_list[j]
            mse, rmse, mae, mape, loss = client.iteration(mode="val")
            mse_list.append(mse)
            rmse_list.append(rmse)
            mae_list.append(mae)
            mape_list.append(mape)
            loss_list.append(loss)
            log = f"Round {i} (valid) | MSE: {mse:.4f} , RMSE {rmse:.4f} , MAE {mae:.4f} , MAPE {mape:.4f} , Loss {loss:.4f}"
            with open(f"./{str(args.mode)}/{str(args.num_client)}/valid_{j}.txt", 'a') as file:
                file.write(log + "\n")
        mse, rmse, mae, mape, loss = np.mean(mse_list), np.mean(rmse_list), np.mean(mae_list), np.mean(
            mape_list), np.mean(loss_list)
        log = f"Round {i} (valid) | MSE: {mse:.4f} , RMSE {rmse:.4f} , MAE {mae:.4f} , MAPE {mape:.4f} , Loss {loss:.4f}"
        print(log)
        with open(f"./{str(args.mode)}/{str(args.num_client)}/valid.txt", 'a') as file:
            file.write(log + "\n")


        mse_list, rmse_list, mae_list, mape_list, loss_list = [], [], [], [], []
        for j in range(len(clients_list)):
            client = clients_list[j]
            mse, rmse, mae, mape, loss = client.iteration(mode="test")
            mse_list.append(mse)
            rmse_list.append(rmse)
            mae_list.append(mae)
            mape_list.append(mape)
            loss_list.append(loss)
            log = f"Round {i} (test) | MSE: {mse:.4f} , RMSE {rmse:.4f} , MAE {mae:.4f} , MAPE {mape:.4f} , Loss {loss:.4f}"
            with open(f"./{str(args.mode)}/{str(args.num_client)}/test_{j}.txt", 'a') as file:
                file.write(log + "\n")
        mse, rmse, mae, mape, loss = np.mean(mse_list), np.mean(rmse_list), np.mean(mae_list), np.mean(
            mape_list), np.mean(loss_list)
        log = f"Round {i} (test) | MSE: {mse:.4f} , RMSE {rmse:.4f} , MAE {mae:.4f} , MAPE {mape:.4f} , Loss {loss:.4f}"
        print(log)
        with open(f"./{str(args.mode)}/{str(args.num_client)}/test.txt", 'a') as file:
            file.write(log + "\n")

if __name__ == '__main__':
    main()