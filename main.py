import argparse

import torch

from Server import Server
from Client import Client
from utils.dataset import graph_partition
import numpy as np
import time

def get_args():
    # follow the similar setting as https://github.com/lichuan210/FedTPS
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--dataset_path', type=str, default="../../datasets")
    parser.add_argument('--dataset', type=str, choices=["PEMS03", "PEMS04", "PEMS08", "PEMS07"],
                      default='PEMS03', help='which dataset to run')
    parser.add_argument('--num_nodes', type=int, default=170, help='num_nodes')
    parser.add_argument('--seq_len', type=int, default=12, help='input sequence length')
    parser.add_argument('--horizon', type=int, default=12, help='output sequence length')
    # -------------------------------model------------------------------------------#
    parser.add_argument('--input_dim', type=int, default=1, help='number of input channel (encoder)')
    parser.add_argument('--input_dec_dim', type=int, default=1, help='number of input channel (decoder)')
    parser.add_argument('--output_dim', type=int, default=1, help='number of output channel')
    parser.add_argument('--layer', type=int, default=1, help='number of rnn layers')
    parser.add_argument('--hidden_dim', type=int, default=64, help='number of rnn units')
    parser.add_argument('--cheb_k', type=int, default=3, help='max diffusion step or Cheb K')
    parser.add_argument('--pattern_num', type=int, default=20, help='number of meta-nodes/prototypes')
    parser.add_argument('--pattern_dim', type=int, default=64, help='dimension of meta-nodes/prototypes')
    # -------------------------------train------------------------------------------#
    parser.add_argument("--loss", type=str, default='mask_mae_loss', help="mask_mae_loss")
    parser.add_argument("--epochs", type=int, default=100, help="number of epochs of training")
    parser.add_argument("--batch_size", type=int, default=128, help="size of the batches")
    parser.add_argument("--lr", type=float, default=0.01, help="base learning rate")
    parser.add_argument("--epsilon", type=float, default=1e-3, help="optimizer epsilon")
    # -------------------------------static------------------------------------------#
    parser.add_argument('--cuda', type=str, default="0", help='which gpu to use')
    parser.add_argument("--cpu", action="store_true",default=False)
    parser.add_argument('--num_client', type=int, default=4, help="number of clients")
    args = parser.parse_args()
    return args

def main():
    args = get_args()

    server = Server(args)

    clients_list = []
    clients_data = graph_partition(args)
    for client_id, dataset in enumerate(clients_data):
        client = Client(dataset, client_id, args)
        clients_list.append(client)

    for i in range(1, 1 + args.epochs):
        server.aggregate(clients_list)

        torch.set_grad_enabled(True)
        start_time = time.time()
        mse_list, rmse_list, mae_list, mape_list, loss_list = [], [], [], [], []
        for j in range(len(clients_list)):
            client = clients_list[j]
            client.update_weight(server.W)
            mse, rmse, mae, mape, loss = client.train()
            mse_list.append(mse)
            rmse_list.append(rmse)
            mae_list.append(mae)
            mape_list.append(mape)
            loss_list.append(loss)
            log = f"Round {i} (train) | MSE: {mse:.4f} , RMSE {rmse:.4f} , MAE {mae:.4f} , MAPE {mape:.4f} , Loss {loss:.4f}"
            with open(f"train_{j}.txt", 'a') as file:
                file.write(log + "\n")
            client.save()
        mse, rmse, mae, mape, loss = np.mean(mse_list), np.mean(rmse_list), np.mean(mae_list), np.mean(mape_list), np.mean(loss_list)
        # server.aggregate(clients_list)

        end_time = time.time()
        time_cost = end_time - start_time

        log = f"Round {i} (train) | MSE: {mse:.4f} , RMSE {rmse:.4f} , MAE {mae:.4f} , MAPE {mape:.4f} , Loss {loss:.4f} , Time {time_cost:.1f}"
        print(log)
        with open("train.txt", 'a') as file:
            file.write(log + "\n")

            # if i % 5 == 0:
            if True:
                torch.set_grad_enabled(False)
                mse_list, rmse_list, mae_list, mape_list, loss_list = [], [], [], [], []
                for j in range(len(clients_list)):
                    client = clients_list[j]
                    mse, rmse, mae, mape, loss = client.eval("val")
                    mse_list.append(mse)
                    rmse_list.append(rmse)
                    mae_list.append(mae)
                    mape_list.append(mape)
                    loss_list.append(loss)
                    log = f"Round {i} (valid) | MSE: {mse:.4f} , RMSE {rmse:.4f} , MAE {mae:.4f} , MAPE {mape:.4f} , Loss {loss:.4f}"
                    with open(f"valid_{j}.txt", 'a') as file:
                        file.write(log + "\n")

                mse, rmse, mae, mape, loss = np.mean(mse_list), np.mean(rmse_list), np.mean(mae_list), np.mean(
                    mape_list), np.mean(loss_list)

                log = f"Round {i} (valid) | MSE: {mse:.4f} , RMSE {rmse:.4f} , MAE {mae:.4f} , MAPE {mape:.4f} , Loss {loss:.4f}"
                print(log)
                with open("valid.txt", 'a') as file:
                    file.write(log + "\n")

                mse_list, rmse_list, mae_list, mape_list, loss_list = [], [], [], [], []
                for j in range(len(clients_list)):
                    client = clients_list[j]
                    mse, rmse, mae, mape, loss = client.eval("test")
                    mse_list.append(mse)
                    rmse_list.append(rmse)
                    mae_list.append(mae)
                    mape_list.append(mape)
                    loss_list.append(loss)
                    log = f"Round {i} (test) | MSE: {mse:.4f} , RMSE {rmse:.4f} , MAE {mae:.4f} , MAPE {mape:.4f} , Loss {loss:.4f}"
                    with open(f"test_{j}.txt", 'a') as file:
                        file.write(log + "\n")
                mse, rmse, mae, mape, loss = np.mean(mse_list), np.mean(rmse_list), np.mean(mae_list), np.mean(
                    mape_list), np.mean(loss_list)

                log = f"Round {i} (test) | MSE: {mse:.4f} , RMSE {rmse:.4f} , MAE {mae:.4f} , MAPE {mape:.4f} , Loss {loss:.4f}"
                print(log)
                with open("test.txt", 'a') as file:
                    file.write(log + "\n")





if __name__ == '__main__':
    main()