# AutoFed

**Article:** Zijian Zhao, Yitong Shang, Sen Li*, "Stop Adapting Models by Hand: A Personalized Federated Traffic Prediction Framework via Auto Prefix" (in preparation)

This repository builds upon the work of [lichuan210/FedTPS: PyTorch Implementation of "Traffic Pattern Sharing for Federated Traffic Flow Prediction with Personalization"](https://github.com/lichuan210/FedTPS) and [LeiBAI/AGCRN: Adaptive Graph Convolutional Recurrent Network](https://github.com/LeiBAI/AGCRN).

An extended version of this work will be available in [AutoFed2](https://github.com/RS2002/AutoFed2).

## 1. Workflow

![](./img/main.png)

## 2. Dataset

For the experimental setup, we utilize the configurations from [lichuan210/FedTPS](https://github.com/lichuan210/FedTPS), specifically using the PEMS03, PEMS04, PEMS07, and PEMS08 datasets.

## 3. How to Run

To execute the training process, use the following command:

```shell
python train.py --dataset <dataset_name> --num_client <client_amount>
```

## 4. Parameters

The trained parameters and log files are located in the `./parameters` folder.

## 5. Citation

