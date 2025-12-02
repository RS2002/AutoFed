import pickle
import numpy as np
from torch.utils.data import Dataset
import math

class Traffic_Dataset(Dataset):
    def __init__(self, data, labels):
        self.data = data
        self.labels = labels

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data[idx]
        label = self.labels[idx]
        return sample, label

def sliding_window(arr, l, r, window_size, slide):
    # 提取样本范围
    selected_samples = arr[l:r]
    # 计算窗口的数量
    num_windows = (len(selected_samples) - window_size) // slide + 1

    # 使用滑动窗口提取样本
    result = []
    for i in range(num_windows):
        start_idx = i * slide
        end_idx = start_idx + window_size
        window = selected_samples[start_idx:end_idx]
        result.append(window)

    return np.array(result)


# def load(data_path = "./dataset/dataset.pkl", history=12, horizon=12, slide=1):
#     # train_set, valid_set, test_set = [], [], []
#
#     train_x, valid_x, test_x = [], [], []
#     train_y, valid_y, test_y = [], [], []
#     history, horizon, slide = int(history), int(horizon), int(slide)
#
#     with open(data_path, 'rb') as file:
#         dataset = pickle.load(file)
#     for i in range(len(dataset)): # client (company-region)
#         data_temp = dataset[i]['companies']
#         # print(data_temp.keys())
#         for j in range(len(data_temp.values())): # day
#             v = list(data_temp.values())[j]['data']
#             v[...,0] = (v[...,0] - np.min(v[...,0],axis=1,keepdims=True)) / (np.max(v[...,0],axis=1,keepdims=True) - np.min(v[...,0],axis=1,keepdims=True) +1e-5)
#             # if i==0:
#             #     train_set.append(v)
#             # elif i==len(dataset)-2:
#             #     valid_set.append(v)
#             # elif i==len(dataset)-1:
#             #     test_set.append(v)
#             # else:
#             #     train_set[j] = np.concatenate([train_set[j],v],axis=0)
#
#             l = 0
#             r = v.shape[0] - horizon
#             x = sliding_window(v, l, r, history, slide)
#             y = sliding_window(v, l+history, r+horizon, horizon, slide)
#             y = y[...,-1:]
#
#             if i==0:
#                 train_x.append(x)
#                 train_y.append(y)
#             elif i==len(dataset)-2:
#                 valid_x.append(x)
#                 valid_y.append(y)
#             elif i==len(dataset)-1:
#                 test_x.append(x)
#                 test_y.append(y)
#             else:
#                 train_x[j] = np.concatenate([train_x[j],x],axis=0)
#                 train_y[j] = np.concatenate([train_y[j],y],axis=0)
#
#     # return train_set, valid_set, test_set
#     return train_x, train_y, valid_x, valid_y, test_x, test_y

def load(data_path = "../dataset/dataset.pkl", history=12, horizon=12, slide=1, client_num=5, mode=0):

    train_x, valid_x, test_x = [], [], []
    train_y, valid_y, test_y = [], [], []
    history, horizon, slide = int(history), int(horizon), int(slide)

    with open(data_path, 'rb') as file:
        dataset = pickle.load(file)

    for i in range(len(dataset)): # day

        data_temp = dataset[i]['companies']
        for j in range(len(data_temp.values())): # client (company-region)

            if j % 2 == mode:
                continue


            v = list(data_temp.values())[j]['data']
            v[...,0] = (v[...,0] - np.min(v[...,0],axis=1,keepdims=True)) / (np.max(v[...,0],axis=1,keepdims=True) - np.min(v[...,0],axis=1,keepdims=True) +1e-5)

            l = 0
            r = v.shape[0] - horizon
            x = sliding_window(v, l, r, history, slide)
            y = sliding_window(v, l+history, r+horizon, horizon, slide)
            y = y[...,-1:]

            if i==0:
                train_x.append(x)
                train_y.append(y)
            elif i==len(dataset)-2:
                valid_x.append(x)
                valid_y.append(y)
            elif i==len(dataset)-1:
                test_x.append(x)
                test_y.append(y)
            else:
                train_x[j//2] = np.concatenate([train_x[j//2],x],axis=0)
                train_y[j//2] = np.concatenate([train_y[j//2],y],axis=0)

    if client_num == len(train_x):
        return train_x, train_y, valid_x, valid_y, test_x, test_y

    train_x2, valid_x2, test_x2 = [], [], []
    train_y2, valid_y2, test_y2 = [], [], []

    if client_num < len(train_x):
        group_num = math.ceil(len(train_x)/client_num)
        for i in range(client_num):
            if i != client_num - 1:
                train_x2.append(np.concatenate([train_x[j] for j in range(i*group_num, (i+1)*group_num)],axis=2))
                valid_x2.append(np.concatenate([valid_x[j] for j in range(i*group_num, (i+1)*group_num)],axis=2))
                test_x2.append(np.concatenate([test_x[j] for j in range(i*group_num, (i+1)*group_num)],axis=2))
                train_y2.append(np.concatenate([train_y[j] for j in range(i*group_num, (i+1)*group_num)],axis=2))
                valid_y2.append(np.concatenate([valid_y[j] for j in range(i*group_num, (i+1)*group_num)],axis=2))
                test_y2.append(np.concatenate([test_y[j] for j in range(i*group_num, (i+1)*group_num)],axis=2))
            else:
                train_x2.append(np.concatenate([train_x[j] for j in range(len(train_x)-group_num, len(train_x))],axis=2))
                valid_x2.append(np.concatenate([valid_x[j] for j in range(len(train_x)-group_num, len(train_x))],axis=2))
                test_x2.append(np.concatenate([test_x[j] for j in range(len(train_x)-group_num, len(train_x))],axis=2))
                train_y2.append(np.concatenate([train_y[j] for j in range(len(train_x)-group_num, len(train_x))],axis=2))
                valid_y2.append(np.concatenate([valid_y[j] for j in range(len(train_x)-group_num, len(train_x))],axis=2))
                test_y2.append(np.concatenate([test_y[j] for j in range(len(train_x)-group_num, len(train_x))],axis=2))

    else:
        divide_num = math.ceil(client_num/len(train_x))
        mod = client_num % len(train_x)
        for i in range(len(train_x)):
            if i < mod or mod == 0:
                group = divide_num
            else:
                group = divide_num -1
            node_num = train_x[i].shape[2] // group
            for j in range(group):
                if j != group - 1:
                    train_x2.append(train_x[i][:,:,j*node_num:(j+1)*node_num])
                    valid_x2.append(valid_x[i][:,:,j*node_num:(j+1)*node_num])
                    test_x2.append(test_x[i][:,:,j*node_num:(j+1)*node_num])
                    train_y2.append(train_y[i][:,:,j*node_num:(j+1)*node_num])
                    valid_y2.append(valid_y[i][:,:,j*node_num:(j+1)*node_num])
                    test_y2.append(test_y[i][:,:,j*node_num:(j+1)*node_num])
                else:
                    train_x2.append(train_x[i][:, :, -j * node_num:])
                    valid_x2.append(valid_x[i][:, :, -j * node_num:])
                    test_x2.append(test_x[i][:, :, -j * node_num:])
                    train_y2.append(train_y[i][:, :, -j * node_num:])
                    valid_y2.append(valid_y[i][:, :, -j * node_num:])
                    test_y2.append(test_y[i][:, :, -j * node_num:])

    return train_x2, train_y2, valid_x2, valid_y2, test_x2, test_y2
# test
if __name__ == '__main__':
    # train_set, valid_set, test_set = load()
    # print(len(train_set))
    # print(train_set[0].shape)
    # print(len(valid_set))
    # print(valid_set[0].shape)
    # print(len(test_set))
    # print(test_set[0].shape)

    train_x, train_y, valid_x, valid_y, test_x, test_y = load(client_num=12)
    print(len(train_x), len(train_y))
    print(train_x[0].shape, train_y[0].shape)
    print(len(valid_x), len(valid_y))
    print(valid_x[0].shape, valid_y[0].shape)
    print(len(test_x), len(test_y))
    print(test_x[0].shape, test_y[0].shape)