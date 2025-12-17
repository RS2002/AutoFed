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


def load(data_path = "../dataset/dataset.pkl", history=12, horizon=12, slide=1, client_num=5, mode=0):

    train_x, valid_x, test_x = [], [], []
    train_y, valid_y, test_y = [], [], []
    history, horizon, slide = int(history), int(horizon), int(slide)

    with open(data_path, 'rb') as file:
        dataset = pickle.load(file)

    for i in range(len(dataset)): # day

        data_temp = dataset[i]['companies']
        # print(data_temp.keys())

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
                if mode !=2:
                    train_x[j//2] = np.concatenate([train_x[j//2],x],axis=0)
                    train_y[j//2] = np.concatenate([train_y[j//2],y],axis=0)
                else:
                    train_x[j] = np.concatenate([train_x[j],x],axis=0)
                    train_y[j] = np.concatenate([train_y[j],y],axis=0)

    return train_x, train_y, valid_x, valid_y, test_x, test_y


# test
if __name__ == '__main__':
    # train_set, valid_set, test_set = load()
    # print(len(train_set))
    # print(train_set[0].shape)
    # print(len(valid_set))
    # print(valid_set[0].shape)
    # print(len(test_set))
    # print(test_set[0].shape)

    train_x, train_y, valid_x, valid_y, test_x, test_y = load(client_num=4)
    print(len(train_x), len(train_y))
    print(train_x[0].shape, train_y[0].shape)
    print(len(valid_x), len(valid_y))
    print(valid_x[0].shape, valid_y[0].shape)
    print(len(test_x), len(test_y))
    print(test_x[0].shape, test_y[0].shape)