
r"""
下面是只保留N和A两类的代码，然后将数据预处理后保存为.npy文件，以便后续快速加载
然后在N和A两类中分类出A类的数据，将其标签改为1，N类的数据标签改为0
"""


# deep learning libraries
import torch
import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from scipy.signal import butter, lfilter, resample

# plotting libraries
import matplotlib.pyplot as plt

# other libraries
import os
import wfdb
import zipfile
import requests
from typing import List, Dict

from tqdm import tqdm  # Import tqdm for progress bar

pd.set_option("future.no_silent_downcasting", True)

class EKGDataset(Dataset):
    def __init__(self, X: List[str], y: List[int], path: str) -> None:
        self.X = X
        self.y = torch.tensor(y, dtype=torch.float32) #  原先是long改为 float32 以适应 BCEWithLogitsLoss
        self._path = path
        # self._processed_data_path = processed_data_path

    def __len__(self) -> int:
        return len(self.X)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        # file_path = os.path.join(self._processed_data_path, f"{self.X[index].split('/')[-1].replace('.hea', '')}.npy")
        file_path = os.path.join(self._path, self.X[index])
        # print(f"Processing file: {file_path}")
        ekg_data = wfdb.rdsamp(file_path)[0]  # 读取ECG数据
        ekg_data = preprocess_ekg_data(np.array(ekg_data))  # 预处理数据
        return torch.tensor(ekg_data, dtype=torch.double).unsqueeze(0), self.y[index].unsqueeze(0)

def filter_and_convert_labels(X, y):
    filtered_X, filtered_y = [], []
    for i, label in enumerate(y):
        if label in ['N', 'A']:
            filtered_X.append(X[i])
            filtered_y.append(1 if label == 'A' else 0)
    return filtered_X, filtered_y

def preprocess_ekg_data(ecg_signal, target_length=9000, fs=300):
    def butter_bandpass(lowcut, highcut, fs, order=5):
        nyq = 0.5 * fs
        low = lowcut / nyq
        high = highcut / nyq
        b, a = butter(order, [low, high], btype='band')
        return b, a

    def butter_bandpass_filter(data, lowcut, highcut, fs, order=5):
        b, a = butter_bandpass(lowcut, highcut, fs, order=order)
        y = lfilter(b, a, data)
        return y

    filtered_signal = butter_bandpass_filter(ecg_signal, 0.5, 45, fs)

    scaler = StandardScaler()
    normalized_signal = scaler.fit_transform(filtered_signal.reshape(-1, 1)).flatten()

    resampled_signal = resample(normalized_signal, target_length).astype(np.float32)  # 确保数据类型为 float32

    if len(resampled_signal) < target_length:
        padded_signal = np.pad(resampled_signal, (0, target_length - len(resampled_signal)), 'constant')
    else:
        padded_signal = resampled_signal[:target_length]

    return padded_signal

def load_ekg_data2017(path: str, batch_size: int = 128, shuffle: bool = True,
                      drop_last: bool = False, num_workers: int = 0):
    if not os.path.isdir(f"{path}"):
        os.makedirs(f"{path}")
        print("NO data, need to download")
        #download_data(path)

    Y = pd.read_csv(path + "REFERENCE.csv", header=None)
    X = Y[0].to_numpy()
    y = Y[1].to_numpy()

    X, y = filter_and_convert_labels(X, y)

    # preprocess_and_save_all_data(X, path, processed_data_path)

    # 将文件名列表和标签分割成训练集、验证集和测试集
    train_ratio, val_ratio = 0.7, 0.15
    train_end = int(len(X) * train_ratio)
    val_end = int(len(X) * (train_ratio + val_ratio))

    X_train, y_train = X[:train_end], y[:train_end]
    X_val, y_val = X[train_end:val_end], y[train_end:val_end]
    X_test, y_test = X[val_end:], y[val_end:]

    train_dataset = EKGDataset(X_train, y_train, path)
    val_dataset = EKGDataset(X_val, y_val, path)
    test_dataset = EKGDataset(X_test, y_test, path)

    # Create dataloaders
    train_dataloader: DataLoader = DataLoader(train_dataset, batch_size=batch_size, shuffle=shuffle,
                                              num_workers=num_workers, drop_last=drop_last)
    val_dataloader: DataLoader = DataLoader(val_dataset, batch_size=batch_size, shuffle=shuffle,
                                            num_workers=num_workers, drop_last=drop_last)
    test_dataloader: DataLoader = DataLoader(test_dataset, batch_size=batch_size, shuffle=shuffle,
                                             num_workers=num_workers, drop_last=drop_last)

    return train_dataloader, val_dataloader, test_dataloader

if __name__ == "__main__":
    train_loader, val_loader, test_loader = load_ekg_data2017("./data/training2017/")
    print(train_loader)



# r"""
# 下面的是我尝试把preprocess_ekg_data从__getitem__中拿出来放到load_ekg_data2017里来保证每个record只会滤波一次
# """
#
# # deep learning libraries
# import torch
# import numpy as np
# import pandas as pd
# from torch.utils.data import Dataset, DataLoader
# from sklearn.preprocessing import LabelEncoder
# from sklearn.preprocessing import StandardScaler
# from scipy.signal import butter, lfilter, resample
#
# # plotting libraries
# import matplotlib.pyplot as plt
#
# # other libraries
# import os
# import wfdb
# import zipfile
# import requests
# from typing import List, Dict
#
# from tqdm import tqdm  # Import tqdm for progress bar
#
# pd.set_option("future.no_silent_downcasting", True)
#
# class EKGDataset(Dataset):
#     def __init__(self, X: List[str], y: List[int], path: str) -> None:
#         self.X = X
#         self.y = torch.tensor(y, dtype=torch.float32)
#         self._path = path
#
#     def __len__(self) -> int:
#         return len(self.X)
#
#     def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
#         file_path = os.path.join(self._path, self.X[index])
#         ekg_data = np.load(file_path)  # 直接加载预处理后的数据
#         return torch.tensor(ekg_data, dtype=torch.double).unsqueeze(0), self.y[index].unsqueeze(0)
#
# def filter_and_convert_labels(X, y):
#     filtered_X, filtered_y = [], []
#     for i, label in enumerate(y):
#         if label in ['N', 'A']:
#             filtered_X.append(X[i])
#             filtered_y.append(1 if label == 'A' else 0)
#     return filtered_X, filtered_y
#
# def preprocess_ekg_data(ecg_signal, target_length=9000, fs=300):
#     def butter_bandpass(lowcut, highcut, fs, order=5):
#         nyq = 0.5 * fs
#         low = lowcut / nyq
#         high = highcut / nyq
#         b, a = butter(order, [low, high], btype='band')
#         return b, a
#
#     def butter_bandpass_filter(data, lowcut, highcut, fs, order=5):
#         b, a = butter_bandpass(lowcut, highcut, fs, order=order)
#         y = lfilter(b, a, data)
#         return y
#
#     filtered_signal = butter_bandpass_filter(ecg_signal, 0.5, 45, fs)
#
#     scaler = StandardScaler()
#     normalized_signal = scaler.fit_transform(filtered_signal.reshape(-1, 1)).flatten()
#
#     resampled_signal = resample(normalized_signal, target_length).astype(np.float32)  # 确保数据类型为 float32
#
#     if len(resampled_signal) < target_length:
#         padded_signal = np.pad(resampled_signal, (0, target_length - len(resampled_signal)), 'constant')
#     else:
#         padded_signal = resampled_signal[:target_length]
#
#     return padded_signal
#
# def load_ekg_data2017(path: str, batch_size: int = 128, shuffle: bool = True,
#                       drop_last: bool = False, num_workers: int = 0):
#     if not os.path.isdir(f"{path}"):
#         os.makedirs(f"{path}")
#         print("NO data, need to download")
#         #download_data(path)
#
#     Y = pd.read_csv(path + "REFERENCE.csv", header=None)
#     X = Y[0].to_numpy()
#     y = Y[1].to_numpy()
#
#     X, y = filter_and_convert_labels(X, y)
#
#     processed_X = [preprocess_ekg_data(wfdb.rdsamp(os.path.join(path, x))[0]) for x in tqdm(X, desc="Preprocessing EKG data")]
#
#     # 将预处理后的数据分割成训练集、验证集和测试集
#     train_ratio, val_ratio = 0.7, 0.15
#     train_end = int(len(processed_X) * train_ratio)
#     val_end = int(len(processed_X) * (train_ratio + val_ratio))
#
#     X_train, y_train = processed_X[:train_end], y[:train_end]
#     X_val, y_val = processed_X[train_end:val_end], y[train_end:val_end]
#     X_test, y_test = processed_X[val_end:], y[val_end:]
#
#
#     train_dataset = EKGDataset(X_train, y_train, path)
#     val_dataset = EKGDataset(X_val, y_val, path)
#     test_dataset = EKGDataset(X_test, y_test, path)
#
#     # Create dataloaders
#     train_dataloader: DataLoader = DataLoader(train_dataset, batch_size=batch_size, shuffle=shuffle,
#                                               num_workers=num_workers, drop_last=drop_last)
#     val_dataloader: DataLoader = DataLoader(val_dataset, batch_size=batch_size, shuffle=shuffle,
#                                             num_workers=num_workers, drop_last=drop_last)
#     test_dataloader: DataLoader = DataLoader(test_dataset, batch_size=batch_size, shuffle=shuffle,
#                                              num_workers=num_workers, drop_last=drop_last)
#
#     return train_dataloader, val_dataloader, test_dataloader



