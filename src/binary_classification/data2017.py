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
# from tqdm import tqdm  # Import tqdm for progress bar#进度条
#
# pd.set_option("future.no_silent_downcasting", True)
#
#
# class EKGDataset(Dataset):
#
#     def __init__(self, X: List[str], y: List[str], path: str) -> None:
#         self._path = path
#         self.X = X
#
#         # Create a LabelEncoder object
#         self._le = LabelEncoder()
#
#         # Fit the LabelEncoder to the labels and transform the labels to integers
#         self.y = torch.tensor(self._le.fit_transform(y), dtype=torch.long)
#
#     def __len__(self) -> int:
#         return len(self.X)
#
#     def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
#         return (
#             torch.tensor(self.load_raw_data(index), dtype=torch.double).unsqueeze(0),  # 增加一个维度作为通道数
#             self.y[index],
#         )
#
#     def load_raw_data(self, index: int):
#         file_path = self._path + self.X[index]
#         print(f"Processing file: {file_path}")
#         ekg_data = wfdb.rdsamp(self._path + self.X[index])[0]
#         return preprocess_ekg_data(np.array(ekg_data))
#         # return np.array(ekg_data)
#
#
# def preprocess_ekg_data(ecg_signal, target_length=9000, fs=300):
#     """
#     Preprocess ECG data:
#     1. Bandpass filter: 0.5-45Hz
#     2. Standardization: Zero Mean, Unit Variance
#     3. Resample: downsample
#     4. Padding: zero-padded to a fixed length
#     """
#     # 1. Bandpass filter
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
#     # 2. Standardization
#     scaler = StandardScaler()
#     normalized_signal = scaler.fit_transform(filtered_signal.reshape(-1, 1)).flatten()
#
#     # 3. Resample
#     resampled_signal = resample(normalized_signal, target_length)
#
#     # 4. Padding
#     if len(resampled_signal) < target_length:
#         padded_signal = np.pad(resampled_signal, (0, target_length - len(resampled_signal)), 'constant')
#     else:
#         padded_signal = resampled_signal[:target_length]
#
#
#     return padded_signal
#
#
# def load_ekg_data2017(path: str, batch_size: int = 128, shuffle: bool = True,
#                   drop_last: bool = False, num_workers: int = 0):
#     if not os.path.isdir(f"{path}"):
#         os.makedirs(f"{path}")
#         print("NO data, need to download")
#         # download_data(path)
#
#     # 读取标签文件
#     Y = pd.read_csv(path + "REFERENCE.csv", header=None)
#     X = Y[0].to_numpy()  # 文件名列表
#     y = Y[1].to_numpy()  # 标签列表
#
#     # 将文件名列表和标签分割成训练集、验证集和测试集
#     train_ratio, val_ratio = 0.7, 0.15
#     train_end = int(len(X) * train_ratio)
#     val_end = int(len(X) * (train_ratio + val_ratio))
#
#     X_train, y_train = X[:train_end], y[:train_end]
#     X_val, y_val = X[train_end:val_end], y[train_end:val_end]
#     X_test, y_test = X[val_end:], y[val_end:]
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
#
#
# def plot_ekg(dataloader: DataLoader, sampling_rate: int = 300, num_plots: int = 5) -> None:
#     """
#     Plot EKG signals from a dataloader.
#
#     Args:
#         dataloader (DataLoader): The dataloader containing the EKG signals and labels.
#         sampling_rate (int, optional): The sampling rate of the EKG signals. Defaults to 300.
#         num_plots (int, optional): The number of EKG signals to plot. Defaults to 5.
#     """
#
#     # Get a batch of data
#     ekg_signals, labels = next(iter(dataloader))
#
#     # Define the grid and colors
#     color_major = (1, 0, 0)
#     color_minor = (1, 0.7, 0.7)
#     color_line = (0, 0, 0.7)
#
#     # Plot the first `num_plots` EKG signals
#     for i in range(num_plots):
#         # Convert tensor to numpy array and select all leads
#         signal = ekg_signals[i].numpy()
#
#         fig, axes = plt.subplots(signal.shape[1], 1, figsize=(10, 10), sharex=True)
#
#         for c in np.arange(signal.shape[1]):
#             # Set grid
#             axes[c].grid(True, which="both", color=color_major, linestyle="-", linewidth=0.5)
#             axes[c].minorticks_on()
#             axes[c].grid(which="minor", linestyle=":", linewidth=0.5, color=color_minor)
#
#             # Plot EKG signal in blue
#             axes[c].plot(signal[:, c], color=color_line)
#
#             # If it's not the last subplot, remove the x-axis label
#             if c < signal.shape[1] - 1:
#                 axes[c].set_xticklabels([])
#             else:
#                 # Set x-ticks for the last subplot
#                 axes[c].set_xticks(np.arange(0, len(signal[:, c]), step=sampling_rate))
#                 axes[c].set_xticklabels(np.arange(0, len(signal[:, c]) / sampling_rate, step=1))
#
#         # Reduce the vertical distance between subplots
#         plt.subplots_adjust(hspace=0.5)
#
#         # Set y label in the middle left
#         fig.text(0.04, 0.5, "Amplitude", va="center", rotation="vertical")
#
#         # Set title for the entire figure
#         axes[0].set_title(f"EKG Signal {i+1}, Label: {labels[i]}")
#
#         # Set x label
#         plt.xlabel("Time (seconds)")
#         plt.tight_layout(pad=4, w_pad=1.0, h_pad=0.1)
#         plt.show()
#
#
# def download_data(path: str) -> None:
#     url: str = "https://physionet.org/static/published-projects/challenge-2017/physionet.org-challenge-2017-1.0.0.zip"
#     target_path: str = path + "/challenge-2017.zip"
#     response: requests.Response = requests.get(url, stream=True)
#     if response.status_code == 200:
#         with open(target_path, "wb") as f:
#             f.write(response.raw.read())
#
#     with zipfile.ZipFile(target_path, "r") as zip_ref:
#         zip_ref.extractall(path)
#     os.remove(target_path)
#
#
# if __name__ == "__main__":
#     train_loader, val_loader, test_loader = load_ekg_data2017("./data/training2017/")
#     plot_ekg(train_loader)

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
        print(f"Processing file: {file_path}")
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

# def preprocess_and_save_all_data(X: List[str], path: str, processed_data_path: str) -> None:
#     if not os.path.exists(processed_data_path):
#         os.makedirs(processed_data_path)
#
#     for file_name in tqdm(X, desc="Preprocessing and saving all data"):
#         file_path = path + file_name
#         processed_file_path = os.path.join(processed_data_path, f"{file_name.split('/')[-1].replace('.hea', '')}.npy")
#
#         if not os.path.exists(processed_file_path):
#             ekg_data = wfdb.rdsamp(file_path)[0]
#             ekg_data = preprocess_ekg_data(np.array(ekg_data))
#             np.save(processed_file_path, ekg_data)

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