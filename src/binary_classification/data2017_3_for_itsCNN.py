import os
import numpy as np
import wfdb
import pandas as pd
from scipy.signal import butter, lfilter, resample
from sklearn.preprocessing import MinMaxScaler
import torch
from torch.utils.data import Dataset, DataLoader, random_split


# 定义用于预处理ECG数据的函数
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

    # 滤波处理
    filtered_signal = butter_bandpass_filter(ecg_signal, 0.5, 45, fs)

    # 归一化处理
    scaler = MinMaxScaler(feature_range=(-1, 1))
    normalized_signal = scaler.fit_transform(filtered_signal.reshape(-1, 1)).reshape(-1)

    # 重采样处理
    resampled_signal = resample(normalized_signal, target_length).astype(np.float32)

    # 填充或截断处理
    if len(resampled_signal) < target_length:
        padded_signal = np.pad(resampled_signal, (0, target_length - len(resampled_signal)), 'constant')
    else:
        padded_signal = resampled_signal[:target_length]

    return padded_signal


# 过滤并转换标签
def filter_and_convert_labels(data, labels):
    filtered_data = []
    filtered_labels = []
    for i, label in enumerate(labels):
        if label in ['N', 'A']:
            filtered_data.append(data[i])
            filtered_labels.append([1, 0] if label == 'A' else [0, 1])
    return filtered_data, filtered_labels


# 自定义Dataset类，用于加载预处理后的ECG数据
class ECGDataset(Dataset):
    def __init__(self, csv_file, data_dir):
        self.data_dir = data_dir
        self.data_info = pd.read_csv(csv_file)
        self.file_names = self.data_info['file_name']
        self.labels = self.data_info[['label_0', 'label_1']].values

    def __len__(self):
        return len(self.file_names)

    def __getitem__(self, idx):
        file_path = os.path.join(self.data_dir, self.file_names[idx])
        data = np.load(file_path)
        label = self.labels[idx]
        return torch.tensor(data, dtype=torch.float32), torch.tensor(label, dtype=torch.float32)


# 平衡数据集
def balance_dataset(csv_file):
    data_info = pd.read_csv(csv_file)
    label_counts = data_info[['label_0', 'label_1']].value_counts().to_dict()

    min_count = min(label_counts.values())

    balanced_data_list = []

    label0_count, label1_count = 0, 0

    for _, row in data_info.iterrows():
        label = tuple(row[['label_0', 'label_1']])
        if label == (0, 1) and label0_count < min_count:
            balanced_data_list.append(row)
            label0_count += 1
        elif label == (1, 0) and label1_count < min_count:
            balanced_data_list.append(row)
            label1_count += 1

    balanced_data_info = pd.DataFrame(balanced_data_list)
    balanced_csv_file = csv_file.replace('.csv', '_balanced.csv')
    balanced_data_info.to_csv(balanced_csv_file, index=False)

    return balanced_csv_file


# 主函数
def get_dataloaders(batch_size):
    # 设置PhysioNet数据所在目录的路径
    script_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(script_dir, 'data', 'training2017')  # 替换为实际的数据目录路径
    output_dir = os.path.join(script_dir, 'data', 'training2017', 'output')  # 替换为实际的输出目录路径
    os.makedirs(output_dir, exist_ok=True)

    # 读取标签信息
    label_file = os.path.join(data_dir, 'REFERENCE.csv')  # 假设标签信息保存在REFERENCE.csv文件中
    labels_df = pd.read_csv(label_file, header=None, names=['record', 'label'])

    # 初始化用于保存预处理后的数据和标签的列表
    preprocessed_data = []
    preprocessed_labels = []
    file_names = []

    # 批量处理数据
    for index, row in labels_df.iterrows():
        record_name = row['record']
        label = row['label']

        if label not in ['N', 'A']:
            continue

        # 使用wfdb读取原始ECG信号
        record = wfdb.rdrecord(os.path.join(data_dir, record_name))
        ecg_signal = record.p_signal[:, 0]  # 假设第一个通道是需要的ECG信号

        # 预处理ECG信号
        preprocessed_signal = preprocess_ekg_data(ecg_signal)

        # 保存预处理后的信号为numpy文件
        file_path = os.path.join(output_dir, f'{record_name}.npy')
        np.save(file_path, preprocessed_signal)

        # 保存文件名和标签
        file_names.append(f'{record_name}.npy')
        preprocessed_data.append(preprocessed_signal)
        preprocessed_labels.append(label)

    # 过滤并转换标签
    filtered_data, filtered_labels = filter_and_convert_labels(preprocessed_data, preprocessed_labels)

    # 保存到CSV文件中
    csv_output_path = os.path.join(output_dir, 'preprocessed_data.csv')
    csv_data = {
        'file_name': file_names,
        'label_0': [label[0] for label in filtered_labels],
        'label_1': [label[1] for label in filtered_labels]
    }
    csv_df = pd.DataFrame(csv_data)
    csv_df.to_csv(csv_output_path, index=False)

    # 平衡数据集
    balanced_csv_file = balance_dataset(csv_output_path)

    # 创建Dataset对象
    dataset = ECGDataset(balanced_csv_file, output_dir)

    # 设置划分比例
    train_size = int(0.6 * len(dataset))
    valid_size = int(0.2 * len(dataset))
    test_size = len(dataset) - train_size - valid_size

    # 划分数据集
    train_dataset, valid_dataset, test_dataset = random_split(dataset, [train_size, valid_size, test_size])

    # 创建DataLoader
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)

    return train_loader, valid_loader, test_loader


if __name__ == "__main__":
    train_loader, valid_loader, test_loader = get_dataloaders(batch_size=128)
    print(f"训练集大小: {len(train_loader.dataset)}")
    print(f"验证集大小: {len(valid_loader.dataset)}")
    print(f"测试集大小: {len(test_loader.dataset)}")

    # 示例：从DataLoader中获取一个批次的数据
    for data, label in train_loader:
        print(data.shape, label.shape)
        break
