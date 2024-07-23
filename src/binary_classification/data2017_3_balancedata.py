import os
import numpy as np
import wfdb
import pandas as pd
from scipy.signal import butter, lfilter, resample
from sklearn.preprocessing import MinMaxScaler
import torch
from torch.utils.data import Dataset, DataLoader, random_split


# function to preprocess ECG data
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

    # passband: 0.5-45Hz
    filtered_signal = butter_bandpass_filter(ecg_signal, 0.5, 45, fs)

    # normalization
    scaler = MinMaxScaler(feature_range=(-1, 1))
    normalized_signal = scaler.fit_transform(filtered_signal.reshape(-1, 1)).reshape(-1)



    # resample to target length
    resampled_signal = resample(normalized_signal, target_length).astype(np.float32)

    # padding or truncating
    if len(resampled_signal) < target_length:
        padded_signal = np.pad(resampled_signal, (0, target_length - len(resampled_signal)), 'constant')
    else:
        padded_signal = resampled_signal[:target_length]

    return padded_signal


# filter and convert labels, only keep 'N' and 'A' labels, use one-hot encoding
def filter_and_convert_labels(data, labels):
    filtered_data = []
    filtered_labels = []
    for i, label in enumerate(labels):
        if label in ['N', 'A']:
            filtered_data.append(data[i])
            filtered_labels.append([1, 0] if label == 'A' else [0, 1])
    return filtered_data, filtered_labels


# Class for ECG dataset
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


# recordings with label 'N' are much more than 'A', here balance dataset, use _balanced.csv as the output file
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


# main function to get data loaders
def get_dataloaders(batch_size):
    # set data directory and output directory
    script_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(script_dir, 'data', 'training2017')
    output_dir = os.path.join(script_dir, 'data', 'training2017', 'output')
    os.makedirs(output_dir, exist_ok=True)

    # read label file
    label_file = os.path.join(data_dir, 'REFERENCE.csv')
    labels_df = pd.read_csv(label_file, header=None, names=['record', 'label'])

    # initialize lists to store preprocessed data and labels
    preprocessed_data = []
    preprocessed_labels = []
    file_names = []

    # iterate over all records
    for index, row in labels_df.iterrows():
        record_name = row['record']
        label = row['label']

        if label not in ['N', 'A']:
            continue

        # use WFDB library to read ECG data
        record = wfdb.rdrecord(os.path.join(data_dir, record_name))
        ecg_signal = record.p_signal[:, 0]  # only use the first channel

        # preprocess ECG data
        preprocessed_signal = preprocess_ekg_data(ecg_signal)

        # save preprocessed data
        file_path = os.path.join(output_dir, f'{record_name}.npy')
        np.save(file_path, preprocessed_signal)

        # store file name and label
        file_names.append(f'{record_name}.npy')
        preprocessed_data.append(preprocessed_signal)
        preprocessed_labels.append(label)

    # filter and convert labels
    filtered_data, filtered_labels = filter_and_convert_labels(preprocessed_data, preprocessed_labels)

    # save preprocessed data and labels to a csv file
    csv_output_path = os.path.join(output_dir, 'preprocessed_data.csv')
    csv_data = {
        'file_name': file_names,
        'label_0': [label[0] for label in filtered_labels],
        'label_1': [label[1] for label in filtered_labels]
    }
    csv_df = pd.DataFrame(csv_data)
    csv_df.to_csv(csv_output_path, index=False)

    # balance dataset
    balanced_csv_file = balance_dataset(csv_output_path)

    # create ECGDataset
    dataset = ECGDataset(balanced_csv_file, output_dir)

    # split dataset into train, validation, and test sets
    train_size = int(0.6 * len(dataset))
    valid_size = int(0.2 * len(dataset))
    test_size = len(dataset) - train_size - valid_size

    # split dataset randomly
    train_dataset, valid_dataset, test_dataset = random_split(dataset, [train_size, valid_size, test_size])

    # create DataLoader
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)

    return train_loader, valid_loader, test_loader


if __name__ == "__main__":
    train_loader, valid_loader, test_loader = get_dataloaders(batch_size=128)
    print(f"length of train dataset: {len(train_loader.dataset)}")
    print(f"length of valid dataset: {len(valid_loader.dataset)}")
    print(f"length of test dataset: {len(test_loader.dataset)}")

    # check the shape of data and label
    for data, label in train_loader:
        print(data.shape, label.shape)
        break
