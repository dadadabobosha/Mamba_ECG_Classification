import os
import numpy as np
import pandas as pd
from data2017_3 import get_dataloaders

# 获取数据加载器
train_loader, valid_loader, test_loader = get_dataloaders(batch_size=128)

# 获取原始数据集和对应的索引
train_dataset = train_loader.dataset
original_dataset = train_dataset.dataset
train_indices = train_dataset.indices

# 打印所有样本的文件名、数据和标签
print("检查所有样本的文件名、数据和标签：")
for i in range(len(train_indices)):
    index = train_indices[i]
    data, label = original_dataset[index]
    file_name = original_dataset.file_names[index]
    print(f"Sample {i}:")
    print(f"File: {file_name}")
    print(f"Data: {data[:10]}...")  # 打印前10个数据点，确保不输出过多信息
    print(f"Label: {label}")

# 检查所有批次的标签和对应的文件名
print("检查所有批次的标签和对应的文件名：")
for batch_idx, (data, labels) in enumerate(train_loader):
    print(f"Batch {batch_idx}:")
    for i in range(len(data)):
        index = train_indices[batch_idx * train_loader.batch_size + i]
        file_name = original_dataset.file_names[index]
        print(f"File: {file_name}, Label: {labels[i]}")
