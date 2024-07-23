# import torch
# import torch.nn as nn
# import torch.optim as optim
# from sklearn.metrics import confusion_matrix, classification_report
# import matplotlib.pyplot as plt
# import seaborn as sns
# from data2017_3 import get_dataloaders
#
# # 定义一个更复杂的CNN模型
# class SimpleCNN(nn.Module):
#     def __init__(self):
#         super(SimpleCNN, self).__init__()
#         self.conv1 = nn.Conv1d(in_channels=1, out_channels=16, kernel_size=7, stride=1, padding=3)
#         self.conv2 = nn.Conv1d(in_channels=16, out_channels=32, kernel_size=7, stride=1, padding=3)
#         self.conv3 = nn.Conv1d(in_channels=32, out_channels=64, kernel_size=7, stride=1, padding=3)
#         self.fc1 = nn.Linear(64 * 1125, 128)
#         self.fc2 = nn.Linear(128, 2)
#         self.relu = nn.ReLU()
#         self.maxpool = nn.MaxPool1d(kernel_size=2, stride=2)
#         self.dropout = nn.Dropout(0.5)
#         self.softmax = nn.Softmax(dim=1)
#
#     def forward(self, x):
#         x = self.relu(self.conv1(x))
#         x = self.maxpool(x)
#         x = self.relu(self.conv2(x))
#         x = self.maxpool(x)
#         x = self.relu(self.conv3(x))
#         x = self.maxpool(x)
#         x = x.view(x.size(0), -1)
#         x = self.relu(self.fc1(x))
#         x = self.dropout(x)
#         x = self.fc2(x)
#         x = self.softmax(x)
#         return x
#
# # 初始化模型、损失函数和优化器
# model = SimpleCNN()
# # criterion = nn.CrossEntropyLoss()
# # optimizer = optim.Adam(model.parameters(), lr=0.001)
# #
# # # 获取数据加载器
# # train_loader, valid_loader, test_loader = get_dataloaders(batch_size=128)
#
# # 计算类别权重
# train_loader, valid_loader, test_loader = get_dataloaders(batch_size=128)
# class_counts = [0, 0]
# for _, labels in train_loader:
#     labels = torch.argmax(labels, dim=1)
#     for label in labels:
#         class_counts[label] += 1
#
# class_weights = [sum(class_counts) / count for count in class_counts]
# class_weights = torch.tensor(class_weights, dtype=torch.float32)
#
# criterion = nn.CrossEntropyLoss(weight=class_weights)
# optimizer = optim.Adam(model.parameters(), lr=0.001)
#
# # 训练函数
# def train(model, train_loader, valid_loader, criterion, optimizer, epochs=10):
#     model.train()
#     train_losses = []
#     valid_losses = []
#     valid_accuracies = []
#     for epoch in range(epochs):
#         running_loss = 0.0
#         for data, labels in train_loader:
#             # print(f"data.shape{data.shape}\ndata{data}")
#             # # plot the first batch of data
#             # if epoch == 0:  # only plot the first batch of data
#             #     plt.figure(figsize=(12, 4))
#             #     plt.plot(data[0].numpy().squeeze(), label='ECG Signal')
#             #     plt.title('ECG Signal from the First Batch')
#             #     plt.xlabel('Sample')
#             #     plt.ylabel('Amplitude')
#             #     plt.legend()
#             #     plt.show()
#             # print(f"labels.shape{labels.shape}\nlabels{labels}")
#             data = data.unsqueeze(1)  # add channel dimension
#             print(f"data.shape{data.shape}")
#
#             # labels = torch.argmax(labels, dim=1)
#
#             optimizer.zero_grad()
#             outputs = model(data)
#             # print(f"这是output{outputs}")
#             # loss = criterion(outputs, labels.argmax(dim=1))
#             loss = criterion(outputs, labels)
#             # loss = criterion(outputs.argmax(dim=1), labels.argmax(dim=1))
#
#             loss.backward()
#             optimizer.step()
#             running_loss += loss.item()
#
#         train_loss = running_loss / len(train_loader)
#         train_losses.append(train_loss)
#
#         valid_loss, valid_accuracy = evaluate(model, valid_loader, criterion)
#         valid_losses.append(valid_loss)
#         valid_accuracies.append(valid_accuracy)
#
#         print(
#             f'Epoch [{epoch + 1}/{epochs}], Train Loss: {train_loss:.4f}, Valid Loss: {valid_loss:.4f}, Valid Accuracy: {valid_accuracy:.2f}%')
#
#     return train_losses, valid_losses, valid_accuracies
#
# # 评估函数
# def evaluate(model, data_loader, criterion):
#     model.eval()
#     correct = 0
#     total = 0
#     running_loss = 0.0
#     with torch.no_grad():
#         for data, labels in data_loader:
#             # print(f"data.shape{data.shape}\ndata{data}")
#             # print(f"labels.shape{labels.shape}\nlabels{labels}")
#             data = data.unsqueeze(1)  # 添加channel维度
#             # print(f"data.shape{data.shape}\ndata{data}")
#
#             # 将独热编码标签转换为类别索引
#             # labels = torch.argmax(labels, dim=1)
#             outputs = model(data)
#             # print(f"这是output{outputs}")
#             loss = criterion(outputs, labels)
#             running_loss += loss.item()
#             _, predicted = torch.max(outputs.data, 1)
#             total += labels.size(0)
#             correct += (predicted == labels).sum().item()
#             # 打印标签
#             # print(f"Labels (batch): {labels}")
#
#     avg_loss = running_loss / len(data_loader)
#     accuracy = 100 * correct / total
#     return avg_loss, accuracy
#
# # 混淆矩阵和分类报告
# def confusion_matrix_report(model, data_loader):
#     model.eval()
#     all_labels = []
#     all_preds = []
#     with torch.no_grad():
#         for data, labels in data_loader:
#             data = data.unsqueeze(1)  # 添加channel维度
#             outputs = model(data)
#             _, predicted = torch.max(outputs.data, 1)
#             all_labels.extend(labels.cpu().numpy())
#             all_preds.extend(predicted.cpu().numpy())
#     print(f"这是all_labels{all_labels}")
#     print(f"这是all_preds{all_preds}")
#     cm = confusion_matrix(all_labels, all_preds)
#     cr = classification_report(all_labels, all_preds, target_names=['N', 'A'])
#
#     plt.figure(figsize=(10, 7))
#     sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['N', 'A'], yticklabels=['N', 'A'])
#     plt.xlabel('Predicted')
#     plt.ylabel('True')
#     plt.title('Confusion Matrix')
#     plt.show()
#
#     print("Classification Report:\n", cr)
#
# # 训练和测试模型
# train_losses, valid_losses, valid_accuracies = train(model, train_loader, valid_loader, criterion, optimizer, epochs=3)
#
# # 绘制训练和验证过程的损失和准确率曲线
# epochs = range(1, len(train_losses) + 1)
# plt.figure(figsize=(12, 4))
#
# plt.subplot(1, 2, 1)
# plt.plot(epochs, train_losses, 'b', label='Train Loss')
# plt.plot(epochs, valid_losses, 'r', label='Valid Loss')
# plt.title('Training and Validation Loss')
# plt.xlabel('Epochs')
# plt.ylabel('Loss')
# plt.legend()
#
# plt.subplot(1, 2, 2)
# plt.plot(epochs, valid_accuracies, 'b', label='Valid Accuracy')
# plt.title('Validation Accuracy')
# plt.xlabel('Epochs')
# plt.ylabel('Accuracy')
# plt.legend()
#
# plt.tight_layout()
# plt.show()
#
# # 测试模型
# test_loss, test_accuracy = evaluate(model, test_loader, criterion)
# print(f'Test Loss: {test_loss:.4f}, Test Accuracy: {test_accuracy:.2f}%')
#
# # 混淆矩阵和分类报告
# confusion_matrix_report(model, test_loader)
#
#
#
#

import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns
from data2017_3_balancedata import get_dataloaders


# 定义一个更复杂的CNN模型
class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=1, out_channels=16, kernel_size=7, stride=1, padding=3)
        self.conv2 = nn.Conv1d(in_channels=16, out_channels=32, kernel_size=7, stride=1, padding=3)
        self.conv3 = nn.Conv1d(in_channels=32, out_channels=64, kernel_size=7, stride=1, padding=3)
        self.fc1 = nn.Linear(64 * 1125, 128)
        self.fc2 = nn.Linear(128, 2)
        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool1d(kernel_size=2, stride=2)
        self.dropout = nn.Dropout(0.6)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.maxpool(x)
        x = self.relu(self.conv2(x))
        x = self.maxpool(x)
        x = self.relu(self.conv3(x))
        x = self.maxpool(x)
        x = x.view(x.size(0), -1)
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.softmax(x)
        return x


# 定义将标签转换为独热编码的辅助函数
def to_one_hot(labels, num_classes):
    return torch.eye(num_classes)[labels]


# 初始化模型、损失函数和优化器
model = SimpleCNN()

# 计算类别权重
train_loader, valid_loader, test_loader = get_dataloaders(batch_size=128)
class_counts = [0, 0]
for _, labels in train_loader:
    labels = torch.argmax(labels, dim=1)
    for label in labels:
        class_counts[label] += 1

class_weights = [sum(class_counts) / count for count in class_counts]
class_weights = torch.tensor(class_weights, dtype=torch.float32)

criterion = nn.CrossEntropyLoss(weight=class_weights)
# optimizer = optim.Adam(model.parameters(), lr=0.001)
optimizer = optim.Adam(model.parameters(), lr=0.0001, weight_decay=1e-5)



# 训练函数
def train(model, train_loader, valid_loader, criterion, optimizer, epochs=10):
    model.train()
    train_losses = []
    valid_losses = []
    valid_accuracies = []
    for epoch in range(epochs):
        running_loss = 0.0
        for data, labels in train_loader:
            # print(f"data.shape: {data.shape}")

            # 检查并调整数据形状
            data = data.unsqueeze(1)  # 添加channel维度

            # 将独热编码标签转换为类别索引
            labels_idx = torch.argmax(labels, dim=1)

            # # 绘制数据
            # if epoch == 0:  # 只绘制第一个epoch的第一批数据
            #     plt.figure(figsize=(12, 4))
            #     plt.plot(data[0].cpu().numpy().squeeze(), label='ECG Signal')
            #     plt.title('ECG Signal from the First Batch')
            #     plt.xlabel('Sample')
            #     plt.ylabel('Amplitude')
            #     plt.legend()
            #     plt.show()

            optimizer.zero_grad()
            outputs = model(data)
            loss = criterion(outputs, labels_idx)  # 确保使用整数标签
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        train_loss = running_loss / len(train_loader)
        train_losses.append(train_loss)

        valid_loss, valid_accuracy, _ = evaluate(model, valid_loader, criterion)
        valid_losses.append(valid_loss)
        valid_accuracies.append(valid_accuracy)

        print(
            f'Epoch [{epoch + 1}/{epochs}], Train Loss: {train_loss:.4f}, Valid Loss: {valid_loss:.4f}, Valid Accuracy: {valid_accuracy:.2f}%')

    return train_losses, valid_losses, valid_accuracies


# 评估函数
def evaluate(model, data_loader, criterion):
    model.eval()
    correct = 0
    total = 0
    running_loss = 0.0
    correct_per_class = [0, 0]
    total_per_class = [0, 0]
    with torch.no_grad():
        for data, labels in data_loader:
            data = data.unsqueeze(1)  # 添加channel维度

            # 将独热编码标签转换为类别索引
            labels_idx = torch.argmax(labels, dim=1)

            outputs = model(data)
            loss = criterion(outputs, labels_idx)  # 确保使用整数标签
            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels_idx.size(0)
            correct += (predicted == labels_idx).sum().item()

            for i in range(len(labels_idx)):
                label = labels_idx[i].item()
                total_per_class[label] += 1
                if predicted[i].item() == label:
                    correct_per_class[label] += 1

    avg_loss = running_loss / len(data_loader)
    accuracy = 100 * correct / total
    accuracy_per_class = [100 * correct_per_class[i] / total_per_class[i] for i in range(2)]
    return avg_loss, accuracy, accuracy_per_class


# 混淆矩阵和分类报告
def confusion_matrix_report(model, data_loader):
    model.eval()
    all_labels = []
    all_preds = []
    correct_per_class = [0, 0]
    total_per_class = [0, 0]
    with torch.no_grad():
        for data, labels in data_loader:
            data = data.unsqueeze(1)  # 添加channel维度

            # 将独热编码标签转换为类别索引
            labels_idx = torch.argmax(labels, dim=1)
            outputs = model(data)
            print(f"outputs\n{outputs}")
            _, predicted = torch.max(outputs.data, 1)
            all_labels.extend(labels_idx.cpu().numpy())
            all_preds.extend(predicted.cpu().numpy())

            for i in range(len(labels_idx)):
                label = labels_idx[i].item()
                total_per_class[label] += 1
                if predicted[i].item() == label:
                    correct_per_class[label] += 1

    cm = confusion_matrix(all_labels, all_preds)
    cr = classification_report(all_labels, all_preds, target_names=['A', 'N'])
    accuracy_per_class = [100 * correct_per_class[i] / total_per_class[i] for i in range(2)]

    plt.figure(figsize=(10, 7))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['A', 'N'], yticklabels=['A', 'N'])
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    plt.show()

    print("Classification Report:\n", cr)
    print(f"Accuracy per class: {accuracy_per_class}")


# 训练和测试模型
train_losses, valid_losses, valid_accuracies = train(model, train_loader, valid_loader, criterion, optimizer, epochs=100)

# 绘制训练和验证过程的损失和准确率曲线
epochs = range(1, len(train_losses) + 1)
plt.figure(figsize=(12, 4))

plt.subplot(1, 2, 1)
plt.plot(epochs, train_losses, 'b', label='Train Loss')
plt.plot(epochs, valid_losses, 'r', label='Valid Loss')
plt.title('Training and Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(epochs, valid_accuracies, 'b', label='Valid Accuracy')
plt.title('Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()

plt.tight_layout()
plt.show()

# 测试模型
test_loss, test_accuracy, test_accuracy_per_class = evaluate(model, test_loader, criterion)
print(f'Test Loss: {test_loss:.4f}, Test Accuracy: {test_accuracy:.2f}%')
print(f'Test Accuracy per class: {test_accuracy_per_class}')

# 混淆矩阵和分类报告
confusion_matrix_report(model, test_loader)
