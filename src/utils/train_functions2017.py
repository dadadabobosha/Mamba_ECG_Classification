#
# import torch
# import numpy as np
# from torch.utils.data import DataLoader
# from torch.utils.tensorboard import SummaryWriter
#
# # other libraries
# from typing import Optional
#
# from src.utils.metrics import Accuracy
# from tqdm import tqdm
#
# @torch.enable_grad()
# def train_step(
#     model: torch.nn.Module,
#     train_data: DataLoader,
#     loss: torch.nn.Module,
#     optimizer: torch.optim.Optimizer,
#     writer: SummaryWriter,
#     epoch: int,
#     device: torch.device,
#     accuracy: Accuracy = Accuracy(),
# ) -> None:
#     """
#     This function train the model.
#
#     Args:
#         model: model to train.
#         train_data: dataloader of train data.
#         loss: loss function.
#         optimizer: optimizer.
#         writer: writer for tensorboard.
#         epoch: epoch of the training.
#         device: device for running operations.
#     """
#     # Training
#     model.train()
#     losses = []
#
#     # 使用 tqdm 包装数据加载器
#     for inputs, targets in tqdm(train_data, desc=f"Training Epoch {epoch + 1}"):
#     # for inputs, targets in train_data:  # 修改为从数据加载器中解包两个值
#         inputs, targets = inputs.to(device), targets.to(device)
#
#         # forward
#         outputs = model(inputs)
#
#         # 打印 outputs 的维度
#         print(f"outputs.shape: {outputs.shape}")
#
#         # Compute loss
#         loss_value = loss(outputs, targets)
#         losses.append(loss_value.item())
#
#         optimizer.zero_grad()
#         loss_value.backward()
#         optimizer.step()
#
#         # 更新准确度
#         predictions = torch.argmax(outputs, dim=1)
#         accuracy.update(predictions, targets)
#
#     # Write to tensorboard
#     writer.add_scalar("train/loss", np.mean(losses), epoch)
#     writer.add_scalar("train/accuracy", accuracy.compute(), epoch)
#     accuracy.reset()
#
# @torch.no_grad()
# def val_step(
#     model: torch.nn.Module,
#     val_data: DataLoader,
#     loss: torch.nn.Module,
#     scheduler: Optional[torch.optim.lr_scheduler.LRScheduler],
#     writer: SummaryWriter,
#     epoch: int,
#     device: torch.device,
#     accuracy: Accuracy = Accuracy(),
# ) -> None:
#     """
#     This function validate the model.
#
#     Args:
#         model: model to validate.
#         val_data: dataloader of validation data.
#         loss: loss function.
#         scheduler: scheduler.
#         writer: writer for tensorboard.
#         epoch: epoch of the training.
#         device: device for running operations.
#     """
#     # Validation
#     model.eval()
#     losses = []
#
#     with torch.no_grad():
#         # 使用 tqdm 包装数据加载器
#         for inputs, targets in tqdm(val_data, desc=f"Validation Epoch {epoch + 1}"):
#         # for inputs, targets in val_data:  # 修改为从数据加载器中解包两个值
#             inputs, targets = inputs.to(device), targets.to(device)
#
#             # forward
#             outputs = model(inputs)
#
#             # Compute loss
#             loss_value = loss(outputs, targets)
#             losses.append(loss_value.item())
#
#             # 更新准确度
#             predictions = torch.argmax(outputs, dim=1)
#             accuracy.update(predictions, targets)
#
#     if scheduler is not None:
#         scheduler.step()
#
#     # Write to tensorboard
#     writer.add_scalar("val/loss", np.mean(losses), epoch)
#     writer.add_scalar("val/accuracy", accuracy.compute(), epoch)
#     accuracy.reset()




r"""
下面是只保留N和A两类的代码，
然后在N和A两类中分类出A类的数据
"""

import torch
import numpy as np
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from typing import Optional
from src.utils.metrics import BinaryAccuracy
from tqdm import tqdm

@torch.enable_grad()
def train_step(
    model: torch.nn.Module,
    train_data: DataLoader,
    loss: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    writer: SummaryWriter,
    epoch: int,
    device: torch.device,
    accuracy: BinaryAccuracy = BinaryAccuracy(),
) -> None:
    """
    This function trains the model.

    Args:
        model: model to train.
        train_data: dataloader of train data.
        loss: loss function.
        optimizer: optimizer.
        writer: writer for tensorboard.
        epoch: epoch of the training.
        device: device for running operations.
    """
    model.train()
    losses = []

    for inputs, targets in tqdm(train_data, desc=f"Training Epoch {epoch + 1}"):
        inputs, targets = inputs.to(device), targets.to(device)

        # forward
        outputs = model(inputs)

        # Compute loss
        loss_value = loss(outputs, targets)
        losses.append(loss_value.item())

        optimizer.zero_grad()
        loss_value.backward()
        optimizer.step()


        # 更新准确度
        accuracy.update(outputs, targets)

        print(f"Batch Loss: {loss_value.item()}")
        print(f"Batch Accuracy: {accuracy.compute()}")

    avg_loss = np.mean(losses)
    avg_accuracy = accuracy.compute()

    # Write to tensorboard
    writer.add_scalar("train/loss", avg_loss, epoch)
    writer.add_scalar("train/accuracy", avg_accuracy, epoch)
    accuracy.reset()
    return avg_loss, avg_accuracy

@torch.no_grad()
def val_step(
    model: torch.nn.Module,
    val_data: DataLoader,
    loss: torch.nn.Module,
    scheduler: Optional[torch.optim.lr_scheduler.LRScheduler],
    writer: SummaryWriter,
    epoch: int,
    device: torch.device,
    accuracy: BinaryAccuracy = BinaryAccuracy(),
) -> None:
    """
    This function validates the model.

    Args:
        model: model to validate.
        val_data: dataloader of validation data.
        loss: loss function.
        scheduler: scheduler.
        writer: writer for tensorboard.
        epoch: epoch of the training.
        device: device for running operations.
    """
    model.eval()
    losses = []

    with torch.no_grad():
        for inputs, targets in tqdm(val_data, desc=f"Validation Epoch {epoch + 1}"):
            inputs, targets = inputs.to(device), targets.to(device)

            # forward
            outputs = model(inputs)

            # Compute loss
            loss_value = loss(outputs, targets)
            losses.append(loss_value.item())

            # 更新准确度
            accuracy.update(outputs, targets)

            print(f"Validation Batch Loss: {loss_value.item()}")
            print(f"Validation Batch Accuracy: {accuracy.compute()}")
    avg_loss = np.mean(losses)
    avg_accuracy = accuracy.compute()

    if scheduler is not None:
        scheduler.step()

    # Write to tensorboard
    writer.add_scalar("val/loss", avg_loss, epoch)
    writer.add_scalar("val/accuracy", avg_accuracy, epoch)

    return avg_loss, avg_accuracy
