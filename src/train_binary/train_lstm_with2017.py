# deep learning libraries
import torch
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader


# other libraries
from tqdm.auto import tqdm

# own modules
from src.utils.train_functions import train_step, val_step
from src.utils.metrics import BinaryAccuracy


from src.binary_classification.data2017_3_balancedata import (
    get_dataloaders,
)
from src.utils.torchutils import set_seed, save_model

from src.modules.lstm import LSTM
import matplotlib.pyplot as plt

# set device
device: torch.device = (
    torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
)

# set all seeds and set number of threads
set_seed(42)
torch.set_num_threads(8)

# static variables
DATA_PATH: str = "./data/"
N_CLASSES: int = 2


def main() -> None:
    """
    This function is the main program for the training.
    """
    # hyperparameters
    epochs: int = 20
    lr: float = 5e-3
    batch_size: int = 128

    # Mamba Hyperparameters
    n_layers: int = 1
    hidden_size: int = 64
    bidirectional: bool = True
    gamma: float = 0.8
    step_size: int = 10
    # empty nohup file
    open("nohup.out", "w").close()

    # load data
    train_data: DataLoader
    val_data: DataLoader
    train_data, val_data, test_dataset = get_dataloaders(batch_size=batch_size)

    torch.save(test_dataset,
               'C:\\wenjian\\MasterArbeit\\Code\\repo\\ECG+SSM\\Mamba-Biometric-EKG-Analysis-Technology-MambaBEAT-master\\src\\test_dataset.pth')




    # define name and writer
    name: str = f"binary_{'Bi'*bidirectional}LSTM723922"
    writer: SummaryWriter = SummaryWriter(f"runs/{name}")
    inputs: torch.Tensor = next(iter(train_data))[0]

    # define model
    model: torch.nn.Module = (
        LSTM(inputs.size(2), hidden_size, n_layers, N_CLASSES, bidirectional)
        .to(device)
        .to(torch.double)
    )

    # define loss and optimizer
    loss: torch.nn.Module = torch.nn.CrossEntropyLoss
    optimizer: torch.optim.Optimizer = torch.optim.AdamW(model.parameters(), lr=lr)

    # compute the accuracy

    accuracy: BinaryAccuracy = BinaryAccuracy()

    # define an empty scheduler
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer, step_size=step_size, gamma=gamma
    )
    # Train the model
    try:

        train_losses = []
        train_accuracies = []
        val_losses = []
        val_accuracies = []

        for epoch in tqdm(range(epochs)):  # loop over the dataset multiple times
            # # call train step
            # train_step(
            #     model, train_data, loss, optimizer, writer, epoch, device, accuracy
            # )
            #
            # # call val step
            # val_step(model, val_data, loss, scheduler, writer, epoch, device, accuracy)
            #
            # # clear the GPU cache
            # torch.cuda.empty_cache()
            for batch in train_data:
                print(batch)
                break


            train_loss, train_accuracy = train_step(
                model, train_data, loss, optimizer, writer, epoch, device, accuracy
            )
            train_losses.append(train_loss)
            train_accuracies.append(train_accuracy)

            val_loss, val_accuracy = val_step(
                model, val_data, loss, scheduler, writer, epoch, device, accuracy
            )
            val_losses.append(val_loss)
            val_accuracies.append(val_accuracy)

    except KeyboardInterrupt:
        pass
    # save model
    save_model(model, f"./models/{name}.pth")

    return None
def plot_metrics(train_losses, train_accuracies, val_losses, val_accuracies):
    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(train_accuracies, label='Train Accuracy')
    plt.plot(val_accuracies, label='Validation Accuracy')
    plt.title('Training and Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()

    plt.show()

if __name__ == "__main__":
    main()
