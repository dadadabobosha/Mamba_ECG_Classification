# deep learning libraries
import torch
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader


# other libraries
from tqdm.auto import tqdm

# own modules
from src.utils.train_functions import train_step, val_step

from src.binary_classification.data2017_3_balancedata import (
    get_dataloaders,
)

from src.utils.metrics import BinaryAccuracy

from src.utils.torchutils import set_seed, save_model
from src.modules.cnn1 import YModel
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
    epochs: int = 200
    lr: float = 0.05
    batch_size: int = 128
    step_size: int = 10
    gamma: float = 0.5

    hidden_sizes = (256, 128, 64)
    # empty nohup file
    open("nohup.out", "w").close()

    # load data
    train_data: DataLoader
    val_data: DataLoader
    # train_data, val_data, _ = get_dataloaders(batch_size=batch_size)
    train_data, val_data, test_dataset = get_dataloaders(batch_size=batch_size)

    # save test_dataset,because I am afraid if I run the get_dataloaders again, the test_dataset will change
    torch.save(test_dataset, 'C:\\wenjian\\MasterArbeit\\Code\\repo\\ECG+SSM\\Mamba-Biometric-EKG-Analysis-Technology-MambaBEAT-master\\src\\test_dataset.pth')

    # define name and writer
    name: str = "binary_cnn_2017_5_200epoch"
    writer: SummaryWriter = SummaryWriter(f"runs/{name}")
    inputs: torch.Tensor = next(iter(train_data))[0]
    inputs = inputs.unsqueeze(1)  # add channel dimension from [batch_size, seq_len] to [batch_size, channels, seq_len]

    # define model
    model: torch.nn.Module = (
        YModel(
            input_channels=inputs.shape[1],
            hidden_sizes=hidden_sizes,
            output_channels=N_CLASSES,
        )
        .to(device)
        # .double() #my data is float32, so I should use float32
    )

    # define loss and optimizer
    loss: torch.nn.Module = torch.nn.CrossEntropyLoss()
    optimizer: torch.optim.Optimizer = torch.optim.AdamW(
        model.parameters(), lr=lr, weight_decay=0.05
    )

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

            # these commented codes are the ones can't do the plot
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

        # plot the loss and accuracy
    plot_metrics(train_losses, train_accuracies, val_losses, val_accuracies)



    save_model(model, f"./models/{name}_whole_model.pth")
    torch.save(model.state_dict(), f"./models/{name}_weight.pth")
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