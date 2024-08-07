import sys
# sys.path.insert(0, 'C:\\wenjian\\MasterArbeit\\REPOS\Mamba-Biometric-EKG-Analysis-Technology-MambaBEAT-master\\src')
sys.path.insert(0, r'C:\wenjian\MasterArbeit\REPOS\Mamba-Biometric-EKG-Analysis-Technology-MambaBEAT-master\src')
# deep learning libraries
import torch
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader


# other libraries
from tqdm.auto import tqdm

# own modules
# from src.utils.train_functions import train_step, val_step
from ..utils.train_functions import train_step, val_step

'''
In your case, you're trying to run train_mamba.py as a script, and it's located inside the train_binary package, which is inside the src package. The relative import from ..utils.train_functions import train_step, val_step is trying to import from the utils package, which is a sibling package to train_binary.  To fix this issue, you can run your script as a module from the root directory of your project. Here's how you can do it:  
Navigate to the root directory of your project in your terminal. In your case, it's C:\wenjian\MasterArbeit\REPOS\Mamba-Biometric-EKG-Analysis-Technology-MambaBEAT-master.  
Run your script as a module using the -m flag. Make sure to replace the slashes in the path with dots and remove the .py extension:      

(venv) PS C:\wenjian\MasterArbeit\REPOS\Mamba-Biometric-EKG-Analysis-Technology-MambaBEAT-master> python -m src.train_binary.train_mamba
'''
from src.binary_classification.data import (
    load_ekg_data,
)
from src.utils.metrics import BinaryAccuracy

from src.utils.torchutils import set_seed, save_model

from src.modules.mamba import MambaBEAT


# set device
device: torch.device = (
    torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
)

# set all seeds and set number of threads
set_seed(42)
torch.set_num_threads(8)

# static variables
DATA_PATH: str = "./data/"
N_CLASSES: int = 5


def main() -> None:
    """
    This function is the main program for the training.
    """
    # hyperparameters
    # epochs: int = 60
    # lr: float = 1e-2
    # batch_size: int = 128
    # step_size: int = 25
    # gamma: float = 0.8
    #
    # # Mamba Hyperparameters
    # n_layers: int = 1
    # latent_state_dim: int = 12
    # expand: int = 2
    # dt_rank: int = None
    # kernel_size: int = 12
    # conv_bias: bool = True
    # bias: bool = False
    # method: str = "zoh"
    # dropout: float = 0.2

    # 下面是我为了快速地测试改的一批超参数
    epochs: int = 1
    lr: float = 0.1
    batch_size: int = 256
    step_size: int = 25
    gamma: float = 0.8

    # Mamba Hyperparameters
    n_layers: int = 1
    latent_state_dim: int = 12
    expand: int = 2
    dt_rank: int = None
    kernel_size: int = 12
    conv_bias: bool = True
    bias: bool = False
    method: str = "zoh"
    dropout: float = 0.0


    # empty nohup file
    open("nohup.out", "w").close()

    # load data
    train_data: DataLoader
    val_data: DataLoader
    train_data, val_data, _ = load_ekg_data(DATA_PATH, batch_size=batch_size)

    # define name and writer
    name: str = "binary_MambaBEAT"  # personalizable name
    writer: SummaryWriter = SummaryWriter(f"runs/{name}")
    inputs: torch.Tensor = next(iter(train_data))[0]

    # define model
    model: torch.nn.Module = (
        MambaBEAT(
            inputs.size(2),
            N_CLASSES,
            n_layers,
            latent_state_dim,
            expand,
            dt_rank,
            kernel_size,
            conv_bias,
            bias,
            method,
            dropout,
        )
        .to(device)
        .double()
    )

    # define loss and optimizer
    loss: torch.nn.Module = torch.nn.BCELoss()
    optimizer: torch.optim.Optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    accuracy: BinaryAccuracy = BinaryAccuracy()

    # define an empty scheduler
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer, step_size=step_size, gamma=gamma
    )
    # Train the model
    try:
        for epoch in tqdm(range(epochs)):  # loop over the dataset multiple times
            # call train step
            train_step(
                model, train_data, loss, optimizer, writer, epoch, device, accuracy
            )

            # call val step
            val_step(model, val_data, loss, scheduler, writer, epoch, device, accuracy)

            # clear the GPU cache
            torch.cuda.empty_cache()

    except KeyboardInterrupt:
        pass

    save_model(model, f"./models/{name}.pth")
    return None


if __name__ == "__main__":
    main()
