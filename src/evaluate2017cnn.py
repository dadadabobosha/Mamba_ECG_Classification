import torch
from src.binary_classification.data2017_3_balancedata import (
    get_dataloaders,
)
from src.utils.torchutils import set_seed, load_model
from src.utils.metrics import Accuracy, BinaryAccuracy

from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import numpy as np
import itertools

r"""

(venv) PS C:\wenjian\MasterArbeit\REPOS\Mamba-Biometric-EKG-Analysis-Technology-MambaBEAT-master> python -m src.evaluate

"""

# static variables
DATA_PATH: str = "../data/"
NUM_CLASSES: int = 2

# set device
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
set_seed(42)


def main(name, accuracy: Accuracy = Accuracy()) -> None:
    """
    This function is the main program.
    """

    # Load test data
    # _, _, test_data = get_dataloaders(batch_size=128)

    #here get_dataloaders was used, but now use torch.load to load the test_dataset to make sure the train data won't get into the test data

    test_data = torch.load('C:\\wenjian\\MasterArbeit\\Code\\repo\\ECG+SSM\\Mamba-Biometric-EKG-Analysis-Technology-MambaBEAT-master\\src\\test_dataset.pth')

    # Load model
    model = load_model(name).to(device)

    all_predictions = []
    all_targets = []

    with torch.no_grad():
        for inputs, targets in test_data:
            inputs = inputs.unsqueeze(1)
            inputs, targets = inputs.to(device), targets.to(device)
            print("inputs shape:", inputs.shape)
            print("targets shape:", targets.shape)

            # forward
            outputs = model(inputs)
            print("outputs shape:", outputs.shape)
            print("outputs:", outputs)
            print("targets:", targets)

            _, predicted_classes = torch.max(outputs, 1)
            _, true_classes = torch.max(targets, 1)
            all_predictions.extend(predicted_classes.cpu().numpy())
            all_targets.extend(true_classes.cpu().numpy())


            # Compute accuracy
            accuracy.update(outputs, targets)

        # show the accuracy
        print(f"Accuracy: {accuracy.compute()}")
        accuracy.reset()

    # plot confusion matrix
    plot_confusion_matrix(all_targets, all_predictions)

def plot_confusion_matrix(true_labels, pred_labels):
    cm = confusion_matrix(true_labels, pred_labels)
    plt.figure(figsize=(8, 8))
    plt.imshow(cm, interpolation='nearest', cmap=plt.get_cmap('Blues'))
    plt.title('Confusion Matrix')
    plt.colorbar()

    tick_marks = np.arange(NUM_CLASSES)
    plt.xticks(tick_marks, range(NUM_CLASSES))
    plt.yticks(tick_marks, range(NUM_CLASSES))

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, f"{cm[i, j]}", horizontalalignment="center",
                 color="red" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.title('Confusion Matrix')
    plt.show()

if __name__ == "__main__":
    # path = "./report_models/model_name.pth"

    #path of the model
    path = "C:\\wenjian\\MasterArbeit\\Code\\repo\\ECG+SSM\\Mamba-Biometric-EKG-Analysis-Technology-MambaBEAT-master\\src\\train_binary\\models\\binary_cnn_2017_5_200epoch_whole_model.pth"

    # change to BinaryAccuracy() if model has a binary approach (starts with binary)
    # accuracy = Accuracy()
    accuracy = BinaryAccuracy()
    main(path, accuracy)
