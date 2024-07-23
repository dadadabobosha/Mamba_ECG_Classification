import torch
from src.binary_classification.data2017 import load_ekg_data2017
from src.utils.torchutils import set_seed, load_model
from src.utils.metrics import Accuracy, BinaryAccuracy
from sklearn.metrics import f1_score

r"""

(venv) PS C:\wenjian\MasterArbeit\REPOS\Mamba-Biometric-EKG-Analysis-Technology-MambaBEAT-master> python -m src.evaluate2017

"""

# static variables
DATA_PATH: str = "C:\\wenjian\\MasterArbeit\\Code\\repo\\ECG+SSM\\Mamba-Biometric-EKG-Analysis-Technology-MambaBEAT-master\\src\\binary_classification\\data\\training2017\\"
NUM_CLASSES: int = 1

# set device
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
set_seed(42)


def main(name, accuracy: Accuracy = Accuracy()) -> None:
    """
    This function is the main program.
    """

    # Load test data
    _, _, test_data = load_ekg_data2017(DATA_PATH, num_workers=4)

    # Load mode
    model = load_model(name).to(device).float()  # 确保模型参数为 float32

    all_targets = []
    all_predictions = []
    with torch.no_grad():
        # for inputs, _, targets in test_data:
        for inputs, targets in test_data:
            inputs, targets = inputs.to(device), targets.to(device).float()  # 确保输入为 float32

            # forward
            outputs = model(inputs)

            # Compute accuracy
            accuracy.update(outputs, targets)

            # 将当前批次的目标和预测添加到列表中
            all_targets.append(targets.cpu())
            all_predictions.append(outputs.argmax(1).cpu())
        # show the accuracy
        print(f"Accuracy: {accuracy.compute()}")
        accuracy.reset()

        # 在循环结束后，合并所有批次的目标和预测
        all_targets = torch.cat(all_targets).numpy()
        all_predictions = torch.cat(all_predictions).numpy()
        # Compute F1 score
        f1 = f1_score(all_targets, all_predictions, average='weighted')
        print(f"F1 Score: {f1}")


if __name__ == "__main__":
    # path = "./report_models/model_name.pth"
    # path = "C:\\wenjian\\MasterArbeit\\Code\\repo\\ECG+SSM\\Mamba-Biometric-EKG-Analysis-Technology-MambaBEAT-master\\src\\benchmarks\\binary_MambaBEAT.pth"
    # path = "C:\\wenjian\\MasterArbeit\\Code\\repo\\ECG+SSM\\Mamba-Biometric-EKG-Analysis-Technology-MambaBEAT-master\\models\\binary_MambaBEAT.pth"
    path = "C:\\wenjian\\MasterArbeit\\Code\\repo\\ECG+SSM\\Mamba-Biometric-EKG-Analysis-Technology-MambaBEAT-master\\models\\binary_MambaBEAT+data2017_30epoch_NA.pth"


    # change to BinaryAccuracy() if model has a binary approach (starts with binary)
    # accuracy = Accuracy()
    accuracy = BinaryAccuracy()
    main(path, accuracy)
