# # deep learning libraries
# import torch
#
#
# class Accuracy:
#     """
#     This class is the accuracy object.
#
#     Attributes:
#         correct: number of correct predictions.
#         total: number of total examples to classify.
#     """
#
#     correct: int
#     total: int
#
#     def __init__(self) -> None:
#         """
#         This is the constructor of Accuracy class. It should
#         initialize correct and total to zero.
#         """
#
#         self.correct = 0
#         self.total = 0
#
#     def update(self, logits: torch.Tensor, labels: torch.Tensor) -> None:
#         """
#         This method update the value of correct and total counts.
#
#         Args:
#             logits: outputs of the model.
#                 Dimensions: [batch, number of classes]
#             labels: labels of the examples. Dimensions: [batch].
#         """
#
#         # 确保 logits 是二维张量
#         assert logits.dim() == 2, f"Expected logits to be 2D, got {logits.dim()}D"
#
#         # compute predictions
#         predictions = logits.argmax(1).type_as(labels)
#
#         # update counts
#         self.correct += int(predictions.eq(labels).sum().item())
#         self.total += labels.shape[0]
#
#         return None
#
#     def compute(self) -> float:
#         """
#         This method returns the accuracy value.
#
#         Returns:
#             accuracy value.
#         """
#         #56-58后加的
#         if self.total == 0:
#             return 0.0
#         return self.correct / self.total
#
#         return self.correct / self.total
#
#     def reset(self) -> None:
#         """
#         This method resets to zero the count of correct and total number of
#         examples.
#         """
#
#         # init to zero the counts
#         self.correct = 0
#         self.total = 0
#
#         return None
#
#
# class BinaryAccuracy(Accuracy):
#     """
#     This class is the accuracy object.
#
#     Attributes:
#         correct: number of correct predictions.
#         total: number of total examples to classify.
#     """
#
#     correct: int
#     total: int
#
#     def __init__(self, threshold: float = 0.5) -> None:
#         """
#         This is the constructor of Accuracy class. It should
#         initialize correct and total to zero.
#         """
#
#         super().__init__()
#         self.treshold = threshold
#
#     def update(self, logits: torch.Tensor, labels: torch.Tensor) -> None:
#         """
#         This method update the value of correct and total counts.
#
#         Args:
#             logits: outputs of the model.
#                 Dimensions: [batch, number of classes]
#             labels: labels of the examples. Dimensions: [batch].
#         """
#
#         # 打印 logits 的维度
#         print(f"logits.shape: {logits.shape}")
#
#         # compute predictions（给PTB-XL）
#         predictions = torch.where(logits > self.treshold, 1, 0)
#         # #compute predictions by taking the argmax（给2017用的）
#         # predictions = torch.argmax(logits, dim=1)
#
#         # update counts
#         self.correct += int(predictions.eq(labels).sum().item())
#         self.total += labels.nelement()
#
#         return None



r"""
下面是只保留N和A两类的代码，然后将数据预处理后保存为.npy文件，以便后续快速加载
然后在N和A两类中分类出A类的数据，将其标签改为1，N类的数据标签改为0
"""

# deep learning libraries
import torch

class Accuracy:
    """
    This class is the accuracy object.

    Attributes:
        correct: number of correct predictions.
        total: number of total examples to classify.
    """

    correct: int
    total: int

    def __init__(self) -> None:
        self.correct = 0
        self.total = 0

    def update(self, logits: torch.Tensor, labels: torch.Tensor) -> None:
        assert logits.dim() == 2, f"Expected logits to be 2D, got {logits.dim()}D"
        predictions = logits.argmax(1).type_as(labels)
        self.correct += int(predictions.eq(labels).sum().item())
        self.total += labels.shape[0]

    def compute(self) -> float:
        if self.total == 0:
            return 0.0
        return self.correct / self.total

    def reset(self) -> None:
        self.correct = 0
        self.total = 0

class BinaryAccuracy(Accuracy):
    def __init__(self, threshold: float = 0.5) -> None:
        super().__init__()
        self.threshold = threshold

    def update(self, logits: torch.Tensor, labels: torch.Tensor) -> None:
        predictions = (torch.sigmoid(logits) > self.threshold).long()
        self.correct += int(predictions.eq(labels).sum().item())
        self.total += labels.nelement()
