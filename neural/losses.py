import math
from typing import List


class MSELoss:
    def __init__(self):
        self.predictions: List[List[float]] = []
        self.targets: List[List[float]] = []

    def forward(
        self, predictions: List[List[float]], targets: List[List[float]]
    ) -> float:
        self.predictions = predictions
        self.targets = targets
        total = 0.0
        count = 0
        for prediction_row, target_row in zip(predictions, targets):
            for prediction, target in zip(prediction_row, target_row):
                total += (prediction - target) ** 2
                count += 1
        return total / count

    def backward(self) -> List[List[float]]:
        count = len(self.predictions) * len(self.predictions[0])
        return [
            [
                2.0 * (prediction - target) / count
                for prediction, target in zip(prediction_row, target_row)
            ]
            for prediction_row, target_row in zip(self.predictions, self.targets)
        ]


class SoftmaxCrossEntropyLoss:
    def __init__(self):
        self.probabilities: List[List[float]] = []
        self.labels: List[int] = []

    @staticmethod
    def softmax(logits: List[float]) -> List[float]:
        highest = max(logits)
        exponentials = [math.exp(value - highest) for value in logits]
        total = sum(exponentials)
        return [value / total for value in exponentials]

    def forward(self, logits: List[List[float]], labels: List[int]) -> float:
        self.probabilities = [self.softmax(row) for row in logits]
        self.labels = labels
        total = 0.0
        for probabilities, label in zip(self.probabilities, labels):
            total -= math.log(max(probabilities[label], 1e-12))
        return total / len(labels)

    def backward(self) -> List[List[float]]:
        batch_size = len(self.probabilities)
        grads = []
        for probabilities, label in zip(self.probabilities, self.labels):
            row = [probability / batch_size for probability in probabilities]
            row[label] -= 1.0 / batch_size
            grads.append(row)
        return grads
