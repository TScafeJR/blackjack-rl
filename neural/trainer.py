import random
from typing import Dict, List


class Trainer:
    def __init__(self, **kwargs):
        self.model = kwargs.get("model")
        self.loss = kwargs.get("loss")
        self.optimizer = kwargs.get("optimizer")
        self.epochs = kwargs.get("epochs", 10)
        self.batch_size = kwargs.get("batch_size", 32)
        self.validation_split = kwargs.get("validation_split", 0.0)
        self.shuffle = kwargs.get("shuffle", True)
        self.rng = random.Random(kwargs.get("seed"))

    def split_data(self, inputs: List, targets: List):
        indices = list(range(len(inputs)))
        if self.shuffle:
            self.rng.shuffle(indices)
        validation_count = int(len(indices) * self.validation_split)
        train_indices = indices[: len(indices) - validation_count]
        validation_indices = indices[len(indices) - validation_count :]
        train_data = (
            [inputs[i] for i in train_indices],
            [targets[i] for i in train_indices],
        )
        validation_data = (
            [inputs[i] for i in validation_indices],
            [targets[i] for i in validation_indices],
        )
        return train_data, validation_data

    def run_epoch(self, train_inputs: List, train_targets: List) -> float:
        indices = list(range(len(train_inputs)))
        if self.shuffle:
            self.rng.shuffle(indices)
        total_loss = 0.0
        batch_count = 0
        for start in range(0, len(indices), self.batch_size):
            batch_indices = indices[start : start + self.batch_size]
            batch_inputs = [train_inputs[i] for i in batch_indices]
            batch_targets = [train_targets[i] for i in batch_indices]
            predictions = self.model.forward(batch_inputs)
            total_loss += self.loss.forward(predictions, batch_targets)
            batch_count += 1
            self.optimizer.zero_grad()
            self.model.backward(self.loss.backward())
            self.optimizer.step()
        return total_loss / batch_count

    def evaluate(self, inputs: List, targets: List) -> float:
        self.model.set_training(False)
        predictions = self.model.forward(inputs)
        validation_loss = self.loss.forward(predictions, targets)
        self.model.set_training(True)
        return validation_loss

    def fit(self, inputs: List, targets: List) -> Dict[str, List[float]]:
        (train_inputs, train_targets), (validation_inputs, validation_targets) = (
            self.split_data(inputs, targets)
        )
        history: Dict[str, List[float]] = {"train_loss": [], "validation_loss": []}
        for _ in range(self.epochs):
            history["train_loss"].append(self.run_epoch(train_inputs, train_targets))
            if validation_inputs:
                history["validation_loss"].append(
                    self.evaluate(validation_inputs, validation_targets)
                )
        return history
