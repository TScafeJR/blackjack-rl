from __future__ import annotations

import json
from typing import List, Self

from .matrix import zeros_like


class Parameter:
    def __init__(self, value: List[List[float]]):
        self.value = value
        self.grad = zeros_like(value)

    def zero_grad(self) -> None:
        self.grad = zeros_like(self.value)


class Module:
    def __init__(self):
        self.training = True

    def forward(self, inputs: List[List[float]]) -> List[List[float]]:
        raise Exception("Forward is not implemented")

    def backward(self, output_grad: List[List[float]]) -> List[List[float]]:
        raise Exception("Backward is not implemented")

    def get_parameters(self) -> List[Parameter]:
        return []

    def set_training(self, training: bool) -> Self:
        self.training = training
        return self

    def count_parameters(self) -> int:
        total = 0
        for parameter in self.get_parameters():
            total += len(parameter.value) * len(parameter.value[0])
        return total

    def get_weights(self) -> List[List[List[float]]]:
        return [
            [list(row) for row in parameter.value]
            for parameter in self.get_parameters()
        ]

    def set_weights(self, weights: List[List[List[float]]]) -> Self:
        parameters = self.get_parameters()
        if len(parameters) != len(weights):
            raise Exception("Weight count does not match parameter count")
        for parameter, value in zip(parameters, weights):
            parameter.value = [list(row) for row in value]
        return self

    def save(self, path: str) -> None:
        with open(path, "w", encoding="utf-8") as weights_file:
            json.dump({"weights": self.get_weights()}, weights_file)

    def load(self, path: str) -> Self:
        with open(path, "r", encoding="utf-8") as weights_file:
            stored = json.load(weights_file)
        return self.set_weights(stored["weights"])


class Sequential(Module):
    def __init__(self, layers: List[Module]):
        super().__init__()
        self.layers = layers

    def forward(self, inputs: List[List[float]]) -> List[List[float]]:
        outputs = inputs
        for layer in self.layers:
            outputs = layer.forward(outputs)
        return outputs

    def backward(self, output_grad: List[List[float]]) -> List[List[float]]:
        grad = output_grad
        for layer in reversed(self.layers):
            grad = layer.backward(grad)
        return grad

    def get_parameters(self) -> List[Parameter]:
        parameters = []
        for layer in self.layers:
            parameters.extend(layer.get_parameters())
        return parameters

    def set_training(self, training: bool) -> Self:
        super().set_training(training)
        for layer in self.layers:
            layer.set_training(training)
        return self
