import math
import random
from typing import List

from .matrix import add_row_vector, matmul, transpose
from .module import Module, Parameter


class Linear(Module):
    def __init__(self, **kwargs):
        super().__init__()
        in_features = kwargs.get("in_features", 1)
        out_features = kwargs.get("out_features", 1)
        rng = kwargs.get("rng", random.Random())
        limit = (6.0 / (in_features + out_features)) ** 0.5
        self.weights = Parameter(
            [
                [rng.uniform(-limit, limit) for _ in range(out_features)]
                for _ in range(in_features)
            ]
        )
        self.bias = Parameter([[0.0] * out_features])
        self.inputs: List[List[float]] = []

    def forward(self, inputs: List[List[float]]) -> List[List[float]]:
        self.inputs = inputs
        return add_row_vector(matmul(inputs, self.weights.value), self.bias.value[0])

    def backward(self, output_grad: List[List[float]]) -> List[List[float]]:
        weight_grad = matmul(transpose(self.inputs), output_grad)
        for row_index, row in enumerate(weight_grad):
            for col_index, value in enumerate(row):
                self.weights.grad[row_index][col_index] += value
        for col_index in range(len(self.bias.value[0])):
            self.bias.grad[0][col_index] += sum(row[col_index] for row in output_grad)
        return matmul(output_grad, transpose(self.weights.value))

    def get_parameters(self) -> List[Parameter]:
        return [self.weights, self.bias]


class ReLU(Module):
    def __init__(self):
        super().__init__()
        self.mask: List[List[bool]] = []

    def forward(self, inputs: List[List[float]]) -> List[List[float]]:
        self.mask = [[value > 0.0 for value in row] for row in inputs]
        return [[value if value > 0.0 else 0.0 for value in row] for row in inputs]

    def backward(self, output_grad: List[List[float]]) -> List[List[float]]:
        return [
            [value if keep else 0.0 for value, keep in zip(row, mask_row)]
            for row, mask_row in zip(output_grad, self.mask)
        ]


class LeakyReLU(Module):
    def __init__(self, **kwargs):
        super().__init__()
        self.slope = kwargs.get("slope", 0.01)
        self.mask: List[List[bool]] = []

    def forward(self, inputs: List[List[float]]) -> List[List[float]]:
        self.mask = [[value > 0.0 for value in row] for row in inputs]
        return [
            [value if value > 0.0 else value * self.slope for value in row]
            for row in inputs
        ]

    def backward(self, output_grad: List[List[float]]) -> List[List[float]]:
        return [
            [
                value if keep else value * self.slope
                for value, keep in zip(row, mask_row)
            ]
            for row, mask_row in zip(output_grad, self.mask)
        ]


class Sigmoid(Module):
    def __init__(self):
        super().__init__()
        self.outputs: List[List[float]] = []

    @staticmethod
    def activate(value: float) -> float:
        if value >= 0:
            return 1.0 / (1.0 + math.exp(-value))
        exp_value = math.exp(value)
        return exp_value / (1.0 + exp_value)

    def forward(self, inputs: List[List[float]]) -> List[List[float]]:
        self.outputs = [[self.activate(value) for value in row] for row in inputs]
        return self.outputs

    def backward(self, output_grad: List[List[float]]) -> List[List[float]]:
        return [
            [
                value * activated * (1.0 - activated)
                for value, activated in zip(row, output_row)
            ]
            for row, output_row in zip(output_grad, self.outputs)
        ]


class Dropout(Module):
    def __init__(self, **kwargs):
        super().__init__()
        self.probability = kwargs.get("probability", 0.5)
        self.rng = kwargs.get("rng", random.Random())
        self.mask: List[List[bool]] = []

    def forward(self, inputs: List[List[float]]) -> List[List[float]]:
        if not self.training or self.probability <= 0.0:
            self.mask = []
            return inputs
        keep_scale = 1.0 / (1.0 - self.probability)
        self.mask = [
            [self.rng.random() >= self.probability for _ in row] for row in inputs
        ]
        return [
            [value * keep_scale if keep else 0.0 for value, keep in zip(row, mask_row)]
            for row, mask_row in zip(inputs, self.mask)
        ]

    def backward(self, output_grad: List[List[float]]) -> List[List[float]]:
        if not self.mask:
            return output_grad
        keep_scale = 1.0 / (1.0 - self.probability)
        return [
            [value * keep_scale if keep else 0.0 for value, keep in zip(row, mask_row)]
            for row, mask_row in zip(output_grad, self.mask)
        ]
