from typing import List

from .matrix import zeros_like
from .module import Parameter


class BaseOptimizer:
    def __init__(self, parameters: List[Parameter], **kwargs):
        self.parameters = parameters
        self.learning_rate = kwargs.get("learning_rate", 0.01)

    def zero_grad(self) -> None:
        for parameter in self.parameters:
            parameter.zero_grad()

    def step(self) -> None:
        raise Exception("Step is not implemented")


class SGD(BaseOptimizer):
    def step(self) -> None:
        for parameter in self.parameters:
            for row_index, row in enumerate(parameter.grad):
                for col_index, grad in enumerate(row):
                    parameter.value[row_index][col_index] -= self.learning_rate * grad


class RMSprop(BaseOptimizer):
    def __init__(self, parameters: List[Parameter], **kwargs):
        super().__init__(parameters, **kwargs)
        self.learning_rate = kwargs.get("learning_rate", 0.001)
        self.decay = kwargs.get("decay", 0.9)
        self.epsilon = kwargs.get("epsilon", 1e-8)
        self.cache = [zeros_like(parameter.value) for parameter in parameters]

    def step(self) -> None:
        for parameter, cache in zip(self.parameters, self.cache):
            for row_index, row in enumerate(parameter.grad):
                for col_index, grad in enumerate(row):
                    cache[row_index][col_index] = (
                        self.decay * cache[row_index][col_index]
                        + (1.0 - self.decay) * grad * grad
                    )
                    parameter.value[row_index][col_index] -= (
                        self.learning_rate
                        * grad
                        / (cache[row_index][col_index] ** 0.5 + self.epsilon)
                    )


class Adam(BaseOptimizer):
    def __init__(self, parameters: List[Parameter], **kwargs):
        super().__init__(parameters, **kwargs)
        self.learning_rate = kwargs.get("learning_rate", 0.001)
        self.beta1 = kwargs.get("beta1", 0.9)
        self.beta2 = kwargs.get("beta2", 0.999)
        self.epsilon = kwargs.get("epsilon", 1e-8)
        self.first_moments = [zeros_like(parameter.value) for parameter in parameters]
        self.second_moments = [zeros_like(parameter.value) for parameter in parameters]
        self.timestep = 0

    def step(self) -> None:
        self.timestep += 1
        bias_correction1 = 1.0 - self.beta1**self.timestep
        bias_correction2 = 1.0 - self.beta2**self.timestep
        for parameter, first_moment, second_moment in zip(
            self.parameters, self.first_moments, self.second_moments
        ):
            for row_index, row in enumerate(parameter.grad):
                for col_index, grad in enumerate(row):
                    first_moment[row_index][col_index] = (
                        self.beta1 * first_moment[row_index][col_index]
                        + (1.0 - self.beta1) * grad
                    )
                    second_moment[row_index][col_index] = (
                        self.beta2 * second_moment[row_index][col_index]
                        + (1.0 - self.beta2) * grad * grad
                    )
                    corrected_first = (
                        first_moment[row_index][col_index] / bias_correction1
                    )
                    corrected_second = (
                        second_moment[row_index][col_index] / bias_correction2
                    )
                    parameter.value[row_index][col_index] -= (
                        self.learning_rate
                        * corrected_first
                        / (corrected_second**0.5 + self.epsilon)
                    )
