import unittest

from .module import Parameter
from .optimizers import SGD, Adam, RMSprop


def build_parameter():
    parameter = Parameter([[1.0]])
    parameter.grad = [[0.5]]
    return parameter


class TestSGD(unittest.TestCase):
    def test_step(self):
        parameter = build_parameter()
        SGD([parameter], learning_rate=0.1).step()
        self.assertAlmostEqual(parameter.value[0][0], 0.95, places=10)

    def test_zero_grad(self):
        parameter = build_parameter()
        SGD([parameter]).zero_grad()
        self.assertEqual(parameter.grad, [[0.0]])


class TestRMSprop(unittest.TestCase):
    def test_step(self):
        parameter = build_parameter()
        RMSprop([parameter], learning_rate=0.001, decay=0.9).step()
        self.assertAlmostEqual(parameter.value[0][0], 0.9968377, places=6)


class TestAdam(unittest.TestCase):
    def test_step(self):
        parameter = build_parameter()
        Adam([parameter], learning_rate=0.001).step()
        self.assertAlmostEqual(parameter.value[0][0], 0.999, places=6)

    def test_two_steps_use_bias_correction(self):
        parameter = build_parameter()
        optimizer = Adam([parameter], learning_rate=0.001)
        optimizer.step()
        parameter.grad = [[0.5]]
        optimizer.step()
        self.assertAlmostEqual(parameter.value[0][0], 0.998, places=6)


if __name__ == "__main__":
    unittest.main()
