import random
import unittest

from .layers import Dropout, LeakyReLU, Linear, ReLU, Sigmoid
from .losses import MSELoss
from .module import Sequential
from .optimizers import SGD


def build_worked_example_network():
    network = Sequential(
        [
            Linear(in_features=2, out_features=2),
            ReLU(),
            Linear(in_features=2, out_features=1),
        ]
    )
    network.set_weights(
        [
            [[0.11, 0.12], [0.21, 0.08]],
            [[-10.0, -20.0]],
            [[0.14], [0.15]],
            [[0.0]],
        ]
    )
    return network


class TestWorkedExample(unittest.TestCase):
    def setUp(self):
        # Hand-worked 2-2-1 example from the coursework notes
        self.network = build_worked_example_network()

    def test_forward(self):
        outputs = self.network.forward([[100.0, 50.0]])

        hidden_outputs = self.network.layers[1].forward(
            self.network.layers[0].forward([[100.0, 50.0]])
        )
        self.assertEqual(hidden_outputs, [[11.5, 0.0]])
        self.assertAlmostEqual(outputs[0][0], 1.61, places=10)

    def test_backward_blame_assignment(self):
        self.network.forward([[100.0, 50.0]])
        self.network.backward([[0.61]])

        output_layer = self.network.layers[2]
        self.assertAlmostEqual(output_layer.weights.grad[0][0], 7.015, places=10)
        self.assertAlmostEqual(output_layer.weights.grad[1][0], 0.0, places=10)
        self.assertAlmostEqual(output_layer.bias.grad[0][0], 0.61, places=10)

        hidden_layer = self.network.layers[0]
        self.assertAlmostEqual(hidden_layer.weights.grad[0][0], 8.54, places=10)
        self.assertAlmostEqual(hidden_layer.weights.grad[1][0], 4.27, places=10)
        # The dead ReLU unit receives zero blame and zero updates
        self.assertAlmostEqual(hidden_layer.weights.grad[0][1], 0.0, places=10)
        self.assertAlmostEqual(hidden_layer.weights.grad[1][1], 0.0, places=10)
        self.assertAlmostEqual(hidden_layer.bias.grad[0][0], 0.0854, places=10)
        self.assertAlmostEqual(hidden_layer.bias.grad[0][1], 0.0, places=10)

    def test_sgd_update_low_learning_rate(self):
        self.network.forward([[100.0, 50.0]])
        self.network.backward([[0.61]])
        SGD(self.network.get_parameters(), learning_rate=0.001).step()

        output_layer = self.network.layers[2]
        hidden_layer = self.network.layers[0]
        self.assertAlmostEqual(output_layer.weights.value[0][0], 0.132985, places=10)
        self.assertAlmostEqual(output_layer.weights.value[1][0], 0.15, places=10)
        self.assertAlmostEqual(hidden_layer.weights.value[0][0], 0.10146, places=10)

    def test_sgd_update_high_learning_rate(self):
        self.network.forward([[100.0, 50.0]])
        self.network.backward([[0.61]])
        SGD(self.network.get_parameters(), learning_rate=0.01).step()

        output_layer = self.network.layers[2]
        self.assertAlmostEqual(output_layer.weights.value[0][0], 0.06985, places=10)

    def test_count_parameters(self):
        self.assertEqual(self.network.count_parameters(), 9)


class TestActivations(unittest.TestCase):
    def test_relu(self):
        layer = ReLU()
        self.assertEqual(layer.forward([[-1.0, 0.0, 2.0]]), [[0.0, 0.0, 2.0]])
        self.assertEqual(layer.backward([[5.0, 5.0, 5.0]]), [[0.0, 0.0, 5.0]])

    def test_leaky_relu(self):
        layer = LeakyReLU(slope=0.1)
        self.assertEqual(layer.forward([[-1.0, 2.0]]), [[-0.1, 2.0]])
        self.assertEqual(layer.backward([[5.0, 5.0]]), [[0.5, 5.0]])

    def test_sigmoid(self):
        layer = Sigmoid()
        outputs = layer.forward([[0.0]])
        self.assertAlmostEqual(outputs[0][0], 0.5, places=10)

        grads = layer.backward([[1.0]])
        self.assertAlmostEqual(grads[0][0], 0.25, places=10)

    def test_dropout_training(self):
        layer = Dropout(probability=0.5, rng=random.Random(5))
        outputs = layer.forward([[1.0] * 100])

        dropped = [value for value in outputs[0] if value == 0.0]
        kept = [value for value in outputs[0] if value != 0.0]
        self.assertGreater(len(dropped), 0)
        self.assertGreater(len(kept), 0)
        for value in kept:
            self.assertAlmostEqual(value, 2.0, places=10)

    def test_dropout_eval_identity(self):
        layer = Dropout(probability=0.5, rng=random.Random(5))
        layer.set_training(False)
        inputs = [[1.0, 2.0, 3.0]]
        self.assertEqual(layer.forward(inputs), inputs)
        self.assertEqual(layer.backward(inputs), inputs)


def numeric_gradient(network, loss, batch, entry):
    inputs, targets = batch
    parameter, row_index, col_index = entry
    epsilon = 1e-5
    original = parameter.value[row_index][col_index]

    parameter.value[row_index][col_index] = original + epsilon
    loss_high = loss.forward(network.forward(inputs), targets)
    parameter.value[row_index][col_index] = original - epsilon
    loss_low = loss.forward(network.forward(inputs), targets)
    parameter.value[row_index][col_index] = original

    return (loss_high - loss_low) / (2 * epsilon)


class TestGradientCheck(unittest.TestCase):
    def test_finite_difference(self):
        rng = random.Random(11)
        network = Sequential(
            [
                Linear(in_features=3, out_features=4, rng=rng),
                Sigmoid(),
                Linear(in_features=4, out_features=2, rng=rng),
            ]
        )
        loss = MSELoss()
        inputs = [[rng.uniform(-1, 1) for _ in range(3)] for _ in range(5)]
        targets = [[rng.uniform(-1, 1) for _ in range(2)] for _ in range(5)]

        loss.forward(network.forward(inputs), targets)
        network.backward(loss.backward())

        for parameter in network.get_parameters():
            for row_index, row in enumerate(parameter.value):
                for col_index in range(len(row)):
                    entry = (parameter, row_index, col_index)
                    numeric_grad = numeric_gradient(
                        network, loss, (inputs, targets), entry
                    )
                    analytic_grad = parameter.grad[row_index][col_index]
                    self.assertAlmostEqual(numeric_grad, analytic_grad, places=6)


if __name__ == "__main__":
    unittest.main()
