import unittest

from .losses import MSELoss, SoftmaxCrossEntropyLoss


class TestMSELoss(unittest.TestCase):
    def setUp(self):
        self.loss = MSELoss()

    def test_forward(self):
        value = self.loss.forward([[1.0, 2.0]], [[0.0, 0.0]])
        self.assertAlmostEqual(value, 2.5, places=10)

    def test_backward(self):
        self.loss.forward([[1.0, 2.0]], [[0.0, 0.0]])
        grads = self.loss.backward()
        self.assertAlmostEqual(grads[0][0], 1.0, places=10)
        self.assertAlmostEqual(grads[0][1], 2.0, places=10)


class TestSoftmaxCrossEntropyLoss(unittest.TestCase):
    def setUp(self):
        self.loss = SoftmaxCrossEntropyLoss()

    def test_softmax(self):
        probabilities = SoftmaxCrossEntropyLoss.softmax([2.0, 1.0, 0.1])
        self.assertAlmostEqual(sum(probabilities), 1.0, places=10)
        self.assertAlmostEqual(probabilities[0], 0.6590, places=3)
        self.assertAlmostEqual(probabilities[1], 0.2424, places=3)
        self.assertAlmostEqual(probabilities[2], 0.0986, places=3)

    def test_forward(self):
        value = self.loss.forward([[2.0, 1.0, 0.1]], [0])
        self.assertAlmostEqual(value, 0.4170, places=3)

    def test_backward(self):
        self.loss.forward([[2.0, 1.0, 0.1]], [0])
        grads = self.loss.backward()
        self.assertAlmostEqual(grads[0][0], -0.3410, places=3)
        self.assertAlmostEqual(grads[0][1], 0.2424, places=3)
        self.assertAlmostEqual(grads[0][2], 0.0986, places=3)

    def test_backward_sums_to_zero(self):
        self.loss.forward([[2.0, 1.0, 0.1], [0.5, 0.5, 3.0]], [0, 2])
        grads = self.loss.backward()
        for row in grads:
            self.assertAlmostEqual(sum(row), 0.0, places=10)


if __name__ == "__main__":
    unittest.main()
