import os
import random
import tempfile
import unittest

from .layers import Linear, Sigmoid
from .losses import MSELoss
from .module import Sequential
from .optimizers import Adam
from .trainer import Trainer


def build_xor_network(seed):
    rng = random.Random(seed)
    return Sequential(
        [
            Linear(in_features=2, out_features=8, rng=rng),
            Sigmoid(),
            Linear(in_features=8, out_features=1, rng=rng),
        ]
    )


XOR_INPUTS = [[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]]
XOR_TARGETS = [[0.0], [1.0], [1.0], [0.0]]


class TestTrainer(unittest.TestCase):
    def test_fit_learns_xor(self):
        network = build_xor_network(3)
        trainer = Trainer(
            model=network,
            loss=MSELoss(),
            optimizer=Adam(network.get_parameters(), learning_rate=0.05),
            epochs=500,
            batch_size=4,
            seed=3,
        )

        history = trainer.fit(XOR_INPUTS, XOR_TARGETS)

        self.assertEqual(len(history["train_loss"]), 500)
        self.assertLess(history["train_loss"][-1], 0.05)
        self.assertLess(history["train_loss"][-1], history["train_loss"][0])

    def test_fit_records_validation_history(self):
        network = build_xor_network(3)
        trainer = Trainer(
            model=network,
            loss=MSELoss(),
            optimizer=Adam(network.get_parameters(), learning_rate=0.01),
            epochs=5,
            batch_size=2,
            validation_split=0.25,
            seed=3,
        )

        history = trainer.fit(XOR_INPUTS, XOR_TARGETS)

        self.assertEqual(len(history["train_loss"]), 5)
        self.assertEqual(len(history["validation_loss"]), 5)

    def test_save_and_load_round_trip(self):
        network = build_xor_network(7)
        clone = build_xor_network(8)
        inputs = [[0.25, 0.75]]

        with tempfile.TemporaryDirectory() as temp_dir:
            weights_path = os.path.join(temp_dir, "weights.json")
            network.save(weights_path)
            clone.load(weights_path)

        self.assertEqual(network.forward(inputs), clone.forward(inputs))

    def test_count_parameters_default_agent_network(self):
        rng = random.Random(1)
        network = Sequential(
            [
                Linear(in_features=4, out_features=32, rng=rng),
                Sigmoid(),
                Linear(in_features=32, out_features=32, rng=rng),
                Sigmoid(),
                Linear(in_features=32, out_features=3, rng=rng),
            ]
        )
        self.assertEqual(network.count_parameters(), 1315)


if __name__ == "__main__":
    unittest.main()
