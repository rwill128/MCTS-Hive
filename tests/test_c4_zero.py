import os
import tempfile
import unittest

import examples.c4_zero as c4_zero

if not c4_zero.HAS_TORCH:  # pragma: no cover - skip if torch missing
    raise unittest.SkipTest("PyTorch not available")

import torch

from examples.c4_zero import (
    C4ZeroNet,
    ZeroC4Player,
    play_one_game,
    train_step,
    evaluate_loss,
    save_dataset,
    load_dataset,
    save_weights,
    load_weights,
)
from simple_games.connect_four import ConnectFour


class TestC4ZeroTraining(unittest.TestCase):
    def test_training_and_persistence(self):
        net = C4ZeroNet()
        data = []
        for _ in range(2):
            data.extend(play_one_game(net))
        loss_before = evaluate_loss(net, data)
        opt = torch.optim.SGD(net.parameters(), lr=0.01)
        for _ in range(5):
            train_step(net, data, opt)
        loss_after = evaluate_loss(net, data)
        self.assertLess(loss_after, loss_before)

        with tempfile.TemporaryDirectory() as tmp:
            dpath = os.path.join(tmp, "data.pth")
            wpath = os.path.join(tmp, "weights.pth")
            save_dataset(data, dpath)
            save_weights(net, wpath)
            data2 = load_dataset(dpath)
            net2 = C4ZeroNet()
            load_weights(net2, wpath)
            self.assertEqual(len(data), len(data2))
            game = ConnectFour()
            state = game.getInitialState()
            player = ZeroC4Player(net2)
            action = player.search(state)
            self.assertIn(action, game.getLegalActions(state))


if __name__ == "__main__":
    unittest.main()
