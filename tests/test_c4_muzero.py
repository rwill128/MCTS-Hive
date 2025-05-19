import os
import tempfile
import unittest

import examples.c4_muzero as muz

if muz.torch is None or muz.np is None:
    raise unittest.SkipTest("PyTorch or NumPy not available")

class TestC4MuZero(unittest.TestCase):
    def test_run_small(self):
        with tempfile.TemporaryDirectory() as tmp:
            args = muz.parser().parse_args([])
            args.ckpt_dir = tmp
            args.games = 1
            args.epochs = 1
            args.batch = 2
            args.buffer = 5
            args.sims = 1
            muz.run(args)
            self.assertTrue(os.path.exists(os.path.join(tmp, "last.pt")))

if __name__ == "__main__":
    unittest.main()
