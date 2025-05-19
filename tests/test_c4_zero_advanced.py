import os
import tempfile
import unittest

import examples.c4_zero_advanced as adv

if adv.torch is None:  # pragma: no cover - skip if torch missing
    raise unittest.SkipTest("PyTorch not available")

from pathlib import Path


class TestC4ZeroAdvanced(unittest.TestCase):
    def test_run_and_resume(self):
        with tempfile.TemporaryDirectory() as tmp:
            adv.BUFFER_PATH = Path(tmp) / "buffer.pth"
            args = adv.parser().parse_args([])
            args.ckpt_dir = os.path.join(tmp, "ck")
            args.games = 1
            args.epochs = 1
            args.batch = 2
            args.buffer = 10
            args.ckpt_every = 1
            args.skip_bootstrap = False
            args.resume = None

            adv.run(args)

            self.assertTrue(adv.BUFFER_PATH.exists())
            self.assertTrue(os.path.exists(os.path.join(args.ckpt_dir, "last.pt")))

            # run again with auto-resume
            args.skip_bootstrap = True
            args.resume = None
            adv.run(args)


if __name__ == "__main__":
    unittest.main()

