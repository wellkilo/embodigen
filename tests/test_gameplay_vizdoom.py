import importlib.util
import tempfile
import unittest
from pathlib import Path

from embodigen.gameplay.pipeline import curate, run_rollouts
from embodigen.gameplay.recording import read_recording
from embodigen.gameplay.registry import WorldRegistry

HAS_VIZDOOM = importlib.util.find_spec("vizdoom") is not None


@unittest.skipUnless(HAS_VIZDOOM, "needs the [env] extra (vizdoom)")
class VizDoomTest(unittest.TestCase):
    def test_rollout_records_engine_truth_and_passes_the_gate(self):
        with tempfile.TemporaryDirectory() as tmp:
            raw = Path(tmp) / "raw"
            run_rollouts(WorldRegistry.load(), ["doom"], 1, 12.0, raw, seed=3, size=(160, 120))
            rec_dir = raw / "doom-0003"
            rec = read_recording(rec_dir)
            self.assertEqual(rec.meta.frame_hz, 35.0)
            self.assertTrue(rec.meta.exact_state)
            on_grid = [o for o in rec.observations if o.state is not None]
            self.assertTrue(on_grid)
            self.assertTrue((rec_dir / on_grid[0].depth).exists())
            self.assertTrue((rec_dir / on_grid[0].labels).exists())
            self.assertIn("visible", on_grid[0].state)
            ledger = curate(raw, Path(tmp) / "curated")
            self.assertEqual(ledger["summary"]["certified"], 1)


if __name__ == "__main__":
    unittest.main()
