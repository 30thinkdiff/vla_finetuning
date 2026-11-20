"""Static CI checks for fast pinned sparse Recovery simulator bootstrap."""
from pathlib import Path
import subprocess
import unittest

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/bootstrap_recovery_sim_env.sh"

class SparseRecoveryBootstrapTests(unittest.TestCase):
    def test_bash_syntax_and_core_guards(self):
        result = subprocess.run(["bash", "-n", str(SCRIPT)], text=True, capture_output=True)
        self.assertEqual(result.returncode, 0, result.stderr)
        text = SCRIPT.read_text(encoding="utf-8")
        for marker in [
            "--filter=blob:none",
            "sparse-checkout init --cone",
            "sparse-checkout set",
            "LIBERO/libero/libero/assets/stable_hope_objects/ketchup",
            "LIBERO/libero/libero/assets/stable_scanned_objects/basket",
            "VLA_APPROVED_CPU_IO",
            'git -C "$SOURCE" rev-parse HEAD',
        ]:
            self.assertIn(marker,text)

if __name__=="__main__":
    unittest.main()
