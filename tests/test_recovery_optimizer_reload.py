"""Stdlib-only safety checks for exact-one-step, no-overwrite optimizer smoke."""
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/"scripts"))
from smoke_recovery_optimizer_reload import check_output_paths, GPUSmokeError, main


class OneStepTest(unittest.TestCase):
    def test_no_output_overwrite_and_distinct_paths(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            adapter,report=check_output_paths(root/"adapter",root/"report.json")
            self.assertEqual(adapter,root/"adapter")
            self.assertFalse(adapter.exists())
            with self.assertRaises(GPUSmokeError):
                check_output_paths(adapter,adapter)
            (root/"adapter").mkdir()
            with self.assertRaisesRegex(GPUSmokeError,"Refuse to overwrite"):
                check_output_paths(adapter,root/"report.json")
            with self.assertRaisesRegex(GPUSmokeError,"Refuse to overwrite"):
                check_output_paths(root/"new",root/"adapter")

    def test_cli_refuses_without_explicit_approval(self):
        with self.assertRaises(SystemExit) as ctx:
            main(["--sample","/bad.npz",
                  "--sample-report","/bad.json",
                  "--model-dir","/model",
                  "--vlm-runtime-dir","/runtime",
                  "--adapter-dir","/adapter",
                  "--out","/report.json"])
        self.assertEqual(ctx.exception.code,2)


if __name__=="__main__":
    unittest.main()
