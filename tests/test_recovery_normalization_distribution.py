"""CPU-only normalizer distribution audit regression coverage.

Core shape/percentile tests run when NumPy is installed; GitHub stdlib CI
may skip them rather than downloading scientific wheels.
"""
import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"scripts"))
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"src"))
from audit_recovery_normalization_distribution import (
    dimension_report, DistributionAuditError, main,
)


class ReadOnlyCLIGates(unittest.TestCase):
    def test_duplicate_episode_rejected_before_reading_data(self):
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(SystemExit) as ctx:
                main(["--dataset-root",tmp,"--manifest","/missing",
                      "--model-dir","/missing","--vlm-runtime-dir","/missing",
                      "--episodes","0,0","--out",str(Path(tmp)/"new.json")])
            self.assertEqual(ctx.exception.code,2)

    def test_existing_report_refused_before_reading_data(self):
        with tempfile.TemporaryDirectory() as tmp:
            old=Path(tmp)/"old.json"
            old.write_text('{"sentinel":true}')
            with self.assertRaises(SystemExit) as ctx:
                main(["--dataset-root",tmp,"--manifest","/missing",
                      "--model-dir","/missing","--vlm-runtime-dir","/missing",
                      "--episodes","0","--out",str(old)])
            self.assertEqual(ctx.exception.code,2)
            self.assertEqual(json.loads(old.read_text()),{"sentinel":True})


@unittest.skipUnless(importlib.util.find_spec("numpy"),"NumPy optional in stdlib CI")
class NumericDistributionTests(unittest.TestCase):
    def test_zero_shift_and_sigma_quantiles(self):
        import numpy as np
        data=np.array([[-1.0,1.0],[0.0,2.0],[1.0,3.0]],dtype=np.float32)
        r=dimension_report(data,[0.0,2.0],[1.0,1.0],np=np)
        self.assertEqual(r["num_frames"],3)
        self.assertAlmostEqual(r["dimensions"][0]["mean_z"],0)
        self.assertEqual(r["dimensions"][1]["pilot_mean"],2)
        self.assertFalse(r["any_p95_abs_z_gt_3"])

    def test_bad_dimensions_or_std_refused(self):
        import numpy as np
        with self.assertRaises(DistributionAuditError):
            dimension_report([[1,2]],[0],[1],np=np)
        with self.assertRaises(DistributionAuditError):
            dimension_report([[1,2]],[0,0],[0,1],np=np)
        with self.assertRaises(DistributionAuditError):
            dimension_report([[1,2]],[0,0],[float("nan"),1],np=np)


if __name__=="__main__":
    unittest.main()
