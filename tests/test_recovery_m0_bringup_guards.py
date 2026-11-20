"""Pure stdlib validation of M0 simulator/postprocessor safety guards."""
import hashlib
import json
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"scripts"))
from fetch_smolvla_postprocessor_assets import (
    prepare, PostprocessorError, validate_safetensors, SHA, MODEL
)
from bringup_recovery_sim import main as sim_main


class PinnedPostprocessorTests(unittest.TestCase):
    def test_validates_original_model_receipt_and_prevents_overwrite(self):
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp)/SHA
            p.mkdir()
            files={}
            for name,body in (
                ("config.json",b'{"type":"smolvla"}'),
                ("policy_preprocessor.json",b'{"steps":[]}'),
            ):
                (p/name).write_bytes(body)
                files[name]={"sha256":hashlib.sha256(body).hexdigest()}
            (p/"receipt.json").write_text(json.dumps({
                "model_repo":MODEL,"revision":SHA,"files":files}))
            self.assertEqual(prepare(p,"https://hf-mirror.com"),p)
            (p/"policy_postprocessor.json").write_text("{}")
            with self.assertRaisesRegex(PostprocessorError,"overwrite"):
                prepare(p,"https://hf-mirror.com")

    def test_valid_safetensors_header(self):
        hdr=json.dumps({"weight":{"dtype":"F32","shape":[1],
                                  "data_offsets":[0,4]}}).encode()
        data=len(hdr).to_bytes(8,"little")+hdr+b"\0"*4
        self.assertIn("weight",validate_safetensors(data))
        with self.assertRaises(PostprocessorError):
            validate_safetensors(b"garbage")


class SimulatorCLIApprovalTest(unittest.TestCase):
    def test_requires_explicit_run(self):
        with self.assertRaises(SystemExit) as ctx:
            sim_main(["--manifest","/missing.json",
                      "--scene-root","/missing",
                      "--recovery-source","/missing",
                      "--out","/missing_report.json"])
        self.assertEqual(ctx.exception.code,2)


if __name__=="__main__":
    unittest.main()
