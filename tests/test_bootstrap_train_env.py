"""Non-invasive bootstrap environment version gate tests: no installations."""
import os
import subprocess
import tempfile
import unittest
from pathlib import Path


SCRIPT=Path(__file__).resolve().parents[1]/"scripts"/"bootstrap_train_env.sh"


class BootstrapTrainVersionTests(unittest.TestCase):
    def test_apply_rejects_bad_interpreter_before_any_write(self):
        with tempfile.TemporaryDirectory() as tmp:
            fake=Path(tmp)/"bad-python"
            fake.write_text("#!/bin/sh\nexit 1\n")
            fake.chmod(0o755)
            env=os.environ.copy()
            env.update({"VLA_TRAIN_PYTHON":str(fake), "VLA_APPROVED_CPU_IO":"YES"})
            result=subprocess.run(
                ["bash",str(SCRIPT),"--apply"],
                env=env,capture_output=True,text=True,timeout=10,check=False)
            self.assertEqual(result.returncode,2)
            self.assertIn("requires Python >=3.12",result.stdout)
            self.assertNotIn("Dependencies prepared",result.stdout)

    def test_dry_run_never_installs_dependencies(self):
        env=os.environ.copy()
        env["VLA_TRAIN_PYTHON"]="/this/python-does-not-exist"
        result=subprocess.run(
            ["bash",str(SCRIPT)],env=env,capture_output=True,text=True,
            timeout=10,check=False)
        self.assertEqual(result.returncode,0)
        self.assertIn("DRY RUN",result.stdout)
        self.assertIn("Candidate Python executable missing",result.stdout)
        self.assertNotIn("Dependencies prepared",result.stdout)


if __name__=="__main__":
    unittest.main()
