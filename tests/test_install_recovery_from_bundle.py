"""Static guard on the user-facing bundle installer (never mutates on import)."""
from pathlib import Path
import subprocess
import unittest

SCRIPT=Path(__file__).resolve().parents[1]/"scripts/install_recovery_from_bundle.sh"

class BundleInstallTests(unittest.TestCase):
    def test_syntax_and_explicit_apply(self):
        result=subprocess.run(["bash","-n",str(SCRIPT)],capture_output=True,text=True)
        self.assertEqual(result.returncode,0,result.stderr)
        dry=subprocess.run(["bash",str(SCRIPT)],capture_output=True,text=True)
        self.assertEqual(dry.returncode,0,dry.stderr)
        self.assertIn("DRY RUN",dry.stdout)
        body=SCRIPT.read_text()
        for required in ["PINNED_SOURCE_VERIFIED=","sha256sum -c","ZIP_SHA256=",
                         "git -C \"$STAGE\" rev-parse --verify HEAD",
                         "BACKUP_OF_INTERRUPTED_SOURCE=","[[ -s \"$ZIP\" ]]"]:
            self.assertIn(required,body)

if __name__=="__main__":
    unittest.main()
