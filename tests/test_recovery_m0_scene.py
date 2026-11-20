"""Pure stdlib safety and deterministic selection tests for first M0 scene."""
import json
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"scripts"))
from prepare_recovery_m0_scene import (
    choose_first_m0_scene,SceneSelectionError,main
)


def inventory():
    root="scenes/libero_object/"
    files=[
        {"path":root+"alpha.bddl","size_bytes":200,"sha256":None},
        {"path":root+"alpha_sim_state.npy","size_bytes":1500,"sha256":None},
        {"path":root+"beta.bddl","size_bytes":200,"sha256":None},
        {"path":root+"beta_sim_state.npy","size_bytes":1500,"sha256":None},
        {"path":root+"incomplete.bddl","size_bytes":100,"sha256":None},
        {"path":"scenes/libero_goal/g.bddl","size_bytes":200,"sha256":None},
        {"path":"scenes/libero_goal/g_sim_state.npy","size_bytes":1500,"sha256":None},
        {"path":"index/selection.jsonl","size_bytes":100,"sha256":None},
    ]
    return {
        "repo_id":"ataier/LIBERO_Recovery_FailureScenes",
        "revision":"1234567890abcdef"*2+"12345678",
        "kind":"modelscope_recovery_inventory_v1",
        "files":files,
        "total_bytes":sum(e["size_bytes"] for e in files),
    }


class SceneSelectionTests(unittest.TestCase):
    def test_selects_exact_pair_and_optional_index_reproducibly(self):
        inv=inventory()
        a=choose_first_m0_scene(inv,suite="libero_object",seed=7)
        b=choose_first_m0_scene(inv,suite="libero_object",seed=7)
        self.assertEqual(a,b)
        self.assertEqual(a["status"],"PLAN_ONLY_NOT_DOWNLOADED")
        self.assertEqual(a["selection"]["pair_candidates"],2)
        names={e["path"] for e in a["files"]}
        self.assertEqual(len(names),3)
        self.assertIn(a["selection"]["sim_state"],names)
        self.assertIn(a["selection"]["bddl"],names)
        self.assertIn("index/selection.jsonl",names)

    def test_refuses_missing_pairs_wrong_revision_and_bad_seed(self):
        inv=inventory()
        with self.assertRaises(SceneSelectionError):
            choose_first_m0_scene(inv,suite="libero_90")
        inv["revision"]="main"
        with self.assertRaises(SceneSelectionError):
            choose_first_m0_scene(inv,suite="libero_object")
        inv=inventory()
        with self.assertRaises(SceneSelectionError):
            choose_first_m0_scene(inv,suite="libero_object",seed=-1)

    def test_no_overwrite_output(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder)
            path=root/"input.json"
            path.write_text(json.dumps(inventory()))
            dest=root/"manifest.json"
            main(["--inventory",str(path),"--out",str(dest)])
            before=dest.read_bytes()
            with self.assertRaises(SystemExit):
                main(["--inventory",str(path),"--out",str(dest)])
            self.assertEqual(before,dest.read_bytes())


if __name__=="__main__":
    unittest.main()
