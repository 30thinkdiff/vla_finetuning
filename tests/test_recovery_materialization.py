"""Cross-dataset materialization gate tests: schema-only, no actual IO beyond fixtures."""
import json
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from vla_finetuning.core import ContractError
from vla_finetuning.recovery_experiments import EXPERT, NORMAL
from vla_finetuning.recovery_merge_plan import plan_mixed_v3_dataset


class MaterializationContractTests(unittest.TestCase):
    def setUp(self):
        self.t = tempfile.TemporaryDirectory()
        self.addCleanup(self.t.cleanup)
        self.root = Path(self.t.name)
        def write(name, obj):
            p = self.root/name
            p.write_text(json.dumps(obj))
            return p
        def info(fps=20):
            return {"codebase_version":"v3.0", "fps":fps,
                    "features":{"action":{"shape":[7]}, "observation.state":{"shape":[8]},
                                "front_cam":{"dtype":"video","shape":[256,256,3]},
                                "wrist_cam":{"dtype":"video","shape":[256,256,3]}}}
        self.einfo=write("expert.json",info())
        self.ninfo=write("normal.json",info())
        def row(ds, ep, group):
            return {"dataset_id":ds, "episode_id":ep,"source_group":group,
                    "source_evidence":"review/asset_"+group, "review_status":"verified"}
        self.split={"kind":"recovery_source_split_v1","real_evidence_reviewed":True,
                    "datasets":{EXPERT:{"revision":"fixed1","source_identity_checked":True},
                                NORMAL:{"revision":"fixed2","source_identity_checked":True}},
                    "reserved_benchmark_source_groups":["official-test-source"],
                    "splits":{"train":[row(EXPERT,1,"r1"),row(NORMAL,3,"n1")],
                              "validation":[row(EXPERT,2,"r2")],
                              "test":[row(EXPERT,4,"official-test-source")]}}
        self.align={"kind":"recovery_alignment_v1","verified":True,
                    "target_dataset_id":"local/recovery_normal_mixed_v3",
                    "metadata_paths":{EXPERT:str(self.einfo),NORMAL:str(self.ninfo)},
                    "camera_features":{EXPERT:{"front":"front_cam","wrist":"wrist_cam"},
                                       NORMAL:{"front":"front_cam","wrist":"wrist_cam"}},
                    "evidence":{key:"manually_reviewed/"+key for key in
                                ("gripper","action_coordinates","state_order","camera_transforms",
                                 "timebase","normalization","task_labels")}}
        self.write=write

    def test_plan_does_not_materialize(self):
        result=plan_mixed_v3_dataset(self.write("split.json",self.split),
                                     self.write("align.json",self.align))
        self.assertEqual(result["status"],"PLAN_ONLY_NO_DATASET_MATERIALIZED")
        self.assertEqual(len(result["episode_mapping"]),2)
        self.assertEqual(result["source_fps"][EXPERT],20)

    def test_reject_unknown_alignment(self):
        self.align["verified"]=False
        with self.assertRaises(ContractError):
            plan_mixed_v3_dataset(self.write("s.json",self.split),self.write("a.json",self.align))

    def test_reject_mismatched_timing(self):
        self.ninfo.write_text(json.dumps({"codebase_version":"v3.0","fps":10,
              "features":{"action":{"shape":[7]},"observation.state":{"shape":[8]},
                          "front_cam":{"dtype":"video"},"wrist_cam":{"dtype":"video"}}}))
        with self.assertRaises(ContractError):
            plan_mixed_v3_dataset(self.write("s.json",self.split),self.write("a.json",self.align))

    def test_reject_v2_until_converted(self):
        d=json.loads(self.einfo.read_text());d["codebase_version"]="v2.1"
        self.einfo.write_text(json.dumps(d))
        with self.assertRaises(ContractError):
            plan_mixed_v3_dataset(self.write("s.json",self.split),self.write("a.json",self.align))


if __name__=="__main__":
    unittest.main()
