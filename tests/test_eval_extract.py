import json
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from vla_finetuning.core import ContractError
from vla_finetuning.eval_extract import eval_rows, merge_official_eval, save_csv_new
from vla_finetuning.reports import analyze_results


def info(s1=True, s2=False):
    return {"per_task": [
        {"task_group":"libero_object", "task_id":1,
         "metrics":{"successes":[s1,s2], "video_paths":[]}},
        {"task_group":"libero_object", "task_id":8,
         "metrics":{"successes":[False]}}
    ], "overall":{"pc_success":33.33}}


class OfficialEvalTests(unittest.TestCase):
    def test_extract_real_schema_and_merge(self):
        with tempfile.TemporaryDirectory() as d:
            b, c = Path(d)/"base.json", Path(d)/"candidate.json"
            b.write_text(json.dumps(info(True, False)))
            c.write_text(json.dumps(info(True, True)))
            rows = merge_official_eval(b,c,start_seed=1000)
            self.assertEqual(len(rows),6)
            self.assertEqual(len({(r["condition_id"],r["seed"]) for r in rows}),3)
            target=Path(d)/"paired.csv"
            saved=save_csv_new(target,rows)
            self.assertEqual(saved["rows"],6)
            report=analyze_results(target)
            self.assertEqual(report["overall"]["n_pairs"],3)
            self.assertEqual(report["overall"]["gained"],1)
            self.assertEqual(report["overall"]["lost"],0)
            with self.assertRaises(FileExistsError):
                save_csv_new(target,rows)

    def test_mismatch_or_aggregate_only_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            b,c=Path(d)/"b.json",Path(d)/"c.json"
            b.write_text(json.dumps(info()))
            c.write_text(json.dumps({"overall":{"pc_success":50}}))
            with self.assertRaises(ContractError):
                merge_official_eval(b,c)
            c.write_text(json.dumps({"per_task":[
                {"task_group":"libero_object","task_id":99,
                 "metrics":{"successes":[True,False,True]}}
            ]}))
            with self.assertRaises(ContractError):
                merge_official_eval(b,c)

    def test_non_boolean_success_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/"bad.json"
            wrong=info()
            wrong["per_task"][0]["metrics"]["successes"]=["True",False]
            p.write_text(json.dumps(wrong))
            with self.assertRaises(ContractError):
                eval_rows(p,"B0",1000)


if __name__=="__main__":
    unittest.main()
