"""Student-facing tests, intentionally separate from CI.

Expected to fail with NotImplementedError before you implement exercises/student.py.
Run: PYTHONPATH=src python -m unittest discover -s exercises/tests -v
"""
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from student import extract_task_labels, make_action_chunk, paired_outcomes


class TaskATest(unittest.TestCase):
    def test_v3_text_tasks(self):
        self.assertEqual(
            extract_task_labels({"episode_index": 11,
                                 "tasks": ["pick up the soup", "", "pick up the soup", "place cheese"]}),
            ["pick up the soup", "place cheese"])
    def test_legacy_index_not_simulator_task_id(self):
        self.assertEqual(extract_task_labels({"task_index": 3}), ["task_index:3"])
    def test_missing_task_is_error(self):
        with self.assertRaises(ValueError):
            extract_task_labels({"episode_index": 100})


class TaskBTest(unittest.TestCase):
    def test_tail_padding_does_not_cross_episode(self):
        a = [[float(t)] * 7 for t in (1, 2, 3)]
        chunk, mask = make_action_chunk(a, start=1, horizon=4)
        self.assertEqual(chunk, [[2.0] * 7, [3.0] * 7, [3.0] * 7, [3.0] * 7])
        self.assertEqual(mask, [False, False, True, True])
        self.assertIsNot(chunk[2], chunk[3], "No aliasing between padded actions")
    def test_first_frame_no_padding(self):
        a = [[float(t)] * 7 for t in (1, 2, 3)]
        chunk, mask = make_action_chunk(a, 0, 2)
        self.assertEqual(chunk, a[:2])
        self.assertEqual(mask, [False, False])
    def test_invalid_shape_or_bounds(self):
        with self.assertRaises(ValueError):
            make_action_chunk([[1.0, 2.0]], 0, 4)
        with self.assertRaises(ValueError):
            make_action_chunk([[0.0] * 7], 1, 2)


class TaskCTest(unittest.TestCase):
    @staticmethod
    def cases():
        outcomes = [(False, True), (True, False), (True, True), (False, False)]
        rows = []
        for i, (b, c) in enumerate(outcomes):
            rows.extend([
                {"domain": "libero_plus", "condition_id": "view_" + str(i),
                 "seed": 1000, "method": "B0", "success": b},
                {"domain": "libero_plus", "condition_id": "view_" + str(i),
                 "seed": 1000, "method": "L1", "success": c}])
        return rows
    def test_paired_flips(self):
        result = paired_outcomes(self.cases(), "B0", "L1")
        self.assertEqual(result, {"n_pairs": 4, "gained": 1, "lost": 1,
                                  "both_success": 1, "both_failed": 1})
    def test_missing_pair_is_error(self):
        rows = self.cases()
        rows.pop()
        with self.assertRaises(ValueError):
            paired_outcomes(rows, "B0", "L1")
    def test_duplicate_pair_is_error(self):
        rows = self.cases()
        rows.append(dict(rows[0]))
        with self.assertRaises(ValueError):
            paired_outcomes(rows, "B0", "L1")
    def test_success_string_not_boolean(self):
        rows = self.cases()
        rows[0]["success"] = "False"
        with self.assertRaises(ValueError):
            paired_outcomes(rows, "B0", "L1")


if __name__ == "__main__":
    unittest.main()
