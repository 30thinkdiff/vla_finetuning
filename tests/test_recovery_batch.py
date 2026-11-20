"""Pure stdlib regression tests for Recovery v2.1 real sample builder."""
import sys
import unittest
from pathlib import Path
from unittest.mock import patch
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/"src"))
from vla_finetuning.recovery_batch import (
    RecoveryBatchError, select_action_window, require_finite_vector,
    decode_exact_rgb_frame,
)


def fake_episode(n=65, episode_index=0, task_index=2):
    return [
        {"episode_index":episode_index, "task_index":task_index, "frame_index":k,
         "timestamp":k/20.0, "action": [float(k)]*6 + [float(k % 2)],
         "observation.state":[0.0]*8}
        for k in range(n)
    ]


class RecoveryBatchTests(unittest.TestCase):
    def test_full_chunk_is_not_padded(self):
        rows = fake_episode(65)
        result=select_action_window(rows,10,50)
        self.assertEqual(len(result["action"]),50)
        self.assertEqual(result["source_frame_indices"],list(range(10,60)))
        self.assertEqual(result["valid_action_steps"],50)
        self.assertEqual(sum(result["action_is_pad"]),0)
        self.assertEqual(result["action"][0][0],10)
        self.assertEqual(result["action"][-1][0],59)

    def test_episode_tail_clamps_and_marks_mask(self):
        rows=fake_episode(65)
        result=select_action_window(rows,62,50)
        self.assertEqual(result["valid_action_steps"],3)
        self.assertEqual(sum(result["action_is_pad"]),47)
        self.assertEqual(result["source_frame_indices"][:4],[62,63,64,64])
        self.assertTrue(all(k==64 for k in result["source_frame_indices"][3:]))
        self.assertEqual(result["action"][-1],result["action"][2])
        # This directly mirrors LeRobot DatasetReader clamp-to-episode-last,
        # not cross-episode indexing or all-zero surrogate actions.

    def test_missing_or_mismatched_episode_is_rejected(self):
        rows=fake_episode()
        for frame in (-1,65):
            with self.assertRaises(RecoveryBatchError):
                select_action_window(rows,frame,50)
        with self.assertRaises(RecoveryBatchError):
            select_action_window(rows,0,0)
        rows[30]["episode_index"]=1
        with self.assertRaises(RecoveryBatchError):
            select_action_window(rows,10,50)
        rows=fake_episode()
        rows[21]["task_index"]=5
        with self.assertRaises(RecoveryBatchError):
            select_action_window(rows,21,50)

    def test_bad_vectors_rejected(self):
        for v in ([0]*6,[0]*6+[float("nan")],[0]*6+["1"],"0"*7):
            with self.assertRaises(RecoveryBatchError):
                require_finite_vector(v,7,"action")
        self.assertEqual(require_finite_vector([0]*7,7,"action"),[0.0]*7)

    @patch("vla_finetuning.recovery_batch.subprocess.run")
    def test_exact_frame_ffmpeg_command_and_rgb_bytes(self, run):
        raw=b"\x20\x30\x40"*4
        run.return_value=SimpleNamespace(returncode=0, stdout=raw, stderr=b"")
        output=decode_exact_rgb_frame("/tmp/test.mp4",62,width=2,height=2)
        self.assertEqual(output,raw)
        args=run.call_args.args[0]
        self.assertIn("select=eq(n\\,62)",args)
        self.assertIn("rawvideo",args)
        self.assertIn("rgb24",args)
        self.assertEqual(args[-1],"pipe:1")

    @patch("vla_finetuning.recovery_batch.subprocess.run")
    def test_reject_short_or_failed_ffmpeg_output(self, run):
        run.return_value=SimpleNamespace(returncode=1,stdout=b"",stderr=b"AV1 error")
        with self.assertRaises(RecoveryBatchError):
            decode_exact_rgb_frame("/tmp/test.mp4",0,width=2,height=2)
        run.return_value=SimpleNamespace(returncode=0,stdout=b"\x11\x22\x33",stderr=b"")
        with self.assertRaises(RecoveryBatchError):
            decode_exact_rgb_frame("/tmp/test.mp4",0,width=2,height=2)

if __name__=="__main__":
    unittest.main()
