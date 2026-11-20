"""CPU/no-external-data unit tests for real Recovery payload summary logic."""
import sys
import unittest
from pathlib import Path
from unittest.mock import patch
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from audit_recovery_payload import (
    PayloadAuditError, frame_integrity, numeric_vector,
    summarize_vectors, ffprobe_video
)


class RecoveryPayloadAuditTests(unittest.TestCase):
    def test_numeric_vector_shape_and_nonfinite(self):
        self.assertEqual(numeric_vector([1,0,0,0,0,0,1],7,"action")[-1],1.0)
        for values in ([0]*6, [0]*6+[float("nan")], "not a vector", [0]*6+["1"]):
            with self.assertRaises(PayloadAuditError):
                numeric_vector(values,7,"action")

    def test_summarize_vector_representative_values(self):
        summary = summarize_vectors([[0]*6+[0.0], [1]*6+[1.0]], 7, "action")
        self.assertEqual(summary["count"],2)
        self.assertEqual(summary["last_dim_rounded_values_sample"],[0.0,1.0])
        self.assertEqual(summary["dimensions"][0]["mean"],0.5)

    def test_frame_integrity_and_timing(self):
        rows = [
            dict(episode_index=15,frame_index=i,timestamp=i*0.05,task_index=2)
            for i in range(3)
        ]
        result=frame_integrity(rows,15,20)
        self.assertEqual(result["frame_start_end"],[0,2])
        self.assertTrue(result["timing_within_3ms_of_nominal"])
        bad=[dict(x) for x in rows];bad[2]["frame_index"]=3
        with self.assertRaises(PayloadAuditError):
            frame_integrity(bad,15,20)
        bad=[dict(x) for x in rows];bad[1]["timestamp"]=bad[0]["timestamp"]
        with self.assertRaises(PayloadAuditError):
            frame_integrity(bad,15,20)
        bad=[dict(x) for x in rows];bad[1]["task_index"]=10
        with self.assertRaises(PayloadAuditError):
            frame_integrity(bad,15,20)

    @patch("audit_recovery_payload.subprocess.run")
    def test_video_codec_and_decoded_frames(self, run):
        run.return_value=SimpleNamespace(
            returncode=0, stderr="", stdout='{"streams":[{"codec_name":"av1",'
            '"width":256,"height":256,"avg_frame_rate":"20/1","nb_read_frames":"629"}]}'
        )
        result=ffprobe_video("/tmp/fake.mp4",629)
        self.assertEqual(result["decoded_frames"],629)
        with self.assertRaises(PayloadAuditError):
            ffprobe_video("/tmp/fake.mp4",628)
        run.assert_called()

    @patch("audit_recovery_payload.subprocess.run")
    def test_video_decoder_error_is_not_silently_ignored(self, run):
        run.return_value=SimpleNamespace(returncode=1, stderr="Missing AV1 codec", stdout="")
        with self.assertRaises(PayloadAuditError):
            ffprobe_video("/tmp/fake.mp4",100)


if __name__ == "__main__":
    unittest.main()
