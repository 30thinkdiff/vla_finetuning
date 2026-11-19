"""Runs when torch is already installed; never installs torch or uses CUDA."""
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
try:
    import torch
except ImportError:
    torch = None

from vla_finetuning.flow_math import (audit_trainable_parameters, masked_velocity_mse,
                                      torch_flow_pair)


@unittest.skipIf(torch is None, "PyTorch optional; no torch on this CPU-only machine")
class FlowTorchTests(unittest.TestCase):
    def test_flow_target_and_padding_gradient(self):
        action = torch.tensor([[[0.2, 0.8], [3., 4.]]])
        noise = torch.tensor([[[0.9, -0.4], [0., 0.]]])
        xt, target = torch_flow_pair(action, noise, torch.tensor([0.25]))
        self.assertTrue(torch.allclose(xt[0, 0], torch.tensor([0.375, 0.5])))
        pred = torch.nn.Parameter(target.detach().clone())
        with torch.no_grad():
            pred[0, 0, :] += torch.tensor([0.1, 0.1])
            pred[0, 1, :] += 100.0
        loss = masked_velocity_mse(pred, target, torch.tensor([[False, True]]))
        self.assertAlmostEqual(float(loss.detach()), 0.01, places=5)
        loss.backward()
        self.assertTrue(torch.all(pred.grad[0, 1] == 0))
        self.assertTrue(torch.all(pred.grad[0, 0].abs() > 0))

    def test_all_padding_rejected(self):
        x = torch.zeros(1, 2, 7)
        with self.assertRaises(ValueError):
            masked_velocity_mse(x, x, torch.ones(1, 2, dtype=torch.bool))

    def test_lora_gradients_and_frozen_weight(self):
        from vla_finetuning.lora_lab import two_step_lora_demo
        report = two_step_lora_demo()
        self.assertEqual(report["total_parameters"], 16)
        self.assertEqual(report["trainable_parameters"], 10)
        self.assertTrue(report["frozen_base_unchanged"])
        self.assertTrue(report["lora_A_updated"])
        self.assertTrue(report["lora_B_updated"])
        self.assertEqual(report["steps"][0]["A_grad_norm"], 0.0)
        self.assertGreater(report["steps"][0]["B_grad_norm"], 0.0)
        self.assertGreater(report["steps"][1]["A_grad_norm"], 0.0)

    def test_lora_audit_is_not_fake(self):
        m = torch.nn.Linear(3, 2)
        result = audit_trainable_parameters(m)
        self.assertEqual(result["total"], 8)
        self.assertEqual(result["trainable"], 8)
        self.assertFalse(result["has_trainable_lora"])


if __name__ == "__main__":
    unittest.main()
