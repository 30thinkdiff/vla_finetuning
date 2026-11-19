"""Exact SmolVLA-style flow-matching targets, with an intentionally small numeric example.

Convention used in the pinned robot-lab SmolVLA source:
    x_t = t * eps + (1-t) * action
    target_velocity = eps - action
The t=0 endpoint is clean action; t=1 is noise. The prediction is a velocity,
not a directly regressed final action.
"""
import math


def two_number_example():
    action = (0.2, 0.8)
    noise = (0.9, -0.4)
    t = 0.25
    xt = tuple(t * e + (1.0 - t) * a for e, a in zip(noise, action))
    target = tuple(e - a for e, a in zip(noise, action))
    prediction = (0.6, -1.1)
    loss = sum((p - u) ** 2 for p, u in zip(prediction, target)) / len(target)
    return {"action": list(action), "noise": list(noise), "t": t,
            "noisy_action": list(xt), "target_velocity": list(target),
            "predicted_velocity": list(prediction), "mean_squared_error": loss}


def torch_flow_pair(action, noise, t):
    """action/noise: [B,H,D]; t: [B] or [B,1,1]; returns (noisy, target).

    The function does not sample t or noise: the caller supplies both to make
    experiments deterministic and independently auditable.
    """
    import torch
    if action.ndim != 3 or action.shape != noise.shape:
        raise ValueError("action and noise must have identical [B,H,D] shapes")
    if not action.is_floating_point() or not noise.is_floating_point():
        raise TypeError("action and noise must be floating point tensors")
    if not torch.is_tensor(t):
        raise TypeError("t must be a tensor")
    if t.ndim == 1 and t.shape[0] == action.shape[0]:
        t = t.reshape(-1, 1, 1)
    elif t.ndim != 3 or t.shape != (action.shape[0], 1, 1):
        raise ValueError("t must be [B] or [B,1,1]")
    if not bool(torch.isfinite(action).all() and torch.isfinite(noise).all()
                and torch.isfinite(t).all() and (t >= 0).all() and (t <= 1).all()):
        raise ValueError("non-finite action/noise/t or t outside [0,1]")
    return t * noise + (1.0 - t) * action, noise - action


def masked_velocity_mse(pred_velocity, target_velocity, is_pad):
    """MSE averaged over all valid action scalars; is_pad=True means padding.

    This educational loss is equivalent to a masked squared velocity error.
    Do NOT substitute this blindly for an official LeRobot policy.forward loss;
    verify its target, normalization and masking in the pinned source first.
    """
    import torch
    if pred_velocity.ndim != 3 or pred_velocity.shape != target_velocity.shape:
        raise ValueError("velocity tensors must share [B,H,D] shape")
    if is_pad.dtype != torch.bool or is_pad.shape != pred_velocity.shape[:2]:
        raise ValueError("is_pad must be bool [B,H]")
    valid = (~is_pad).unsqueeze(-1).expand_as(pred_velocity)
    if not bool(valid.any()):
        raise ValueError("all time steps are padding")
    if not bool(torch.isfinite(pred_velocity).all() and torch.isfinite(target_velocity).all()):
        raise ValueError("non-finite velocity")
    return ((pred_velocity - target_velocity).square()).masked_select(valid).mean()


def audit_trainable_parameters(model):
    """Inspect PyTorch model; does not mutate or run model; counts are actual tensors."""
    result = {"total": 0, "trainable": 0, "trainable_names": [],
              "has_trainable_lora": False, "unexpected_trainable_names": []}
    for name, parameter in model.named_parameters():
        result["total"] += parameter.numel()
        if parameter.requires_grad:
            result["trainable"] += parameter.numel()
            result["trainable_names"].append(name)
            if "lora_" in name.lower():
                result["has_trainable_lora"] = True
            else:
                result["unexpected_trainable_names"].append(name)
    return result
