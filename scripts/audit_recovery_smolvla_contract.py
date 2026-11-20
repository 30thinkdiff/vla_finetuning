"""Read-only Recovery Expert vs pinned SmolVLA preprocessor contract preflight.

Requires 3 locally obtained exact-revision JSON files:
  1. Expert meta/info.json from pinned ModelScope SHA.
  2. SmolVLA config.json from pinned HF model SHA.
  3. SmolVLA policy_preprocessor.json from SAME HF model SHA.

No network, PyTorch, video decoding or weight loading. This is a static check,
not evidence of controller units, camera orientation or trainability.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path


class RecoverySmolVLAContractError(ValueError):
    pass


def check_metadata(dataset, policy, processor):
    def fail(message):
        raise RecoverySmolVLAContractError(message)

    if not isinstance(dataset, dict) or dataset.get("codebase_version") != "v2.1":
        fail("Expected observed Expert LeRobot v2.1 metadata")
    if dataset.get("fps") != 20:
        fail("Expert metadata must report 20Hz for this experiment")
    features = dataset.get("features")
    if not isinstance(features, dict):
        fail("Expert metadata missing features")
    for key, wanted in (("observation.state", [8]), ("action", [7])):
        feat = features.get(key)
        if not isinstance(feat, dict) or feat.get("shape") != wanted:
            fail(f"Expert {key} shape differs from observed {wanted}")
    cameras = sorted(key for key, val in features.items()
                     if isinstance(val, dict) and val.get("dtype") == "video")
    if len(cameras) != 2:
        fail("Expert must have two video features")
    if any(features[key].get("shape") != [256, 256, 3] for key in cameras):
        fail("Unexpected Expert video shape; expected HWC 256,256,3")

    if not isinstance(policy, dict) or policy.get("type") != "smolvla":
        fail("Model config is not SmolVLA")
    pin = policy.get("input_features")
    pout = policy.get("output_features")
    if not isinstance(pin, dict) or not isinstance(pout, dict):
        fail("Model input/output features missing")
    if pin.get("observation.state", {}).get("shape") != [8]:
        fail("Model expected state is not 8-dimensional")
    if pout.get("action", {}).get("shape") != [7]:
        fail("Model expected action is not 7-dimensional")
    image_targets = sorted(k for k, val in pin.items()
                           if isinstance(val, dict) and val.get("type") == "VISUAL")
    if len(image_targets) != 2 or any(pin[k].get("shape") != [3, 256, 256] for k in image_targets):
        fail("Model must have two 3x256x256 image inputs")
    chunk = policy.get("chunk_size")
    nsteps = policy.get("n_action_steps")
    if type(chunk) is not int or type(nsteps) is not int or not 0 < nsteps <= chunk:
        fail("Invalid model action chunk/execution horizon")

    if not isinstance(processor, dict):
        fail("Policy preprocessor config malformed")
    steps = processor.get("steps")
    if not isinstance(steps, list):
        fail("Policy preprocessor missing steps")
    rename = [v.get("config", {}).get("rename_map") for v in steps if isinstance(v, dict)
              and v.get("registry_name") == "rename_observations_processor"]
    if len(rename) != 1 or not isinstance(rename[0], dict):
        fail("Expected one pretrained rename step")
    mapping = rename[0]
    if set(mapping.values()) != set(image_targets):
        fail("Pretrained rename targets differ from model camera inputs")
    norm_steps = [v for v in steps if isinstance(v, dict) and
                  v.get("registry_name") == "normalizer_processor"]
    if len(norm_steps) != 1:
        fail("Expected exactly one policy normalizer")
    norm = norm_steps[0].get("config", {})
    norm_map = norm.get("norm_map")
    if not isinstance(norm_map, dict) or norm_map.get("STATE") != "MEAN_STD" or norm_map.get("ACTION") != "MEAN_STD":
        fail("Unexpected pretrained state/action normalization")
    norm_features = norm.get("features")
    if not isinstance(norm_features, dict) or norm_features.get("observation.state", {}).get("shape") != [8] or norm_features.get("action", {}).get("shape") != [7]:
        fail("Normalizer tensor feature shapes do not match model")
    if not norm_steps[0].get("state_file"):
        fail("Pretrained normalizer statistics state_file missing")

    matched = {src: dst for src, dst in mapping.items() if src in cameras}
    missing_prekeys = sorted(set(mapping) - set(cameras))
    unmatched_dataset_cameras = sorted(set(cameras) - set(mapping))
    bridge = {}
    # This is a *proposal* only: verify both camera identities and image orientation.
    if (missing_prekeys == ["observation.images.image2"]
            and unmatched_dataset_cameras == ["observation.images.wrist_image"]
            and mapping.get("observation.images.image2") in image_targets):
        bridge["observation.images.wrist_image"] = "observation.images.image2"
    elif missing_prekeys or unmatched_dataset_cameras:
        fail("Camera bridge ambiguous; manually inspect input keys before constructing a batch")
    if len(matched) + len(bridge) != 2:
        fail("Could not map both Expert camera keys")

    camera_plan = {key: mapping.get(bridge.get(key, key)) for key in cameras}
    if set(camera_plan.values()) != set(image_targets):
        fail("Camera bridge does not cover both pretrained inputs")

    warnings = [
        "Camera bridge must be explicitly applied once, before the checkpoint's rename step",
        "Image rotation/orientation and RGB pipeline not validated by JSON shape checks",
        "Recovery 7D actions: coordinate frame, scaling, rotation and gripper semantics not verified",
        "Checkpoint normalization weights, dataset statistics and train/eval stat policy not yet verified",
        "No real SmolVLA model, processor forward, action chunk, LoRA gradients or Recovery rollout tested",
    ]
    return {
        "status": "STATIC_FEATURE_BRIDGE_PROPOSED_NOT_TRAIN_READY",
        "expert_version": dataset["codebase_version"],
        "expert_fps": dataset["fps"],
        "expert_camera_keys": cameras,
        "model_input_keys": sorted(pin),
        "model_action_dim": 7,
        "model_state_dim": 8,
        "model_chunk_size": chunk,
        "model_n_action_steps": nsteps,
        "model_normalization": {"state": norm_map["STATE"], "action": norm_map["ACTION"]},
        "model_normalization_state_file": norm_steps[0]["state_file"],
        "checkpoint_rename_map": mapping,
        "proposed_preprocessor_bridge": bridge,
        "proposed_final_camera_map": camera_plan,
        "camera_semantics_verified": False,
        "action_semantics_verified": False,
        "normalizer_statistics_verified": False,
        "ready_to_train": False,
        "warnings": warnings,
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--expert-info", type=Path, required=True)
    parser.add_argument("--policy-config", type=Path, required=True)
    parser.add_argument("--policy-preprocessor", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        report = check_metadata(
            json.loads(args.expert_info.read_text(encoding="utf-8")),
            json.loads(args.policy_config.read_text(encoding="utf-8")),
            json.loads(args.policy_preprocessor.read_text(encoding="utf-8")),
        )
        if args.out.exists() or args.out.is_symlink():
            raise RecoverySmolVLAContractError("Refuse to overwrite output report")
        args.out.parent.mkdir(parents=True, exist_ok=True)
        with args.out.open("x", encoding="utf-8") as file:
            json.dump(report, file, indent=2, ensure_ascii=False, sort_keys=True)
            file.write("\n")
        print(json.dumps(report, indent=2, ensure_ascii=False))
    except (RecoverySmolVLAContractError, OSError, KeyError, TypeError, json.JSONDecodeError) as exc:
        parser.exit(2, "STOP: " + str(exc) + "\n")


if __name__ == "__main__":
    main()
