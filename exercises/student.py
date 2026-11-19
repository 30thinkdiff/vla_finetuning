"""Student-owned core implementation. Fill each TODO yourself.

These three functions are deliberately NOT implemented by ChatGPT. They are
separate from production data safety gates. Do not replace provenance evidence
with a guessed camera-view or recovery-scene label.
"""


def extract_task_labels(episode_record):
    """Extract task identities from one LeRobot episode metadata record.

    Contract:
    - v3 example: {"episode_index": 3, "tasks": ["pick up the soup"]}
      => ["pick up the soup"]
    - legacy example: {"episode_index": 3, "task_index": 7}
      => ["task_index:7"]  (numeric ID is NOT LIBERO-plus simulator task ID)
    - Remove empty/duplicate labels while preserving first-seen order.
    - Missing/unsupported task metadata must raise ValueError, never guess.

    Skills: dict/list/type checking, task and benchmark ID distinction.
    """
    # TODO(student): implement the contract; do not infer camera perturbation.
    raise NotImplementedError("Task A: implement extract_task_labels in exercises/student.py")


def make_action_chunk(episode_actions, start, horizon):
    """Construct [H,7] action labels and is_pad[H] for one episode.

    Contract:
    - episode_actions: nonempty list of 7-float action lists FROM ONE episode.
    - start: zero-based valid frame index; horizon: positive integer.
    - Use actions[start:start+horizon]; pad to H by COPYING last valid action.
    - is_pad is False for valid actions and True for padding; NEVER cross Episode.
    - Raise ValueError on invalid shape/start/horizon.
    - Return (actions, is_pad) as plain Python lists, not GPU tensors.

    Skills: temporal indexing, action chunk, padding and loss mask.
    """
    # TODO(student): start with one 3-step episode and H=5 on paper.
    raise NotImplementedError("Task B: implement make_action_chunk in exercises/student.py")


def paired_outcomes(rows, baseline="B0", candidate="L1"):
    """Compare SUCCESS of two policies on exactly matched evaluation cases.

    Input rows: list of dicts: domain, condition_id, seed, method, success(bool).
    Key: (domain, condition_id, seed). Require exactly one row per method/key,
    matched key sets and boolean success; raise ValueError otherwise.

    Return dict: n_pairs, gained, lost, both_success, both_failed.
      gained: baseline False, candidate True.
      lost: baseline True, candidate False.
    Do not make significance claims from a few preselected episodes.

    Skills: paired experimental design, dictionary grouping, data integrity.
    """
    # TODO(student): do not compare two unmatched total success percentages.
    raise NotImplementedError("Task C: implement paired_outcomes in exercises/student.py")
