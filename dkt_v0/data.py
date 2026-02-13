"""Dataset utilities for DKT training."""

import csv
import re
from collections import defaultdict
from typing import Dict, List, Tuple, Optional

import numpy as np


_TIME_CANDIDATES = [
    "timestamp",
    "time",
    "event_time",
    "event_timestamp",
    "ts",
    "ms_first_response_time",
]


def _parse_skill_id(raw: str) -> Optional[str]:
    if raw is None:
        return None
    s = str(raw).strip()
    if not s:
        return None
    # Handle multi-skill fields by taking the first token.
    if "~~" in s:
        s = s.replace("~~", ",")
    if any(d in s for d in [",", ";", "|", " "]):
        parts = re.split(r"[,;|\\s]+", s)
        for p in parts:
            if p:
                return p
        return None
    return s


def _parse_correct(raw: str) -> Optional[int]:
    if raw is None:
        return None
    s = str(raw).strip().lower()
    if s in {"1", "1.0", "true", "t", "yes"}:
        return 1
    if s in {"0", "0.0", "false", "f", "no"}:
        return 0
    try:
        return int(float(s))
    except ValueError:
        return None


def _parse_time(raw: str):
    if raw is None:
        return None
    s = str(raw).strip()
    if not s:
        return None
    try:
        return float(s)
    except ValueError:
        return s


def _resolve_columns(fieldnames: List[str]):
    lower_map = {c.lower(): c for c in fieldnames}
    user_col = lower_map.get("user_id")
    skill_col = lower_map.get("skill_id")
    correct_col = lower_map.get("correct")

    used_list_skill_ids = False
    if skill_col is None:
        if "list_skill_ids" in lower_map:
            skill_col = lower_map["list_skill_ids"]
            used_list_skill_ids = True
    time_col = None
    for c in _TIME_CANDIDATES:
        if c in lower_map:
            time_col = lower_map[c]
            break

    return user_col, skill_col, correct_col, time_col, used_list_skill_ids


def load_sequences(data_path: str):
    """Load sequences per user.

    Returns:
        sequences: List[List[Tuple[skill_idx, correct]]]
        num_skills: int
        skill_to_idx: Dict[str, int]
        time_col: Optional[str]
        used_list_skill_ids: bool
        skipped: int
    """
    per_user = defaultdict(list)
    skill_to_idx: Dict[str, int] = {}
    skipped = 0

    with open(data_path, "r", newline="") as f:
        reader = csv.DictReader(f)
        if not reader.fieldnames:
            raise ValueError("CSV has no header")
        user_col, skill_col, correct_col, time_col, used_list_skill_ids = _resolve_columns(reader.fieldnames)
        missing = [c for c in [user_col, skill_col, correct_col] if c is None]
        if missing:
            raise ValueError(
                "Missing required columns. Expected user_id, skill_id, correct (or list_skill_ids for skill_id)."
            )

        for row_idx, row in enumerate(reader):
            user = row.get(user_col, "").strip()
            skill_raw = row.get(skill_col, "")
            correct_raw = row.get(correct_col, "")
            if not user:
                skipped += 1
                continue
            skill_id = _parse_skill_id(skill_raw)
            correct = _parse_correct(correct_raw)
            if skill_id is None or correct is None:
                skipped += 1
                continue

            if skill_id not in skill_to_idx:
                skill_to_idx[skill_id] = len(skill_to_idx)
            skill_idx = skill_to_idx[skill_id]

            if time_col:
                t = _parse_time(row.get(time_col, ""))
                per_user[user].append((t, row_idx, skill_idx, correct))
            else:
                per_user[user].append((row_idx, skill_idx, correct))

    sequences: List[List[Tuple[int, int]]] = []
    if time_col:
        for items in per_user.values():
            # Sort by time when present; otherwise fall back to file order.
            items.sort(key=lambda x: (x[0] is None, x[0] if x[0] is not None else x[1], x[1]))
            sequences.append([(skill, corr) for _, __, skill, corr in items])
    else:
        for items in per_user.values():
            sequences.append([(skill, corr) for _, skill, corr in items])

    return sequences, len(skill_to_idx), skill_to_idx, time_col, used_list_skill_ids, skipped


def build_training_arrays(
    sequences: List[List[Tuple[int, int]]],
    num_skills: int,
    max_len: int,
):
    """Convert sequences into padded training arrays."""
    x_list = []
    next_skill_list = []
    next_correct_list = []
    mask_list = []

    for seq in sequences:
        if len(seq) < 2:
            continue
        seq_len = len(seq)
        for start in range(0, seq_len - 1, max_len):
            chunk = seq[start : start + max_len + 1]
            if len(chunk) < 2:
                continue
            input_len = len(chunk) - 1
            x = np.zeros((max_len,), dtype=np.int32)
            next_skill = np.zeros((max_len,), dtype=np.int32)
            next_correct = np.zeros((max_len,), dtype=np.float32)
            mask = np.zeros((max_len,), dtype=np.float32)

            for t in range(input_len):
                skill_t, corr_t = chunk[t]
                x[t] = skill_t + (num_skills if corr_t == 1 else 0)
                next_skill[t] = chunk[t + 1][0]
                next_correct[t] = float(chunk[t + 1][1])
                mask[t] = 1.0

            x_list.append(x)
            next_skill_list.append(next_skill)
            next_correct_list.append(next_correct)
            mask_list.append(mask)

    if not x_list:
        raise ValueError("No valid sequences found for training.")

    return (
        np.stack(x_list),
        np.stack(next_skill_list),
        np.stack(next_correct_list),
        np.stack(mask_list),
    )
