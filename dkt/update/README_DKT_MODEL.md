# DKT Model (Computer Science Form 3–4)

This document describes the curriculum-anchored DKT pipeline implemented in `dkt-lstm-2.ipynb`.

## 1. Current Run Snapshot
Latest notebook outputs (from saved cell logs) show:
- `val_auc` best around `0.5269`
- `test`: `auc=0.5342`, `acc=0.5253`, `logloss=0.6926`

Interpretation:
- This is near-random predictive quality.
- The current synthetic/bootstrapped data likely does not encode enough consistent learning signal yet.
- Architecture was upgraded moderately to stacked LSTM (2 layers + dropout), but data quality and event realism remain the primary bottleneck.

## 2. Input Data Contract
Notebook expects files in the same directory as the notebook runtime (`Path.cwd()`):
- `cs_form3_form4_dkt_taxonomy.csv`
- `cs_dkt_events.csv`

Required event columns:
- `student_id`
- `skill_code`
- `is_correct` (0/1)
- `event_time` (ISO datetime)

Optional columns (recommended):
- `subject_code`, `score`, `max_score`, `attempt_id`, `question_id`, `id`

## 3. Frozen Skill Map (Critical)
The model uses a frozen mapping artifact:
- `skill_map_v1.json`

Rules:
1. Skill indices are built from taxonomy once and frozen.
2. Training/inference must load this map; do not rebuild from raw events.
3. Taxonomy changes require a new mapping/model version.

## 4. Model Architecture (Moderate Upgrade)
`DKTNetLSTM` now uses:
- Embedding dim: `96`
- Hidden dim: `192`
- LSTM layers: `2`
- Dropout: `0.20` (between stacked layers)

Training defaults:
- `MAX_LEN=50`, `STRIDE=25`, `BATCH_SIZE=32`
- `LEARNING_RATE=8e-4`, `EPOCHS=25`, `PATIENCE=4`

## 5. Artifacts Produced
All outputs are saved in the notebook directory:
- Best checkpoint: `checkpoints/best_dkt_lstm.ckpt`
- Cloud checkpoint copy: `dkt_lstm_cloud.ckpt`
- Edge MindIR: `dkt_lstm_edge.mindir`
- Skill map: `skill_map_v1.json`
- Metadata: `model_meta.json`

## 6. Cloud Integration
Use full model checkpoint (`dkt_lstm_cloud.ckpt`) in cloud service.

Suggested API contracts:
- `POST /dkt/update`
- `GET /dkt/mastery/<student_id>`

`POST /dkt/update` input (example):
```json
{
  "student_id": "stu_0001",
  "subject_code": "computer_science",
  "events": [
    {
      "skill_code": "CS.F3.PROG.TEST_DEBUG_PROGRAMS",
      "is_correct": 1,
      "score": 2.0,
      "max_score": 2.0,
      "event_time": "2026-03-05T12:00:00Z",
      "attempt_id": "att_123",
      "question_id": "q_456"
    }
  ]
}
```

Response should include:
- `average_mastery`
- `risk_level`
- `weak_skills[]` with `skill_code` and `mastery_prob`
- optionally full `mastery_vector`

### Python API Integration Example (FastAPI)
Below is a minimal reference implementation for integrating the trained model into API routes.

```python
from pathlib import Path
from typing import Dict, List
import json
import numpy as np
import mindspore as ms
from fastapi import FastAPI
from pydantic import BaseModel

from dkt_model import DKTNetLSTM  # use your model class from notebook/script

ARTIFACT_DIR = Path(".")
SKILL_MAP_PATH = ARTIFACT_DIR / "skill_map_v1.json"
CKPT_PATH = ARTIFACT_DIR / "dkt_lstm_cloud.ckpt"

with open(SKILL_MAP_PATH, "r", encoding="utf-8") as f:
    skill_map = json.load(f)
skill_to_idx: Dict[str, int] = skill_map["skill_to_idx"]
idx_to_skill: Dict[int, str] = {v: k for k, v in skill_to_idx.items()}
num_skills = len(skill_to_idx)

# Must match training config
model = DKTNetLSTM(
    num_skills=num_skills,
    embed_dim=96,
    hidden_dim=192,
    num_layers=2,
    dropout=0.20,
)
param_dict = ms.load_checkpoint(str(CKPT_PATH))
ms.load_param_into_net(model, param_dict)
model.set_train(False)

app = FastAPI(title="DKT Service")

# In production, store this in Redis/Postgres instead of memory.
student_history: Dict[str, List[int]] = {}


class Event(BaseModel):
    skill_code: str
    is_correct: int
    event_time: str


class UpdateRequest(BaseModel):
    student_id: str
    events: List[Event]


def encode_event(skill_idx: int, is_correct: int, num_skills: int) -> int:
    # 1..N for incorrect, N+1..2N for correct (0 reserved for pad)
    return (skill_idx + 1) + (num_skills if int(is_correct) == 1 else 0)


def infer_mastery(tokens: List[int]) -> np.ndarray:
    x = np.array([tokens], dtype=np.int32)  # [B=1, T]
    probs = model(ms.Tensor(x)).asnumpy()[0]  # [T, num_skills]
    return probs[-1]  # mastery after latest event


@app.post("/dkt/update")
def dkt_update(req: UpdateRequest):
    seq = student_history.get(req.student_id, [])
    for e in req.events:
        if e.skill_code not in skill_to_idx:
            continue
        skill_idx = skill_to_idx[e.skill_code]
        seq.append(encode_event(skill_idx, e.is_correct, num_skills))

    # keep recent context window used by model
    seq = seq[-50:]
    student_history[req.student_id] = seq

    if not seq:
        return {"student_id": req.student_id, "mastery_vector": {}, "weak_skills": []}

    mastery = infer_mastery(seq)
    mastery_dict = {idx_to_skill[i]: float(mastery[i]) for i in range(num_skills)}
    weak = sorted(mastery_dict.items(), key=lambda x: x[1])[:5]

    avg = float(np.mean(mastery))
    risk = "high" if avg < 0.45 else ("medium" if avg < 0.65 else "low")
    return {
        "student_id": req.student_id,
        "average_mastery": avg,
        "risk_level": risk,
        "weak_skills": [{"skill_code": s, "mastery_prob": p} for s, p in weak],
    }


@app.get("/dkt/mastery/{student_id}")
def dkt_mastery(student_id: str):
    seq = student_history.get(student_id, [])
    if not seq:
        return {"student_id": student_id, "mastery_vector": {}, "weak_skills": []}

    mastery = infer_mastery(seq)
    mastery_dict = {idx_to_skill[i]: float(mastery[i]) for i in range(num_skills)}
    weak = sorted(mastery_dict.items(), key=lambda x: x[1])[:5]
    return {
        "student_id": student_id,
        "mastery_vector": mastery_dict,
        "weak_skills": [{"skill_code": s, "mastery_prob": p} for s, p in weak],
    }
```

Run example:
```bash
uvicorn dkt_api:app --host 0.0.0.0 --port 8080
```

Integration notes:
1. Keep `skill_map_v1.json` and checkpoint version aligned.
2. If taxonomy changes, publish new `skill_map` + model version (do not hot-swap old map).
3. Persist student sequence state in DB/cache for multi-instance deployments.

## 7. Edge Integration (MindSpore Lite)
Notebook exports `dkt_lstm_edge.mindir` for edge conversion.

Run conversion where `converter_lite` is available:
```bash
converter_lite \
  --fmk=MINDIR \
  --modelFile=dkt_lstm_edge.mindir \
  --outputFile=dkt_lstm_edge_lite \
  --optimize=ascend_oriented
```

Deploy converted model on edge runtime (Ascend-oriented environment), and keep cloud model as source of truth for periodic retraining.

## 8. Data Feedback Loop (Accuracy Improvement)
For continuous improvement:
1. Export `interaction_events`-style rows from DB regularly.
2. Retrain with frozen mapping.
3. Validate (`AUC/logloss`) against holdout set.
4. Promote only better model versions.
5. Track version in metadata and inference logs.

## 9. Known Priority Improvements
1. Improve event label quality (teacher-final scores, better skill tagging, less noisy synthetic patterns).
2. Add richer features (response length bucket, question type, confidence) if model is extended.
3. Add calibration and thresholding for risk categorization.
4. Consider GRU/LSTM comparison and hyperparameter sweep once real data volume increases.
