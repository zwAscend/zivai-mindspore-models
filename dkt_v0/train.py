"""Train a DKT model with MindSpore (GPU only, Docker only).

How to run (inside container):
  A) Build image:
     docker build -t msmodels-dkt:gpu /workspace/dkt

  B) Train:
     docker run --rm --gpus all -v <host_msmodels>:/workspace -w /workspace/dkt msmodels-dkt:gpu \
       python train.py --data_path /workspace/dkt/<csv> --epochs 10 --batch_size 64
"""

import argparse
import json
import os
import sys
from pathlib import Path
import random


def _ensure_docker():
    if os.path.exists("/.dockerenv"):
        return
    print(
        "ERROR: This training script must be run inside the Docker container.\n"
        "Run via:\n"
        "  docker run --rm --gpus all -v <host_msmodels>:/workspace -w /workspace/dkt msmodels-dkt:gpu \\\n"
        "    python train.py --data_path /workspace/dkt/<csv> --epochs 10 --batch_size 64\n",
        file=sys.stderr,
    )
    sys.exit(1)


def _resolve_data_path(raw_path: str) -> Path:
    data_path = Path(raw_path)
    if not data_path.is_absolute():
        data_path = (Path.cwd() / data_path).resolve()
    if not data_path.exists():
        if str(data_path).startswith("/home/"):
            suggestion = Path("/workspace/dkt") / data_path.name
            print(
                "Data path looks like a host path. Inside Docker, use:\n"
                f"  {suggestion}",
                file=sys.stderr,
            )
        raise FileNotFoundError(f"Data file not found: {data_path}")
    return data_path


def parse_args():
    parser = argparse.ArgumentParser(description="Train DKT with MindSpore")
    parser.add_argument("--data_path", required=True, help="Path to CSV data")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--max_len", type=int, default=100)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--hidden_size", type=int, default=128)
    parser.add_argument("--emb_dim", type=int, default=64)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--log_every", type=int, default=50, help="Log every N steps")
    parser.add_argument(
        "--sink_mode",
        type=int,
        default=1,
        help="1 to enable dataset sink mode (faster), 0 to disable (more frequent logging)",
    )
    return parser.parse_args()


def main():
    _ensure_docker()

    # Heavy deps (MindSpore/Numpy) are imported only after Docker check.
    import numpy as np
    import mindspore as ms
    from mindspore import context, nn, ops
    import mindspore.dataset as ds

    from model import DKTModel
    from data import load_sequences, build_training_arrays

    class DKTWithLoss(nn.Cell):
        """DKT loss wrapper."""

        def __init__(self, model: nn.Cell):
            super().__init__()
            self.network = model
            self.gather = ops.GatherD()

        def construct(self, x, next_skill, next_correct, mask):
            pred = self.network(x)
            idx = ops.ExpandDims()(next_skill, -1)
            pred_skill = self.gather(pred, 2, idx)
            pred_skill = ops.Squeeze(axis=-1)(pred_skill)
            pred_skill = ops.clip_by_value(pred_skill, 1e-7, 1.0 - 1e-7)
            loss = -(next_correct * ops.log(pred_skill) + (1.0 - next_correct) * ops.log(1.0 - pred_skill))
            loss = loss * mask
            denom = ops.reduce_sum(mask) + 1e-9
            return ops.reduce_sum(loss) / denom

    class LossRecorder(ms.Callback):
        def __init__(self, output_dir: Path, model: nn.Cell, steps_per_epoch: int, log_every: int):
            super().__init__()
            self.output_dir = output_dir
            self.model = model
            self.steps_per_epoch = steps_per_epoch
            self.log_every = max(1, log_every)
            self.metrics_path = output_dir / "metrics.csv"
            self._epoch_loss_sum = 0.0
            self._epoch_steps = 0
            if not self.metrics_path.exists():
                with open(self.metrics_path, "w", encoding="utf-8") as f:
                    f.write("epoch,loss\n")

        def on_train_epoch_begin(self, run_context):
            self._epoch_loss_sum = 0.0
            self._epoch_steps = 0

        def on_train_step_end(self, run_context):
            cb_params = run_context.original_args()
            loss = cb_params.net_outputs
            if isinstance(loss, (tuple, list)):
                loss = loss[0]
            loss_val = float(loss.asnumpy())
            self._epoch_loss_sum += loss_val
            self._epoch_steps += 1
            if self._epoch_steps % self.log_every == 0 or self._epoch_steps == self.steps_per_epoch:
                epoch = cb_params.cur_epoch_num
                step = self._epoch_steps
                print(f"Epoch {epoch} step {step}/{self.steps_per_epoch} | loss={loss_val:.6f}")

        def on_train_epoch_end(self, run_context):
            epoch = run_context.original_args().cur_epoch_num
            mean_loss = self._epoch_loss_sum / max(self._epoch_steps, 1)
            with open(self.metrics_path, "a", encoding="utf-8") as f:
                f.write(f"{epoch},{mean_loss:.6f}\n")
            ckpt_path = self.output_dir / f"dkt_epoch_{epoch}.ckpt"
            ms.save_checkpoint(self.model, str(ckpt_path))
            print(f"Epoch {epoch}: loss={mean_loss:.6f} | checkpoint={ckpt_path}")

    args = parse_args()

    context.set_context(mode=context.GRAPH_MODE, device_target="GPU")

    random.seed(args.seed)
    np.random.seed(args.seed)
    ms.set_seed(args.seed)

    base_dir = Path(__file__).resolve().parent
    output_dir = base_dir / "model_output"
    output_dir.mkdir(parents=True, exist_ok=True)

    data_path = _resolve_data_path(args.data_path)

    print(f"Loading data from: {data_path}")
    sequences, num_skills, _, time_col, used_list_skill_ids, skipped = load_sequences(str(data_path))
    print(f"Users: {len(sequences)} | Skills: {num_skills} | Skipped rows: {skipped}")
    if time_col:
        print(f"Sorting by time column: {time_col}")
    if used_list_skill_ids:
        print("Using list_skill_ids as skill_id (first token per row)")

    x_arr, next_skill_arr, next_correct_arr, mask_arr = build_training_arrays(
        sequences, num_skills=num_skills, max_len=args.max_len
    )

    print(f"Training samples: {x_arr.shape[0]} | seq_len: {x_arr.shape[1]}")

    train_dataset = ds.NumpySlicesDataset(
        {"x": x_arr, "next_skill": next_skill_arr, "next_correct": next_correct_arr, "mask": mask_arr},
        shuffle=True,
    )
    train_dataset = train_dataset.batch(args.batch_size, drop_remainder=False)
    steps_per_epoch = train_dataset.get_dataset_size()

    model = DKTModel(num_skills=num_skills, emb_dim=args.emb_dim, hidden_size=args.hidden_size, rnn_type="gru")
    loss_net = DKTWithLoss(model)
    optimizer = nn.Adam(loss_net.trainable_params(), learning_rate=args.lr)
    train_net = nn.TrainOneStepCell(loss_net, optimizer)

    ms_model = ms.Model(train_net)

    with open(output_dir / "run_config.json", "w", encoding="utf-8") as f:
        json.dump(vars(args), f, indent=2)

    print("Starting training...")
    ms_model.train(
        args.epochs,
        train_dataset,
        callbacks=[LossRecorder(output_dir, model, steps_per_epoch, args.log_every)],
        dataset_sink_mode=bool(args.sink_mode),
    )
    print("Training complete.")


if __name__ == "__main__":
    main()
