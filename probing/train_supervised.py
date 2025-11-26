import argparse
import itertools
import json
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from probing.cache_dataset import load_cache_split
from probing.feature_ops import build_feature_matrix
from utils import parser_add, params_add


def _default_parallel_workers(cfg: dict) -> int:
    return int(cfg.get("parallel_workers", 1))


def _prepare_features(cache_bundle: Dict, hyperparams: Dict) -> Tuple[torch.Tensor, torch.Tensor]:
    features = cache_bundle["features"]
    labels = cache_bundle["labels"]
    matrix = build_feature_matrix(
        features,
        hyperparams.get("layers"),
        hyperparams.get("pool", "last"),
        hyperparams.get("candidate_mode", "diff"),
        hyperparams.get("normalize", False),
    )
    return matrix, labels


class LinearProbe(nn.Module):
    def __init__(self, in_dim: int):
        super().__init__()
        self.linear = nn.Linear(in_dim, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x).squeeze(-1)


class MLPProbe(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int, depth: int, dropout: float):
        super().__init__()
        layers: List[nn.Module] = []
        current_dim = in_dim
        for _ in range(max(depth - 1, 0)):
            layers.append(nn.Linear(current_dim, hidden_dim))
            layers.append(nn.GELU())
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            current_dim = hidden_dim
        layers.append(nn.Linear(current_dim, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


def _train_probe(
    model: nn.Module,
    train_x: torch.Tensor,
    train_y: torch.Tensor,
    eval_x: torch.Tensor,
    eval_y: torch.Tensor,
    hyperparams: Dict,
) -> Dict:
    batch_size = int(hyperparams.get("batch_size", 64))
    epochs = int(hyperparams.get("epochs", 5))
    lr = float(hyperparams.get("lr", 1e-3))
    weight_decay = float(hyperparams.get("weight_decay", 0.0))

    dataset = TensorDataset(train_x, train_y.float())
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    model.train()
    for _ in range(epochs):
        for xb, yb in loader:
            optimizer.zero_grad()
            logits = model(xb)
            loss = criterion(logits, yb)
            loss.backward()
            optimizer.step()

    model.eval()
    with torch.no_grad():
        train_logits = model(train_x)
        eval_logits = model(eval_x)

    def _metrics(logits: torch.Tensor, labels: torch.Tensor) -> Dict[str, float]:
        probs = torch.sigmoid(logits)
        preds = (probs >= 0.5).long()
        acc = (preds == labels).float().mean().item()
        return {
            "accuracy": acc,
        }

    return {
        "train": _metrics(train_logits, train_y.long()),
        "eval": _metrics(eval_logits, eval_y.long()),
    }


def _run_single_job(
    probe_type: str,
    hyperparams: Dict,
    train_cache: Dict,
    eval_cache: Dict,
) -> Dict:
    train_x, train_y = _prepare_features(train_cache, hyperparams)
    eval_x, eval_y = _prepare_features(eval_cache, hyperparams)

    input_dim = train_x.shape[1]
    if probe_type == "linear":
        model = LinearProbe(input_dim)
    elif probe_type == "mlp":
        hidden_dim = int(hyperparams.get("hidden_dim", 256))
        depth = int(hyperparams.get("depth", 2))
        dropout = float(hyperparams.get("dropout", 0.0))
        model = MLPProbe(input_dim, hidden_dim, depth, dropout)
    else:
        raise ValueError(f"Unknown probe type '{probe_type}'")

    metrics = _train_probe(model, train_x, train_y, eval_x, eval_y, hyperparams)
    return {
        "probe_type": probe_type,
        "hyperparams": hyperparams,
        "metrics": metrics,
    }


def _grid_from_sweeps(sweeps: Dict[str, Iterable]) -> List[Dict]:
    keys = sorted(sweeps.keys())
    values = []
    for key in keys:
        val = sweeps[key]
        if isinstance(val, list):
            values.append(val)
        else:
            values.append([val])
    combos = []
    for prod in itertools.product(*values):
        combos.append({k: v for k, v in zip(keys, prod)})
    return combos


def main():
    parser = argparse.ArgumentParser()
    parser = parser_add(parser)
    parser.add_argument("--probe_train_split", type=str, default="train")
    parser.add_argument("--probe_eval_split", type=str, default="test")
    parser.add_argument("--probe_types", type=str, default="linear,mlp")
    parser.add_argument("--cache_dir", type=str, default=None)
    args = parser.parse_args()
    params = params_add(args)
    probing_cfg = params.setdefault("probing", {})

    if args.cache_dir:
        probing_cfg["cache_dir"] = args.cache_dir
    cache_root = probing_cfg.get("cache_dir", "results/probing_cache")
    task = params["task"]
    dataset = params["dataset"]
    model_name = params["baseline"]["model"]
    train_split = args.probe_train_split
    eval_split = args.probe_eval_split
    probe_types = [p.strip() for p in args.probe_types.split(",") if p.strip()]

    train_cache = load_cache_split(cache_root, task, dataset, model_name, train_split)
    eval_cache = load_cache_split(cache_root, task, dataset, model_name, eval_split)

    result_dir = (
        Path(cache_root)
        / task
        / dataset
        / model_name.replace("/", "_")
        / "probe_runs"
    )
    result_dir.mkdir(parents=True, exist_ok=True)

    sweeps = probing_cfg.get("sweeps", {})
    workers = _default_parallel_workers(probing_cfg)

    for probe in probe_types:
        probe_sweeps = sweeps.get(probe, {})
        if not probe_sweeps:
            print(f"[probing] No sweep configuration found for probe '{probe}'. Skipping.")
            continue
        grid = _grid_from_sweeps(probe_sweeps)
        jobs = [
            (probe, combo, train_cache, eval_cache)
            for combo in grid
        ]
        results = []
        with ProcessPoolExecutor(max_workers=min(workers, len(jobs))) as executor:
            future_to_combo = {
                executor.submit(_run_single_job, *job): job[1] for job in jobs
            }
            for future in as_completed(future_to_combo):
                result = future.result()
                results.append(result)
                combo = future_to_combo[future]
                eval_acc = result["metrics"]["eval"]["accuracy"]
                print(f"[{probe}] combo={combo} eval_acc={eval_acc:.4f}")

        results_sorted = sorted(
            results, key=lambda r: r["metrics"]["eval"]["accuracy"], reverse=True
        )
        out_path = result_dir / f"{probe}_{train_split}_to_{eval_split}.json"
        with open(out_path, "w") as f:
            json.dump(results_sorted, f, indent=2)
        print(
            f"[probing] Saved {len(results_sorted)} {probe} runs to {out_path}."
            f" Best eval acc={results_sorted[0]['metrics']['eval']['accuracy']:.4f}"
        )


if __name__ == "__main__":
    main()
