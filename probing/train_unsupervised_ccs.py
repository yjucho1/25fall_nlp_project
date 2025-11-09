import argparse
import itertools
import json
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, Iterable, Tuple

import torch
from torch import nn

from probing.cache_dataset import load_cache_split
from probing.feature_ops import select_layers
from utils import parser_add, params_add


def _grid_from_sweeps(sweeps: Dict[str, Iterable]) -> Iterable[Dict]:
    keys = sorted(sweeps.keys())
    value_lists = []
    for key in keys:
        val = sweeps[key]
        if isinstance(val, list):
            value_lists.append(val)
        else:
            value_lists.append([val])
    for combo in itertools.product(*value_lists):
        yield {k: v for k, v in zip(keys, combo)}


def _prepare_pairs(cache_bundle: Dict, hyperparams: Dict) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    features = cache_bundle["features"]
    labels = cache_bundle["labels"]
    pooled = select_layers(
        features,
        hyperparams.get("layers"),
        hyperparams.get("pool", "last"),
    )
    pos_idx = int(hyperparams.get("pos_index", 0))
    neg_idx = int(hyperparams.get("neg_index", 1))
    pos = pooled[:, pos_idx, :]
    neg = pooled[:, neg_idx, :]

    if hyperparams.get("center", True):
        mean = torch.cat([pos, neg], dim=0).mean(dim=0, keepdim=True)
        pos = pos - mean
        neg = neg - mean
    if hyperparams.get("normalize", False):
        pos = torch.nn.functional.normalize(pos, dim=-1)
        neg = torch.nn.functional.normalize(neg, dim=-1)

    return pos, neg, labels


def _binary_entropy(probs: torch.Tensor) -> torch.Tensor:
    eps = 1e-8
    return -(probs * torch.log(probs + eps) + (1 - probs) * torch.log(1 - probs + eps))


class CCSDirection(nn.Module):
    def __init__(self, in_dim: int):
        super().__init__()
        self.linear = nn.Linear(in_dim, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x).squeeze(-1)


def _train_ccs(
    pos_features: torch.Tensor,
    neg_features: torch.Tensor,
    hyperparams: Dict,
) -> CCSDirection:
    steps = int(hyperparams.get("steps", 2000))
    lr = float(hyperparams.get("lr", 0.1))
    temperature = float(hyperparams.get("temperature", 1.0))
    consistency_w = float(hyperparams.get("consistency_weight", 1.0))
    confidence_w = float(hyperparams.get("confidence_weight", 0.1))
    balance_w = float(hyperparams.get("balance_weight", 0.1))

    model = CCSDirection(pos_features.shape[1])
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    for _ in range(steps):
        optimizer.zero_grad()
        pos_logits = model(pos_features) / temperature
        neg_logits = model(neg_features) / temperature
        p_pos = torch.sigmoid(pos_logits)
        p_neg = torch.sigmoid(neg_logits)

        consistency = ((p_pos + p_neg - 1.0) ** 2).mean()
        confidence = (_binary_entropy(p_pos) + _binary_entropy(p_neg)).mean()
        balance = (p_pos.mean() - 0.5) ** 2 + (p_neg.mean() - 0.5) ** 2
        loss = (
            consistency_w * consistency
            + confidence_w * confidence
            + balance_w * balance
        )
        loss.backward()
        optimizer.step()

    return model


def _evaluate(model: CCSDirection, pos_features: torch.Tensor, labels: torch.Tensor) -> Dict[str, float]:
    with torch.no_grad():
        probs = torch.sigmoid(model(pos_features))
        preds = (probs < 0.5).long()
        acc = (preds == labels).float().mean().item()
    return {"accuracy": acc}


def _run_single_job(
    hyperparams: Dict,
    train_cache: Dict,
    eval_cache: Dict,
) -> Dict:
    train_pos, train_neg, train_labels = _prepare_pairs(train_cache, hyperparams)
    eval_pos, _, eval_labels = _prepare_pairs(eval_cache, hyperparams)
    model = _train_ccs(train_pos, train_neg, hyperparams)
    train_metrics = _evaluate(model, train_pos, train_labels)
    eval_metrics = _evaluate(model, eval_pos, eval_labels)
    return {
        "hyperparams": hyperparams,
        "metrics": {
            "train": train_metrics,
            "eval": eval_metrics,
        },
    }


def main():
    parser = argparse.ArgumentParser()
    parser = parser_add(parser)
    parser.add_argument("--probe_train_split", type=str, default="train")
    parser.add_argument("--probe_eval_split", type=str, default="test")
    args = parser.parse_args()
    params = params_add(args)
    probing_cfg = params.setdefault("probing", {})

    cache_root = probing_cfg.get("cache_dir", "results/probing_cache")
    task = params["task"]
    dataset = params["dataset"]
    model_name = params["baseline"]["model"]
    train_split = args.probe_train_split
    eval_split = args.probe_eval_split

    train_cache = load_cache_split(cache_root, task, dataset, model_name, train_split)
    eval_cache = load_cache_split(cache_root, task, dataset, model_name, eval_split)

    sweeps = probing_cfg.get("sweeps", {}).get("ccs", {})
    if not sweeps:
        raise ValueError("No CCS sweep configuration found under probing.sweeps.ccs")

    result_dir = (
        Path(cache_root)
        / task
        / dataset
        / model_name.replace("/", "_")
        / "probe_runs"
    )
    result_dir.mkdir(parents=True, exist_ok=True)

    workers = int(probing_cfg.get("parallel_workers", 1))
    jobs = list(_grid_from_sweeps(sweeps))
    results = []
    with ProcessPoolExecutor(max_workers=min(workers, len(jobs))) as executor:
        future_map = {
            executor.submit(_run_single_job, job, train_cache, eval_cache): job
            for job in jobs
        }
        for future in as_completed(future_map):
            result = future.result()
            results.append(result)
            combo = future_map[future]
            eval_acc = result["metrics"]["eval"]["accuracy"]
            print(f"[CCS] combo={combo} eval_acc={eval_acc:.4f}")

    results_sorted = sorted(
        results, key=lambda r: r["metrics"]["eval"]["accuracy"], reverse=True
    )
    out_path = result_dir / f"ccs_{train_split}_to_{eval_split}.json"
    with open(out_path, "w") as f:
        json.dump(results_sorted, f, indent=2)
    print(
        f"[probing] Saved {len(results_sorted)} CCS runs to {out_path}. "
        f"Best eval acc={results_sorted[0]['metrics']['eval']['accuracy']:.4f}"
    )


if __name__ == "__main__":
    main()
