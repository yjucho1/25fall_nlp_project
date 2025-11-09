from pathlib import Path
from typing import Dict, List

import torch


def _safe_model_name(model_name: str) -> str:
    return model_name.replace("/", "_")


def discover_cache_files(
    cache_root: str, task: str, dataset: str, model_name: str, split: str
) -> List[Path]:
    base = (
        Path(cache_root)
        / task
        / dataset
        / _safe_model_name(model_name)
        / split
    )
    if not base.exists():
        raise FileNotFoundError(f"No cache directory found at {base}")
    return sorted(base.glob("*.pt"))


def load_cache_split(
    cache_root: str, task: str, dataset: str, model_name: str, split: str
) -> Dict[str, object]:
    files = discover_cache_files(cache_root, task, dataset, model_name, split)
    hidden_states = []
    labels = []
    set_sizes = []
    prompts: List[str] = []
    metadata: List[Dict] = []
    sample_ids = []
    candidate_answers = None

    for path in files:
        payload = torch.load(path, map_location="cpu")
        candidate_answers = payload["candidate_answers"]
        hidden_states.append(payload["hidden_states"])
        labels.append(payload["labels"])
        set_sizes.append(payload["set_sizes"])
        sample_ids.append(payload["sample_ids"])
        prompts.extend(payload["prompts"])
        metadata.extend(payload["metadata"])

    if not hidden_states:
        raise RuntimeError(
            f"No cached hidden states loaded from {cache_root} for split={split}"
        )

    features = torch.cat(hidden_states, dim=0).to(torch.float32)
    labels = torch.cat(labels, dim=0).long()
    set_sizes = torch.cat(set_sizes, dim=0).long()
    sample_ids = torch.cat(sample_ids, dim=0).long()

    return {
        "candidate_answers": candidate_answers,
        "features": features,
        "labels": labels,
        "set_sizes": set_sizes,
        "prompts": prompts,
        "metadata": metadata,
        "sample_ids": sample_ids,
        "split": split,
    }
