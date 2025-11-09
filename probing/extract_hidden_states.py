import argparse
import time
from typing import List, Optional, Sequence, Set

import torch

from baselines.LLM.lm_loader import lm_loader
from baselines.baseline_model import baseline_model
from probing.cache import HiddenStateCacheWriter, HiddenStateRecord
from probing.dataset_utils import load_split_pickles_as_datasets, to_display_name
from utils import parser_add, params_add


def _parse_subset(text: Optional[str]) -> Optional[Set[str]]:
    if not text:
        return None
    return {token.strip() for token in text.split(",") if token.strip()}


def _default_probing_cfg(params: dict) -> dict:
    probing_cfg = params.setdefault("probing", {})
    probing_cfg.setdefault("cache_dir", "results/probing_cache")
    probing_cfg.setdefault("split", "test")
    probing_cfg.setdefault("answer_prefix", "\nConsistency: ")
    probing_cfg.setdefault("candidate_answers", ["consistent", "inconsistent"])
    probing_cfg.setdefault("max_examples", None)
    return probing_cfg


def main():
    parser = argparse.ArgumentParser()
    parser = parser_add(parser)
    parser.add_argument("--probe_split", type=str, default=None)
    parser.add_argument("--probe_subset", type=str, default=None)
    parser.add_argument("--probe_limit", type=int, default=None)
    args = parser.parse_args()
    params = params_add(args)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    params["device"] = device
    probing_cfg = _default_probing_cfg(params)

    split = args.probe_split or probing_cfg.get("split", "test")
    subset = _parse_subset(args.probe_subset or probing_cfg.get("subset"))
    limit = args.probe_limit or probing_cfg.get("max_examples")
    candidate_answers: Sequence[str] = probing_cfg["candidate_answers"]
    answer_prefix: str = probing_cfg["answer_prefix"]
    cache_dir: str = probing_cfg["cache_dir"]

    dataset_name = params["dataset"]
    task_name = params["task"]
    model_name = params["baseline"]["model"]

    raw_names, datasets = load_split_pickles_as_datasets(dataset_name, split)
    loaders = [
        lm_loader(ds, params=params).get_loader()
        for ds in datasets
    ]

    base_model = baseline_model(params, "prediction")
    llm_backend = base_model.model
    if not hasattr(llm_backend, "hidden_states_for_completions"):
        raise ValueError(
            "The selected baseline model does not expose hidden state extraction."
        )

    extracted = 0
    start_time = time.time()
    for raw_name, dataloader in zip(raw_names, loaders):
        if subset and raw_name not in subset:
            continue

        writer = HiddenStateCacheWriter(
            cache_dir=cache_dir,
            task=task_name,
            dataset=dataset_name,
            model_name=model_name,
            split=split,
            raw_name=raw_name,
            candidate_answers=list(candidate_answers),
        )

        for batch in dataloader:
            sample = batch[0]
            pairs: List[str] = sample[0]
            inconsistent_indices: List[int] = sample[1]
            label = 0 if len(inconsistent_indices) == 0 else 1
            set_size = len(pairs)

            bundle = llm_backend.hidden_states_for_completions(
                batch,
                completions=candidate_answers,
                answer_prefix=answer_prefix,
            )
            writer.add_record(
                HiddenStateRecord(
                    hidden_states=bundle["hidden_states"],
                    label=label,
                    set_size=set_size,
                    prompt=bundle["prompt"],
                    metadata={
                        "raw_name": raw_name,
                        "display_name": to_display_name(raw_name),
                        "inconsistent_indices": inconsistent_indices,
                        "set_size": set_size,
                    },
                    sample_id=extracted,
                )
            )
            extracted += 1
            if limit and extracted >= limit:
                break
        writer.flush()
        if limit and extracted >= limit:
            break

    elapsed = time.time() - start_time
    print(
        f"[probing] Cached hidden states for {extracted} examples "
        f"(split={split}) in {elapsed:.2f}s."
    )


if __name__ == "__main__":
    main()
