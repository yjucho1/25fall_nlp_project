from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch


@dataclass
class HiddenStateRecord:
    """
    Container for a single example worth of cached hidden states.
    """

    hidden_states: torch.Tensor  # shape: (num_candidates, num_layers, hidden_dim)
    label: int
    set_size: int
    prompt: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    sample_id: Optional[int] = None

    def to_cpu(self):
        self.hidden_states = self.hidden_states.detach().to("cpu", torch.float32)
        return self


class HiddenStateCacheWriter:
    """
    Accumulates hidden state records and stores them to disk as a single torch file.
    """

    def __init__(
        self,
        cache_dir: str,
        task: str,
        dataset: str,
        model_name: str,
        split: str,
        raw_name: str,
        candidate_answers: List[str],
    ) -> None:
        self.records: List[HiddenStateRecord] = []
        self.candidate_answers = candidate_answers
        safe_model = model_name.replace("/", "_")
        self.output_path = (
            Path(cache_dir)
            / task
            / dataset
            / safe_model
            / split
            / f"{raw_name}.pt"
        )
        self.output_path.parent.mkdir(parents=True, exist_ok=True)

    def add_record(self, record: HiddenStateRecord) -> None:
        if record.hidden_states.shape[0] != len(self.candidate_answers):
            raise ValueError(
                f"Hidden state count ({record.hidden_states.shape[0]})"
                f" does not match candidate list ({len(self.candidate_answers)})."
            )
        self.records.append(record.to_cpu())

    def flush(self) -> Path:
        if not self.records:
            return self.output_path

        features = torch.stack([rec.hidden_states for rec in self.records], dim=0)
        labels = torch.tensor([rec.label for rec in self.records], dtype=torch.long)
        set_sizes = torch.tensor(
            [rec.set_size for rec in self.records], dtype=torch.long
        )
        sample_ids = torch.tensor(
            [rec.sample_id if rec.sample_id is not None else -1 for rec in self.records],
            dtype=torch.long,
        )
        prompts = [rec.prompt for rec in self.records]
        metadata = [rec.metadata for rec in self.records]

        payload = {
            "candidate_answers": self.candidate_answers,
            "hidden_states": features,
            "labels": labels,
            "set_sizes": set_sizes,
            "sample_ids": sample_ids,
            "prompts": prompts,
            "metadata": metadata,
        }
        torch.save(payload, self.output_path)
        return self.output_path
