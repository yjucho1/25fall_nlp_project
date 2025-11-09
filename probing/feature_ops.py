from typing import Iterable, List, Optional, Sequence

import torch


def _resolve_layers(num_layers: int, selectors: Optional[Sequence[int]]) -> List[int]:
    if not selectors:
        return list(range(num_layers))
    resolved = []
    for idx in selectors:
        resolved_idx = idx if idx >= 0 else num_layers + idx
        if resolved_idx < 0 or resolved_idx >= num_layers:
            raise ValueError(f"Layer index {idx} is out of range for {num_layers}")
        resolved.append(resolved_idx)
    return resolved


def select_layers(
    features: torch.Tensor, layer_indices: Optional[Sequence[int]], pool: str
) -> torch.Tensor:
    """
    Slice and pool layers.

    Args:
        features: tensor with shape (N, C, L, H)
        layer_indices: indices to select (supports negative indexing).
        pool: one of {'last', 'mean', 'max'}.
    """

    num_layers = features.shape[2]
    indices = _resolve_layers(num_layers, layer_indices)
    subset = features[:, :, indices, :]

    if pool == "last":
        return subset[:, :, -1, :]
    if pool == "mean":
        return subset.mean(dim=2)
    if pool == "max":
        return subset.max(dim=2).values
    raise ValueError(f"Unknown layer pooling strategy '{pool}'")


def build_feature_matrix(
    features: torch.Tensor,
    layer_indices: Optional[Iterable[int]],
    pool: str,
    candidate_mode: str,
    normalize: bool = False,
) -> torch.Tensor:
    """
    Convert cached hidden states into a 2-D matrix suitable for probes.

    Args:
        features: tensor with shape (N, C, L, H)
        layer_indices: subset of layers to use (None selects all).
        pool: pooling strategy over layers ("last", "mean", "max").
        candidate_mode: how to combine candidate answers.
            'diff' : h_consistent - h_inconsistent
            'concat' : concat([h_consistent, h_inconsistent])
            'first' : use the first candidate as-is
            'second' : use the second candidate as-is
        normalize: whether to L2-normalize the final vectors.
    """

    pooled = select_layers(features, layer_indices, pool)

    if candidate_mode == "diff":
        if pooled.shape[1] != 2:
            raise ValueError("Difference mode requires exactly two candidates.")
        matrix = pooled[:, 0, :] - pooled[:, 1, :]
    elif candidate_mode == "concat":
        matrix = torch.cat([pooled[:, i, :] for i in range(pooled.shape[1])], dim=-1)
    elif candidate_mode == "first":
        matrix = pooled[:, 0, :]
    elif candidate_mode == "second":
        matrix = pooled[:, 1, :]
    else:
        raise ValueError(f"Unsupported candidate_mode '{candidate_mode}'")

    if normalize:
        matrix = torch.nn.functional.normalize(matrix, dim=-1)

    return matrix
