import glob
import pickle
from pathlib import Path
from typing import List, Tuple


def _pkl_path(dataset_name: str, split: str, name: str) -> Path:
    """
    Return the pickle path under set_consistency_dataset/{dataset_name}/
    following the naming scheme {dataset}_{split}_{NAME}_dataset.pickle.
    """
    base = Path("set_consistency_dataset") / dataset_name
    fname = f"{dataset_name}_{name}_dataset.pickle"
    return base / fname


def list_split_pickles(dataset_name: str, split: str) -> Tuple[List[str], List[Path]]:
    """
    Collect pickle files for the given split and return both the logical names
    (e.g., C, I, CI, CCC) and their paths, ordered so that C/I appear first.
    """
    base = Path("set_consistency_dataset") / dataset_name
    patt = str(base / f"{dataset_name}_{split}_*_dataset.pickle")
    paths = sorted(glob.glob(patt))

    names = []
    for p in paths:
        fn = Path(p).name
        prefix = f"{dataset_name}_{split}_"
        if not (fn.startswith(prefix) and fn.endswith("_dataset.pickle")):
            raise RuntimeError(f"Unexpected filename: {fn}")
        name = fn[len(prefix) : -len("_dataset.pickle")]
        names.append(name)

    def _order_key(x: str):
        if x == "C":
            return (0, 0, x)
        if x == "I":
            return (0, 1, x)
        return (1, len(x), x)

    names_sorted = sorted(names, key=_order_key)
    paths_sorted = [_pkl_path(dataset_name, split, f"{split}_{nm}") for nm in names_sorted]
    return names_sorted, paths_sorted


def load_split_pickles_as_datasets(dataset_name: str, split: str):
    """
    Load the pickle datasets for the provided split. Returns (names, datasets).
    """
    names, paths = list_split_pickles(dataset_name, split)
    datasets = []
    for nm, p in zip(names, paths):
        if not p.exists():
            raise FileNotFoundError(f"Missing pickle: {p}")
        with open(p, "rb") as f:
            datasets.append(pickle.load(f))
    return names, datasets


def to_display_name(name: str) -> str:
    """
    Map raw names to human-friendly tokens ("C"->"con", "I"->"incon").
    """
    return "con" if name == "C" else ("incon" if name == "I" else name)
