"""Frozen PepLand embeddings with a molecule-disjoint Ridge regression example.

This is a usage example, not a reproduction of a published downstream benchmark.
"""
from __future__ import annotations

import argparse
import contextlib
import csv
from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
import random
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]


@dataclass(frozen=True)
class Sample:
    smiles: str
    target: float
    group: str = ""


def load_samples(path, smiles_column, target_column, group_column=None,
                 drop_missing_target=False):
    """Read finite scalar targets, canonicalize molecules, and collapse exact duplicates."""
    from rdkit import Chem, rdBase

    samples = set()
    stats = {"input_rows": 0, "dropped_missing_targets": 0, "exact_duplicates": 0}
    with Path(path).open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        fields = reader.fieldnames or []
        required = [smiles_column, target_column] + ([group_column] if group_column else [])
        if len(fields) != len(set(fields)) or any(c not in fields for c in required):
            raise ValueError("CSV requires unique headers and the selected columns")
        for number, row in enumerate(reader, 2):
            stats["input_rows"] += 1
            value = (row.get(target_column) or "").strip()
            if value.lower() in ("", "na", "n/a", "none", "null"):
                if drop_missing_target:
                    stats["dropped_missing_targets"] += 1
                    continue
                raise ValueError(f"Missing target at CSV row {number}; opt in to dropping missing targets")
            try:
                target = float(value)
            except ValueError as error:
                raise ValueError(f"Non-numeric target at CSV row {number}") from error
            if not math.isfinite(target):
                raise ValueError(f"Non-finite target at CSV row {number}")
            group = (row.get(group_column) or "").strip() if group_column else ""
            if group_column and not group:
                raise ValueError(f"Missing group at CSV row {number}")
            with rdBase.BlockLogs():
                mol = Chem.MolFromSmiles((row.get(smiles_column) or "").strip())
            if mol is None or mol.GetNumAtoms() == 0:
                raise ValueError(f"Invalid molecule at CSV row {number}")
            canonical = Chem.MolToSmiles(mol, canonical=True, isomericSmiles=True)
            sample = Sample(canonical, target, group)
            if sample in samples:
                stats["exact_duplicates"] += 1
            samples.add(sample)
    if not samples:
        raise ValueError("No usable labeled molecules")
    return sorted(samples, key=lambda s: (s.smiles, s.group, s.target)), stats


def molecule_components(samples):
    """Link both molecular identity and optional group identity transitively."""
    parent = list(range(len(samples)))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    def union(a, b):
        parent[find(b)] = find(a)

    molecules, groups = {}, {}
    for i, sample in enumerate(samples):
        if sample.smiles in molecules:
            union(i, molecules[sample.smiles])
        else:
            molecules[sample.smiles] = i
        if sample.group:
            if sample.group in groups:
                union(i, groups[sample.group])
            else:
                groups[sample.group] = i
    components = {}
    for i, sample in enumerate(samples):
        components.setdefault(find(i), []).append(sample)
    return sorted(
        [sorted(c, key=lambda s: (s.smiles, s.group, s.target))
         for c in components.values()],
        key=lambda c: min(s.smiles for s in c),
    )


def assert_disjoint(partitions):
    """Reject molecular or optional-group overlap; never relax this for small data."""
    names = ("train", "valid", "test")
    molecules, groups = {}, {}
    for name in names:
        if not partitions[name]:
            raise ValueError("Each split must be nonempty")
        molecules[name] = {s.smiles for s in partitions[name]}
        groups[name] = {s.group for s in partitions[name] if s.group}
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            if molecules[a] & molecules[b] or groups[a] & groups[b]:
                raise ValueError(f"{a}/{b} overlap in molecules or groups")


def split_samples(samples, seed=0, valid_fraction=0.2, test_fraction=0.2,
                  max_molecules=None):
    if not isinstance(seed, int) or isinstance(seed, bool) or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    if not all(math.isfinite(v) and 0 < v < 1
               for v in (valid_fraction, test_fraction)) or valid_fraction + test_fraction >= 1:
        raise ValueError("Split fractions must be positive and leave a training fraction")
    if max_molecules is not None and (isinstance(max_molecules, bool)
            or not isinstance(max_molecules, int) or max_molecules <= 0):
        raise ValueError("max_molecules must be a positive integer")
    components = molecule_components(samples)
    random.Random(seed).shuffle(components)
    if max_molecules is not None:
        selected, count = [], 0
        for component in components:
            size = len({s.smiles for s in component})
            if count + size <= max_molecules:
                selected.append(component)
                count += size
        components = selected
    n = len(components)
    n_valid = max(1, round(n * valid_fraction))
    n_test = max(1, round(n * test_fraction))
    if n - n_valid - n_test < 2:
        raise ValueError("Need enough independent components for two training components and nonempty validation/test")
    slices = {
        "test": components[:n_test],
        "valid": components[n_test:n_test + n_valid],
        "train": components[n_test + n_valid:],
    }
    partitions = {name: sorted([s for c in cs for s in c],
                              key=lambda s: (s.smiles, s.group, s.target))
                  for name, cs in slices.items()}
    assert_disjoint(partitions)
    return partitions


def _matrix(values):
    values = np.asarray(values, dtype=np.float64)
    if values.ndim != 2 or not all(values.shape) or not np.isfinite(values).all():
        raise ValueError("Features must be a nonempty finite matrix")
    return values


@dataclass
class RidgeModel:
    mean: np.ndarray
    scale: np.ndarray
    coefficient: np.ndarray
    intercept: float
    settings: dict

    def predict(self, features):
        features = _matrix(features)
        if features.shape[1] != len(self.mean):
            raise ValueError("Feature dimension differs from the saved model")
        prediction = ((features - self.mean) / self.scale) @ self.coefficient + self.intercept
        if not np.isfinite(prediction).all():
            raise ValueError("Non-finite predictions")
        return prediction

    def save(self, path):
        np.savez_compressed(path, mean=self.mean, scale=self.scale,
                            coefficient=self.coefficient, intercept=self.intercept,
                            metadata=json.dumps(self.settings, sort_keys=True))

    @classmethod
    def load(cls, path, expected_checkpoint_sha256=None):
        with np.load(path, allow_pickle=False) as saved:
            settings = json.loads(str(saved["metadata"]))
            mean, scale, coefficient = (saved[k].copy() for k in
                                        ("mean", "scale", "coefficient"))
            intercept = float(saved["intercept"])
        if not isinstance(settings, dict):
            raise ValueError("Invalid model metadata")
        if expected_checkpoint_sha256 is not None and settings.get("checkpoint_sha256") != expected_checkpoint_sha256:
            raise ValueError("Checkpoint differs from the saved extraction configuration")
        if settings.get("format_version") != 1 or settings.get("pooling") != "avg" \
                or settings.get("padding_mode") != "exclude":
            raise ValueError("Unsupported model format or extraction settings")
        dimension = settings.get("feature_dim")
        if not isinstance(dimension, int) or isinstance(dimension, bool) or dimension <= 0:
            raise ValueError("Invalid feature dimension")
        if any(a.ndim != 1 or len(a) != dimension or not np.isfinite(a).all()
               for a in (mean, scale, coefficient)) or not dimension \
                or np.any(scale <= 0) or not math.isfinite(intercept):
            raise ValueError("Invalid saved regression parameters")
        return cls(mean, scale, coefficient, intercept, settings)


def fit_regression(features, targets, alpha=1.0, settings=None):
    """Fit the scaler and fixed-alpha head on training data only."""
    from sklearn.linear_model import Ridge

    features = _matrix(features)
    targets = np.asarray(targets, dtype=np.float64)
    if targets.shape != (len(features),) or not np.isfinite(targets).all():
        raise ValueError("Targets must be a finite scalar per training row")
    if not math.isfinite(alpha) or alpha <= 0:
        raise ValueError("alpha must be finite and positive")
    mean, scale = features.mean(axis=0), features.std(axis=0)
    scale[scale == 0] = 1
    ridge = Ridge(alpha=alpha, solver="svd").fit((features - mean) / scale, targets)
    metadata = dict(settings or {})
    metadata.update({"format_version": 1, "feature_dim": features.shape[1],
                     "alpha": alpha, "pooling": "avg", "padding_mode": "exclude"})
    model = RidgeModel(mean, scale, ridge.coef_.copy(), float(ridge.intercept_), metadata)
    model.predict(features)
    return model


def extract_embeddings(samples, extractor, batch_size=8):
    """Only CPU eval/no-grad inference; the head never trains the backbone."""
    import torch

    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    extractor.to("cpu").eval()
    for parameter in extractor.parameters():
        parameter.requires_grad_(False)
    batches = []
    with torch.no_grad():
        for start in range(0, len(samples), batch_size):
            tensor = extractor([s.smiles for s in samples[start:start + batch_size]])
            if not isinstance(tensor, torch.Tensor) or tensor.device.type != "cpu":
                raise ValueError("Extractor must return a CPU tensor")
            values = _matrix(tensor.detach().numpy())
            if len(values) != len(samples[start:start + batch_size]):
                raise ValueError("Extractor returned an incorrect number of rows")
            batches.append(values)
    if not batches:
        raise ValueError("Cannot extract an empty dataset")
    return _matrix(np.concatenate(batches))


def build_extractor(model_path):
    # Match the package import convention documented by the repository Python API.
    sys.path.insert(0, str(ROOT.parent))
    from pepland.model.core import PepLandFeatureExtractor

    with contextlib.redirect_stdout(sys.stderr):
        return PepLandFeatureExtractor(str(model_path), pooling="avg",
                                       freeze=True, padding_mode="exclude")


def file_digest(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def metrics(target, prediction):
    target, prediction = np.asarray(target), np.asarray(prediction)
    difference = target - prediction
    result = {"mae": float(np.abs(difference).mean()),
              "rmse": float(np.sqrt(np.square(difference).mean()))}
    if not all(math.isfinite(v) for v in result.values()):
        raise ValueError("Non-finite metrics")
    return result


def run(args, extractor=None):
    output = Path(args.output)
    if output.exists():
        raise FileExistsError("Output already exists; choose a new output directory")
    samples, data_stats = load_samples(args.csv, args.smiles_column, args.target_column,
                                      args.group_column, args.drop_missing_target)
    partitions = split_samples(samples, args.seed, args.valid_fraction,
                               args.test_fraction, args.max_molecules)
    settings = {"seed": args.seed, "smiles_column": args.smiles_column,
                "target_column": args.target_column, "group_column": args.group_column,
                "input_sha256": file_digest(args.csv),
                "valid_fraction": args.valid_fraction, "test_fraction": args.test_fraction,
                "max_molecules": args.max_molecules,
                "drop_missing_target": args.drop_missing_target, "batch_size": args.batch_size}
    if extractor is None:
        settings["checkpoint_sha256"] = file_digest(Path(args.checkpoint) / "data/model.pth")
        extractor = build_extractor(args.checkpoint)
    else:
        settings["checkpoint_sha256"] = None  # Injected extractors are unit-test-only.
    features = {name: extract_embeddings(part, extractor, args.batch_size)
                for name, part in partitions.items()}
    targets = {name: np.array([s.target for s in part]) for name, part in partitions.items()}
    model = fit_regression(features["train"], targets["train"], args.alpha, settings)
    baseline = float(targets["train"].mean())
    report = {
        "schema": data_stats,
        "seed": args.seed, "alpha": args.alpha,
        "subset_requested": args.max_molecules,
        "selected_rows": sum(map(len, partitions.values())),
        "selected_molecules": len({s.smiles for part in partitions.values() for s in part}),
        "group_column": args.group_column,
        "molecule_and_supplied_group_disjoint": True,
        "pooling": "avg", "padding_mode": "exclude",
        "scope": "usage demonstration; no paper reproduction or generalization claim",
        "splits": {name: {"rows": len(part), "molecules": len({s.smiles for s in part}),
                         "ridge": metrics(targets[name], model.predict(features[name])),
                         "train_mean_baseline": metrics(targets[name],
                                                      np.full(len(part), baseline))}
                   for name, part in partitions.items()},
    }
    output.mkdir(parents=True, exist_ok=False)
    model.save(output / "model.npz")
    reloaded = RidgeModel.load(output / "model.npz", settings["checkpoint_sha256"])
    for name in partitions:
        np.testing.assert_allclose(reloaded.predict(features[name]), model.predict(features[name]),
                                   rtol=0, atol=0)
    (output / "metrics.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    (output / "config.json").write_text(json.dumps(model.settings, indent=2, allow_nan=False) + "\n")
    return report


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--csv", required=True)
    p.add_argument("--smiles-column", default="smiles")
    p.add_argument("--target-column", required=True)
    p.add_argument("--group-column")
    p.add_argument("--drop-missing-target", action="store_true")
    p.add_argument("--max-molecules", type=int)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--valid-fraction", type=float, default=0.2)
    p.add_argument("--test-fraction", type=float, default=0.2)
    p.add_argument("--alpha", type=float, default=1.0)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--checkpoint", default=str(ROOT / "inference/cpkt/model"))
    p.add_argument("--output", default="outputs/frozen-regression")
    return p


def main():
    import torch

    torch.set_num_threads(1)
    p = parser()
    args = p.parse_args()
    try:
        report = run(args)
    except (ValueError, FileExistsError) as error:
        p.error(str(error))
    print(json.dumps(report, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
