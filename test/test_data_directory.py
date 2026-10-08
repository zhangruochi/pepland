"""Data-directory contracts; artifact tests isolate loader routing, not checkpoints."""
import ast
import os
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest

from model import data

ROOT = Path(__file__).resolve().parents[1]
ARTIFACTS = sorted((ROOT / "cpkt").glob("*/code/model/data.py"))
ARTIFACTS += sorted((ROOT / "inference/cpkt").glob("*/code/model/data.py"))


def config(value=None, missing=False, mode="pretrain"):
    train = SimpleNamespace(fragment="258", model=mode)
    if not missing:
        train.data_dir = value
    return SimpleNamespace(train=train)


def write_splits(directory):
    for split, smiles in (("train", "CC"), ("valid", "CCC"), ("test", "CCCC")):
        path = directory / "example" / (split + ".csv")
        path.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame({"smiles": [smiles]}).to_csv(path, index=False)


@pytest.mark.parametrize("kind", ["absolute", "relative", "default", "null", "missing"])
def test_real_root_loaders_read_configured_splits(tmp_path, monkeypatch, kind):
    # Build a repository-shaped root so defaults can be tested without touching
    # shipped datasets. Run from elsewhere to check stable relative resolution.
    repo = tmp_path / "repo"
    (repo / "tokenizer").mkdir(parents=True)
    (repo / "configs").mkdir()
    (repo / "model").mkdir()
    monkeypatch.setattr(data, "__file__", str(repo / "model" / "data.py"))
    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()
    monkeypatch.chdir(elsewhere)
    directory = repo / ("custom" if kind in ("absolute", "relative") else "data")
    write_splits(directory)
    values = {"absolute": str(directory), "relative": "custom", "default": "data",
              "null": None, "missing": None}
    loaders = data.make_loaders(config(values[kind], missing=kind == "missing"),
                                False, "example", batch_size=1)
    for split, atoms in (("train", 2), ("valid", 3), ("test", 4)):
        graphs = list(loaders[split])
        assert len(graphs) == 1
        assert graphs[0].num_nodes("a") == atoms


def test_real_root_loaders_missing_split_is_reported(tmp_path):
    with pytest.raises(FileNotFoundError):
        data.make_loaders(config(str(tmp_path)), False, "absent")


@pytest.mark.parametrize("value,error", [(False, TypeError), (123, TypeError),
                                        ([], TypeError), ("", ValueError),
                                        ("  ", ValueError), (b"data", TypeError)])
def test_invalid_data_directory(value, error):
    with pytest.raises(error, match="train.data_dir"):
        data.make_loaders(config(value), False, "example")


def artifact_loader(path, module_file):
    # Unit contract only: execute unchanged function bodies with CSV-backed
    # create_dataset capture and lightweight loader objects. No model is loaded.
    tree = ast.parse(path.read_text())
    nodes = [node for node in tree.body if isinstance(node, ast.FunctionDef)
             and node.name in ("_data_directory", "make_loaders")]
    calls = []

    def create_dataset(csv_path, transform):
        calls.append((csv_path, transform))
        return pd.read_csv(csv_path)

    namespace = {"os": os, "__file__": str(module_file),
                 "create_dataset": create_dataset,
                 "GraphDataLoader": lambda dataset, **kwargs: dataset}
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(path), "exec"), namespace)
    return namespace["make_loaders"], calls


@pytest.mark.parametrize("path", ARTIFACTS, ids=lambda p: str(p.relative_to(ROOT)))
@pytest.mark.parametrize("mode", ["fine-tune", "pretrain"])
def test_artifact_routing_preserves_all_splits(path, mode, tmp_path, monkeypatch):
    directory = tmp_path / "custom"
    write_splits(directory)
    loader, calls = artifact_loader(path, path)
    marker = object()
    loaded = loader(config(str(directory), mode=mode), False, "example", transform=marker)
    assert set(loaded) == {"train", "valid", "test"}
    assert loaded["train"]["smiles"].tolist() == ["CC"]
    assert loaded["valid"]["smiles"].tolist() == ["CCC"]
    assert loaded["test"]["smiles"].tolist() == ["CCCC"]
    assert [Path(csv).name for csv, _ in calls] == ["train.csv", "test.csv", "valid.csv"]
    assert all(transform is marker for _, transform in calls)


@pytest.mark.parametrize("path", ARTIFACTS, ids=lambda p: str(p.relative_to(ROOT)))
@pytest.mark.parametrize("kind", ["relative", "null", "missing"])
def test_exported_artifact_routing_uses_cwd(path, kind, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    directory = tmp_path / ("custom" if kind == "relative" else "data")
    write_splits(directory)
    loader, _ = artifact_loader(path, tmp_path / "export" / "model" / "data.py")
    loaded = loader(config("custom" if kind == "relative" else None,
                           missing=kind == "missing"), False, "example")
    assert loaded["train"]["smiles"].tolist() == ["CC"]


def test_repository_default_and_pathlike():
    assert data._data_directory(config(missing=True)) == str(ROOT / "data")
    assert data._data_directory(config(Path("custom"))) == str(ROOT / "custom")


@pytest.mark.parametrize("path", ARTIFACTS, ids=lambda p: str(p.relative_to(ROOT)))
def test_artifact_repository_relative_paths(path, tmp_path, monkeypatch):
    # The supplied module location is nested like each checked-in artifact.
    repo = tmp_path / "repo"
    (repo / "tokenizer").mkdir(parents=True)
    (repo / "configs").mkdir()
    write_splits(repo / "custom")
    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()
    monkeypatch.chdir(elsewhere)
    loader, calls = artifact_loader(path, repo / path.relative_to(ROOT))
    loaded = loader(config("custom"), False, "example")
    assert loaded["valid"]["smiles"].tolist() == ["CCC"]
    assert all(Path(csv).parent == repo / "custom" / "example" for csv, _ in calls)
