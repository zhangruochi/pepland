"""Interface tests use synthetic targets/features; the opt-in test uses a real checkpoint."""
import csv
import json
import os
from pathlib import Path
import sys

import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from examples import frozen_regression as f

SMILES = ["NCC(=O)O", "CC(N)C(=O)O", "CC(C)C(N)C(=O)O",
          "CCC(C)C(N)C(=O)O", "CC(C)CC(N)C(=O)O",
          "NCCCC(N)C(=O)O", "O=C(O)C(N)CO", "O=C(O)C(N)CS",
          "NCC(=O)NCC(=O)O", "CC(N)C(=O)NCC(=O)O",
          "NCC(=O)NC(C)C(=O)O", "NCC(=O)NCC(=O)NCC(=O)O"]


def csv_file(tmp_path, rows, fields=("smiles", "target", "group"), name="data.csv"):
    path = tmp_path / name
    with path.open("w", newline="") as h:
        w = csv.DictWriter(h, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)
    return path


def labeled(tmp_path):
    return csv_file(tmp_path, [{"smiles":s, "target":i / 10, "group":str(i)}
                               for i, s in enumerate(SMILES)])


def arguments(path, output):
    return f.parser().parse_args(["--csv", str(path), "--target-column", "target",
                                 "--output", str(output), "--seed", "17"])


@pytest.mark.parametrize("target", ["NaN", "inf", "-inf", "one", "<1"])
def test_bad_targets_are_not_silently_dropped(tmp_path, target):
    path = csv_file(tmp_path, [{"smiles":"CCO", "target":target, "group":"a"}])
    with pytest.raises(ValueError, match="target"):
        f.load_samples(path, "smiles", "target", drop_missing_target=True)


@pytest.mark.parametrize("smiles", ["", "not-a-molecule", "[Na"])
def test_invalid_molecules(tmp_path, smiles):
    path = csv_file(tmp_path, [{"smiles":smiles, "target":1, "group":"a"}])
    with pytest.raises(ValueError, match="molecule"):
        f.load_samples(path, "smiles", "target")


def test_missing_target_opt_in_and_canonical_duplicate(tmp_path):
    path = csv_file(tmp_path, [{"smiles":"CCO", "target":1, "group":"a"},
                               {"smiles":"OCC", "target":1, "group":"a"},
                               {"smiles":"CN", "target":"", "group":"b"}])
    with pytest.raises(ValueError, match="Missing target"):
        f.load_samples(path, "smiles", "target")
    samples, stats = f.load_samples(path, "smiles", "target", "group", True)
    assert samples == [f.Sample("CCO", 1, "a")]
    assert stats == {"input_rows":3, "dropped_missing_targets":1, "exact_duplicates":1}


def test_missing_group_and_header_rejected(tmp_path):
    path = csv_file(tmp_path, [{"smiles":"CCO", "target":1, "group":""}])
    with pytest.raises(ValueError, match="Missing group"):
        f.load_samples(path, "smiles", "target", "group")
    with pytest.raises(ValueError, match="headers"):
        f.load_samples(path, "missing", "target")
    path.write_text("smiles,target,target\nCCO,1,2\n")
    with pytest.raises(ValueError, match="headers"):
        f.load_samples(path, "smiles", "target")


def test_empty_data_after_filter(tmp_path):
    path = csv_file(tmp_path, [{"smiles":"CCO", "target":"", "group":"a"}])
    with pytest.raises(ValueError, match="No usable"):
        f.load_samples(path, "smiles", "target", drop_missing_target=True)


def test_transitive_molecule_and_group_components():
    samples = [f.Sample("CCO",1,"a"), f.Sample("CCO",2,"b"),
               f.Sample("CN",3,"b"), f.Sample("CO",4,"c")]
    components = f.molecule_components(samples)
    assert sorted(map(len, components)) == [1,3]
    with pytest.raises(ValueError, match="independent components"):
        f.split_samples(samples)


def test_split_seed_order_identity_and_cap(tmp_path):
    samples, _ = f.load_samples(labeled(tmp_path), "smiles", "target", "group")
    expected = f.split_samples(samples, seed=17)
    assert expected == f.split_samples(list(reversed(samples)), seed=17)
    assert expected != f.split_samples(samples, seed=18)
    f.assert_disjoint(expected)
    limited = f.split_samples(samples, seed=17, max_molecules=8)
    assert sum(map(len, limited.values())) == 8
    # Targets cannot influence group assignment.
    changed = [f.Sample(s.smiles, s.target + 100, s.group) for s in samples]
    other = f.split_samples(changed, seed=17)
    assert {k:{s.smiles for s in v} for k,v in expected.items()} == {
        k:{s.smiles for s in v} for k,v in other.items()}


@pytest.mark.parametrize("kwargs", [{"seed":-1},{"seed":True},{"valid_fraction":float("nan")},
    {"test_fraction":0},{"valid_fraction":0.5,"test_fraction":0.5},{"max_molecules":0}])
def test_invalid_split_options(kwargs):
    with pytest.raises(ValueError):
        f.split_samples([f.Sample(s,i) for i,s in enumerate(SMILES)], **kwargs)


def test_shared_69_split_pattern_is_rejected():
    # Synthetic stand-in for the public archive's identical split membership.
    repeated = [f.Sample(str(i),float(i)) for i in range(69)]
    with pytest.raises(ValueError, match="overlap"):
        f.assert_disjoint({k:repeated for k in ("train","valid","test")})


def test_large_components_never_cut_to_fit_cap():
    samples = [f.Sample(s,i,"same") for i,s in enumerate(SMILES)]
    with pytest.raises(ValueError, match="independent components"):
        f.split_samples(samples, max_molecules=8)


def test_train_only_scaler_zero_variance_and_save(tmp_path):
    x = np.array([[0,3],[1,3],[2,3],[3,3]], dtype=float)
    y = np.array([-1,0,1,2], dtype=float)
    model = f.fit_regression(x,y,settings={"checkpoint_sha256":"fixture"})
    np.testing.assert_array_equal(model.mean,[1.5,3])
    assert model.scale[1] == 1
    # Held-out feature/label values never enter fitting.
    heldout = np.array([[1e6,-1e6]])
    assert np.isfinite(model.predict(heldout)).all()
    unchanged = f.fit_regression(x,y)
    np.testing.assert_array_equal(unchanged.coefficient,model.coefficient)
    path = tmp_path / "model.npz"
    model.save(path)
    restored = f.RidgeModel.load(path,expected_checkpoint_sha256="fixture")
    np.testing.assert_array_equal(restored.predict(x), model.predict(x))
    with pytest.raises(ValueError,match="Checkpoint"):
        f.RidgeModel.load(path,expected_checkpoint_sha256="other")
    with np.load(path,allow_pickle=False) as z:
        assert all(z[k].dtype.kind != "O" for k in z.files)


@pytest.mark.parametrize("x,y,alpha", [
    ([[float("nan")]], [1],1), ([[1],[2]], [1],1),
    ([[1],[2]], [1,float("inf")],1), ([[1]], [1],0)])
def test_bad_fit_inputs(x,y,alpha):
    with pytest.raises(ValueError):
        f.fit_regression(x,y,alpha)


class SyntheticExtractor(torch.nn.Module):
    """Deterministic fake features, explicitly not checkpoint validation."""
    def __init__(self, wrong_rows=False):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.ones(1))
        self.wrong_rows = wrong_rows

    def forward(self, smiles):
        assert not self.training and not torch.is_grad_enabled()
        assert not self.weight.requires_grad
        rows = [[len(s),s.count("N"),s.count("O"),1] for s in smiles]
        if self.wrong_rows:
            rows = rows[:1]
        return torch.tensor(rows,dtype=torch.float32)


def test_injected_interface_artifacts_and_heldout_invariance(tmp_path):
    path = labeled(tmp_path)
    args = arguments(path,tmp_path/"first")
    report = f.run(args,extractor=SyntheticExtractor())
    model = f.RidgeModel.load(tmp_path/"first/model.npz")
    partitions = f.split_samples(f.load_samples(path,"smiles","target")[0],seed=17)
    heldout = {s.smiles for k in ("valid","test") for s in partitions[k]}
    rows = [{"smiles":s.smiles,"target":s.target + (100 if s.smiles in heldout else 0),
             "group":s.group} for part in partitions.values() for s in part]
    changed = csv_file(tmp_path,rows,name="changed.csv")
    second_args = arguments(changed,tmp_path/"second")
    f.run(second_args,extractor=SyntheticExtractor())
    second = f.RidgeModel.load(tmp_path/"second/model.npz")
    np.testing.assert_array_equal(model.mean,second.mean)
    np.testing.assert_array_equal(model.scale,second.scale)
    np.testing.assert_array_equal(model.coefficient,second.coefficient)
    assert model.intercept == second.intercept
    assert report["molecule_and_supplied_group_disjoint"]
    assert model.settings["checkpoint_sha256"] is None
    assert json.loads((tmp_path/"first/metrics.json").read_text()) == report
    with pytest.raises(FileExistsError):
        f.run(args,extractor=SyntheticExtractor())


def test_wrong_extractor_rows_rejected():
    with pytest.raises(ValueError,match="incorrect number"):
        f.extract_embeddings([f.Sample("CC",1),f.Sample("CCC",2)],
                             SyntheticExtractor(wrong_rows=True))


@pytest.mark.skipif(os.environ.get("PEPLAND_CHECKPOINT_TESTS") != "1",
                    reason="set PEPLAND_CHECKPOINT_TESTS=1 for actual CPU checkpoint")
def test_real_checkpoint_frozen_eval_reload(tmp_path, monkeypatch):
    torch.set_num_threads(1)
    path = labeled(tmp_path)
    args = arguments(path,tmp_path/"actual")
    extractor = f.build_extractor(args.checkpoint)
    before = {name:value.detach().clone() for name,value in extractor.state_dict().items()}
    forward = extractor.forward
    def checked_forward(smiles):
        assert not extractor.training and not torch.is_grad_enabled()
        assert all(not p.requires_grad for p in extractor.parameters())
        return forward(smiles)
    extractor.forward = checked_forward
    # Capture the real model for immutability checks; run the normal checkpoint path.
    monkeypatch.setattr(f, "build_extractor", lambda path: extractor)
    report = f.run(args)
    assert not extractor.training and all(not p.requires_grad and p.grad is None
                                          for p in extractor.parameters())
    for name,value in extractor.state_dict().items():
        torch.testing.assert_close(value,before[name],rtol=0,atol=0)
    model = f.RidgeModel.load(tmp_path/"actual/model.npz")
    assert model.settings["feature_dim"] == 300
    assert model.settings["checkpoint_sha256"] == f.file_digest(Path(args.checkpoint)/"data/model.pth")
    assert all(np.isfinite(v["ridge"]["rmse"]) for v in report["splits"].values())
    samples,_ = f.load_samples(path,"smiles","target")
    single = f.extract_embeddings(samples,extractor,batch_size=1)
    mixed = f.extract_embeddings(samples,extractor,batch_size=4)
    np.testing.assert_allclose(single,mixed,rtol=2e-5,atol=2e-5)
    np.testing.assert_allclose(model.predict(single),model.predict(mixed),rtol=2e-5,atol=2e-5)
