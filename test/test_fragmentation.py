"""RDKit regression tests for repeated peptide bond cuts (issue #8)."""
import importlib.util
import os
from pathlib import Path
import sys

import pytest
from rdkit import Chem
import torch

ROOT = Path(__file__).resolve().parents[1]
REPORTED_SMILES = 'CC(=O)NC(CSC1CC(=O)N(CCCC(=O)NCCOCCNC(=O)CCCCCNC(=O)CCCCC2SCC3NC(=O)NC32)C1=O)C(=O)NC(CCCCN)C(=O)NC(CCCNC(=N)N)C(=O)NC(CCCNC(=N)N)C(=O)NC(CCCNC(=N)N)C(=O)NC(CCC(N)=O)C(=O)NC(CCCNC(=N)N)C(=O)NC(CCCNC(=N)N)C(=O)NC(CCCCN)C(=O)NC(CCCCN)C(=O)NC(CCCNC(=N)N)C(N)=O'
REGRESSIONS = ["CC(=O)N(C)C(=O)C", "CC(=O)N1CCCCC1=O", REPORTED_SMILES]


@pytest.fixture(params=["tokenizer/pep2fragments.py", "inference/tokenizer/pep2fragments.py"])
def tokenizer(request):
    spec = importlib.util.spec_from_file_location("fragment_regression", ROOT / request.param)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize("smiles", REGRESSIONS)
@pytest.mark.parametrize("side_chain_cut", [False, True])
def test_repeated_amide_matches_fragment_once_per_bond(tokenizer, smiles, side_chain_cut):
    mol = Chem.MolFromSmiles(smiles)
    assert mol is not None
    cuts, pairs = tokenizer.get_cut_bond_idx(mol, side_chain_cut=side_chain_cut)
    assert len(cuts) == len(pairs) == len(set(cuts))
    assert cuts
    for bond_id, (source, target) in zip(cuts, pairs):
        bond = mol.GetBondBetweenAtoms(source, target)
        assert bond is not None and bond.GetIdx() == bond_id
    fragments = Chem.GetMolFrags(Chem.FragmentOnBonds(mol, cuts, addDummies=False), asMols=True)
    assert sum(fragment.GetNumAtoms() for fragment in fragments) == mol.GetNumAtoms()
    parent = tokenizer.get_atom_parentAA(Chem.MolFromSmiles(smiles))
    assert set(parent) == set(range(mol.GetNumAtoms()))


@pytest.mark.parametrize("smiles,side_chain_cut,cuts,pairs", [
    ("NCC(=O)NCC(=O)O", False, [1, 4], [[1, 2], [4, 5]]),
    ("NCC(=O)NCC(=O)O", True, [1, 4], [[1, 2], [4, 5]]),
    ("CC(=O)NCC(=O)O", False, [0, 3], [[0, 1], [3, 4]]),
    ("CC(=O)NCC(=O)O", True, [0, 3], [[0, 1], [3, 4]]),
    ("N[C@@H](C)C(=O)N[C@@H](Cc1ccccc1)C(=O)O", False, [2, 5], [[1, 3], [5, 6]]),
    ("N[C@@H](C)C(=O)N[C@@H](Cc1ccccc1)C(=O)O", True, [2, 5, 7], [[1, 3], [5, 6], [7, 8]]),
])
def test_normal_peptide_cut_order_and_endpoints_unchanged(tokenizer, smiles, side_chain_cut, cuts, pairs):
    assert tokenizer.get_cut_bond_idx(Chem.MolFromSmiles(smiles), side_chain_cut) == (cuts, pairs)


def test_cross_rule_duplicate_retains_first_aligned_pair(tokenizer):
    # The same original bond can be proposed with opposite orientation by
    # different cleavage rules. Distinct bonds sharing an atom remain distinct.
    assert tokenizer._unique_cut_bonds([3, 1, 3, 4], [[2, 3], [1, 2], [3, 2], [3, 4]]) == (
        [3, 1, 4], [[2, 3], [1, 2], [3, 4]])
    assert tokenizer._unique_cut_bonds([], []) == ([], [])


def load_package():
    spec = importlib.util.spec_from_file_location("pepland", ROOT / "__init__.py", submodule_search_locations=[str(ROOT)])
    package = importlib.util.module_from_spec(spec)
    sys.modules["pepland"] = package
    spec.loader.exec_module(package)


@pytest.mark.parametrize("smiles", REGRESSIONS)
def test_real_inference_and_training_graphs_for_modified_peptides(smiles):
    load_package()
    from pepland.utils.process import Mol2HeteroGraph as inference_graph
    from pepland.model.data import Mol2HeteroGraph as training_graph
    mol = Chem.MolFromSmiles(smiles)
    spec = importlib.util.spec_from_file_location("standalone_graph", ROOT / "inference/process.py")
    standalone = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(standalone)
    graphs = [inference_graph(smiles), standalone.Mol2HeteroGraph(smiles)] + [training_graph(Chem.Mol(mol), frag=mode) for mode in ("258", "410")]
    for graph in graphs:
        assert graph.num_nodes("a") == mol.GetNumAtoms()
        assert graph.num_edges("b") == 2 * mol.GetNumBonds()
        for kind in graph.ntypes:
            for key in graph.nodes[kind].data.keys():
                assert torch.isfinite(graph.nodes[kind].data[key]).all()
        for kind in graph.canonical_etypes:
            for key in graph.edges[kind].data.keys():
                assert torch.isfinite(graph.edges[kind].data[key]).all()
        atoms, fragments = graph.edges(etype=("a", "j", "p"))
        assert len(atoms) == mol.GetNumAtoms()
        assert len(fragments) == mol.GetNumAtoms()
        assert torch.equal(torch.sort(atoms).values, torch.arange(mol.GetNumAtoms()))


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_actual_checkpoint_modified_peptides_singleton_mixed(device):
    flag = "PEPLAND_GPU_CHECKPOINT_TESTS" if device == "cuda" else "PEPLAND_CHECKPOINT_TESTS"
    if os.environ.get(flag) != "1":
        pytest.skip("set " + flag + "=1 for actual checkpoint inference")
    guard = None
    if device == "cuda":
        spec = importlib.util.spec_from_file_location("gpu_test_support", ROOT / "test/test_checkpoint_gpu.py")
        support = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(support)
        guard = support.require_idle_gpu
        guard()
        if not torch.cuda.is_available():
            raise RuntimeError("GPU checkpoint tests enabled but CUDA is unavailable")
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
    tolerance = 1e-4 if device == "cuda" else 2e-5
    load_package()
    from pepland.model.core import PepLandFeatureExtractor
    from pepland.utils.process import Mol2HeteroGraph
    torch.set_num_threads(1)
    model = PepLandFeatureExtractor(str(ROOT / "inference/cpkt/model"), pooling="avg").cpu().eval()
    if guard is not None:
        guard()
        model = model.cuda()
        assert all(parameter.is_cuda for parameter in model.model.parameters())
    with torch.no_grad():
        node_singles = [model.extract_atom_fragment_embedding([Mol2HeteroGraph(smiles)], return_counts=True) for smiles in REGRESSIONS]
        singles = torch.cat([model([Mol2HeteroGraph(smiles)]) for smiles in REGRESSIONS])
        for order in ([0, 1, 2], [2, 0, 1], [1]):
            atom, fragment, ac, fc = model.extract_atom_fragment_embedding(
                [Mol2HeteroGraph(REGRESSIONS[i]) for i in order], return_counts=True)
            for row, index in enumerate(order):
                sa, sf, sac, sfc = node_singles[index]
                assert ac[row].item() == sac[0].item()
                assert fc[row].item() == sfc[0].item()
                torch.testing.assert_close(atom[row, :ac[row]], sa[0, :sac[0]], rtol=tolerance, atol=tolerance)
                torch.testing.assert_close(fragment[row, :fc[row]], sf[0, :sfc[0]], rtol=tolerance, atol=tolerance)
            batch = model([Mol2HeteroGraph(REGRESSIONS[i]) for i in order])
            assert batch.device.type == device
            assert torch.isfinite(batch).all()
            torch.testing.assert_close(batch, singles[order], rtol=tolerance, atol=tolerance)
