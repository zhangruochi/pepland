"""Exercise feature and pooling implementations without optional trainer services."""
import ast
from pathlib import Path

import pytest
from rdkit import Chem
import torch

ROOT = Path(__file__).resolve().parents[1]
FEATURE_FILES = [ROOT / 'model/data.py', ROOT / 'utils/process.py',
                 ROOT / 'inference/process.py']
FEATURE_FILES += sorted((ROOT / 'cpkt').glob('*/code/model/data*.py'))
FEATURE_FILES += sorted((ROOT / 'inference/cpkt').glob('*/code/model/data*.py'))


def feature_functions(source):
    tree = ast.parse(source)
    nodes = [node for node in tree.body
             if isinstance(node, ast.FunctionDef)
             and node.name in ('bond_features', 'onek_encoding_unk')]
    namespace = {'Chem': Chem, 'BOND_FDIM': 14}
    exec(compile(ast.Module(body=nodes, type_ignores=[]), '<features>', 'exec'), namespace)
    return namespace['bond_features']


@pytest.mark.parametrize('path', FEATURE_FILES, ids=lambda path: str(path.relative_to(ROOT)))
def test_bond_features_missing_and_historical_parity(path):
    source = path.read_text()
    current = feature_functions(source)
    assert current(None) == [1] + [0] * 13
    # Compare real valid-bond outputs with the unchanged commit implementation.
    original = (ROOT / 'test/fixtures/bond_features_ab0f66e.py.txt').read_text()
    previous = feature_functions(original)
    for smiles in ('CC', 'C=C', 'C#N', 'c1ccccc1', 'F/C=C/F'):
        for bond in Chem.MolFromSmiles(smiles).GetBonds():
            assert current(bond) == previous(bond)
            assert len(current(bond)) == 14
    # Ensure the module itself defines its previously missing constant.
    tree = ast.parse(source)
    constants = [node for node in tree.body if isinstance(node, ast.Assign)
                 and any(isinstance(t, ast.Name) and t.id == 'BOND_FDIM' for t in node.targets)]
    assert len(constants) == 1
    assert ast.literal_eval(constants[0].value) == 14


def pool_function():
    tree = ast.parse((ROOT / 'trainer.py').read_text())
    trainer = next(node for node in tree.body if isinstance(node, ast.ClassDef)
                   and node.name == 'Contextpred_Trainer')
    method = next(node for node in trainer.body if isinstance(node, ast.FunctionDef)
                  and node.name == 'pool_func')
    namespace = {'torch': torch}
    exec(compile(ast.Module(body=[method], type_ignores=[]), '<pool>', 'exec'), namespace)
    return lambda x, batch, mode: namespace['pool_func'](None, x, batch, mode)


@pytest.mark.parametrize('mode', ['sum', 'mean', 'max'])
def test_context_pool_unsorted_negative_missing_and_gradients(mode):
    pool = pool_function()
    x = torch.tensor([[-3., 0.], [-2., -5.], [-4., -1.]], requires_grad=True)
    batch = torch.tensor([2, 0, 2])
    expected = {'sum': [[-2., -5.], [0., 0.], [-7., -1.]],
                'mean': [[-2., -5.], [0., 0.], [-3.5, -.5]],
                'max': [[-2., -5.], [0., 0.], [-3., 0.]]}[mode]
    result = pool(x, batch, mode)
    torch.testing.assert_close(result, torch.tensor(expected))
    result.sum().backward()
    assert x.grad is not None and torch.isfinite(x.grad).all()
    assert pool(torch.empty(0, 2), torch.empty(0, dtype=torch.long), mode).shape == (0, 2)


def test_context_pool_rejects_invalid_inputs():
    pool = pool_function()
    x = torch.ones(2, 3)
    for batch in (torch.tensor([-1, 0]), torch.tensor([0.0, 1.0]), torch.tensor([0])):
        with pytest.raises(ValueError):
            pool(x, batch, 'mean')
    with pytest.raises(ValueError):
        pool(x, torch.tensor([0, 0]), 'unsupported')


def test_real_graph_edgeless_and_invalid_molecule_boundaries():
    import importlib.util
    import sys
    spec = importlib.util.spec_from_file_location(
        'pepland', ROOT / '__init__.py', submodule_search_locations=[str(ROOT)])
    package = importlib.util.module_from_spec(spec)
    sys.modules['pepland'] = package
    spec.loader.exec_module(package)
    from pepland.utils.process import Mol2HeteroGraph
    from model.data import Mol2HeteroGraph as training_graph
    for smiles in ('C', '[NH4+]'):
        for graph in (Mol2HeteroGraph(smiles), training_graph(Chem.MolFromSmiles(smiles))):
            assert tuple(graph.edges['b'].data['x'].shape) == (0, 14)
            assert tuple(graph.edges['r'].data['x'].shape) == (0, 14)
    for value in ('', 'invalid', None, 123):
        with pytest.raises(ValueError, match='molecule must'):
            Mol2HeteroGraph(value)
    for mol in (None, Chem.MolFromSmiles('')):
        with pytest.raises(ValueError, match='molecule must'):
            training_graph(mol)
