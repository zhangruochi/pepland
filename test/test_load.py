"""Portable real graph/loader regressions using repository bundled peptides."""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import dgl
import pandas as pd
import pytest
import torch
from omegaconf import OmegaConf
from rdkit import Chem

from model.data import (
    MaskAtom,
    Mol2HeteroGraph,
    MolGraphSet,
    atom_mask_features,
    create_dataset,
    make_loaders,
)

torch.set_num_threads(1)


@pytest.fixture
def peptide_splits(tmp_path):
    smiles = list(dict.fromkeys((ROOT / 'data' / 'example.smi').read_text().splitlines()))
    smiles.append('NCC(=O)NCC(=O)O')
    folder = tmp_path / 'peptides'
    folder.mkdir()
    for split, values in {'train': smiles, 'valid': smiles[:2], 'test': smiles[2:]}.items():
        pd.DataFrame({'smiles': values}).to_csv(folder / f'{split}.csv', index=False)
    return folder, smiles


def config(fragment='258'):
    return OmegaConf.create({'train': {'fragment': fragment}})


def test_dataloader_size(peptide_splits):
    folder, _ = peptide_splits
    loaders = make_loaders(config(), ddp=False, dataset=str(folder), batch_size=2)
    for split, count in (('train', 3), ('valid', 2), ('test', 1)):
        loader = loaders[split]
        assert isinstance(loader.dataset, MolGraphSet)
        assert len(loader.dataset) == count
        assert loader.batch_size == 2
        batches = list(loader)
        assert len(batches) == (count + 1) // 2
        assert sum(batch.batch_size for batch in batches) == count


@pytest.mark.parametrize('fragment', ['258', '410'])
def test_graph(peptide_splits, fragment):
    folder, smiles = peptide_splits
    loader = make_loaders(config(fragment), ddp=False, dataset=str(folder), batch_size=2)['train']
    graphs = [graph for batch in loader for graph in dgl.unbatch(batch)]
    assert len(graphs) == len(smiles)
    for graph, smi in zip(graphs, smiles):
        mol = Chem.MolFromSmiles(smi)
        expected = Mol2HeteroGraph(mol, frag=fragment)
        assert graph.num_nodes('a') == mol.GetNumAtoms()
        assert graph.num_edges('b') == 2 * mol.GetNumBonds()
        for node_type in expected.ntypes:
            for name, values in expected.nodes[node_type].data.items():
                torch.testing.assert_close(graph.nodes[node_type].data[name], values)
            assert torch.isfinite(graph.nodes[node_type].data['f']).all()
        for edge_type in expected.canonical_etypes:
            for actual, reference in zip(graph.edges(etype=edge_type), expected.edges(etype=edge_type)):
                assert torch.equal(actual, reference)
            for name, values in expected.edges[edge_type].data.items():
                torch.testing.assert_close(graph.edges[edge_type].data[name], values)
        atoms, fragments = graph.edges(etype=('a', 'j', 'p'))
        assert torch.equal(torch.sort(atoms).values, torch.arange(mol.GetNumAtoms()))
        reverse_fragments, reverse_atoms = graph.edges(etype=('p', 'j', 'a'))
        assert torch.equal(atoms, reverse_atoms)
        assert torch.equal(fragments, reverse_fragments)


def test_dataset_transform_preserves_labels(peptide_splits):
    folder, smiles = peptide_splits
    masker = MaskAtom(119, 5, 0.1, mask_edge=False, mask_fragment=False, mask_amino=False, mask_pep=False)

    def transform(graph):
        return masker(graph, masked_atom_indices=[0])

    dataset = create_dataset(config(), folder / 'train.csv', transform)
    for graph, smi in zip(dataset, smiles):
        original = Mol2HeteroGraph(Chem.MolFromSmiles(smi))
        for node_type in original.ntypes:
            assert torch.equal(graph.nodes[node_type].data['label'], original.nodes[node_type].data['label'])
        mask = graph.nodes['a'].data['mask']
        assert mask.dtype == torch.bool
        assert mask.sum().item() == 1 and mask[0].item()
        torch.testing.assert_close(graph.nodes['a'].data['f'][0], torch.tensor(atom_mask_features(), dtype=torch.float32))
        torch.testing.assert_close(graph.nodes['a'].data['f'][1:], original.nodes['a'].data['f'][1:])


def test_invalid_smiles_are_logged_and_skipped():
    records = []
    dataset = MolGraphSet(config(), pd.DataFrame({'smiles': ['NCC(=O)NCC(=O)O', 'not-a-smiles']}), log=lambda *message: records.append(message))
    assert len(list(dataset)) == 1
    assert records == [('invalid', 'not-a-smiles')]


@pytest.mark.parametrize('worker_count', [2, 5])
def test_worker_partition_covers_each_row_once(peptide_splits, monkeypatch, worker_count):
    from types import SimpleNamespace

    folder, smiles = peptide_splits
    dataset = create_dataset(config(), folder / 'train.csv', transform=None)
    graphs = []
    for worker_id in range(worker_count):
        monkeypatch.setattr(torch.utils.data, 'get_worker_info', lambda worker_id=worker_id: SimpleNamespace(id=worker_id, num_workers=worker_count))
        graphs.extend(list(dataset))
    assert len(graphs) == len(smiles)
    counts = [graph.num_nodes('a') for graph in graphs]
    assert counts == [Chem.MolFromSmiles(smi).GetNumAtoms() for smi in smiles]


def test_empty_dataset_produces_no_batches(tmp_path):
    path = tmp_path / 'empty.csv'
    pd.DataFrame({'smiles': []}).to_csv(path, index=False)
    dataset = create_dataset(config(), path, transform=None)
    assert len(dataset) == 0
    assert list(dgl.dataloading.GraphDataLoader(dataset, batch_size=2)) == []


def test_distributed_loader_equal_steps_and_complete_valid_coverage(tmp_path):
    smiles = ['NCC(=O)O', 'NCC(=O)NCC(=O)O', 'NCC(=O)NCC(=O)NCC(=O)O', 'NCC(=O)NCC(=O)NCC(=O)NCC(=O)O']
    for split in ('train', 'test', 'valid'):
        pd.DataFrame({'smiles': smiles[:2] + ['invalid-smiles'] + smiles[2:]}).to_csv(tmp_path / f'{split}.csv', index=False)
    ranks = [make_loaders(config(), ddp=True, dataset=str(tmp_path), world_size=2, global_rank=rank, batch_size=1) for rank in range(2)]
    expected_counts = {Chem.MolFromSmiles(smi).GetNumAtoms() for smi in smiles}
    for epoch in (0, 1):
        for split in ('train', 'valid', 'test'):
            counts = []
            for loaders in ranks:
                loader = loaders[split]
                loader.sampler.set_epoch(epoch)
                assert len(loader.dataset) == 4
                batches = list(loader)
                assert len(batches) == 2
                counts.append({batch.num_nodes('a') for batch in batches})
            assert counts[0].isdisjoint(counts[1])
            assert counts[0] | counts[1] == expected_counts


@pytest.mark.parametrize('world_size,rank', [(0, 0), (2, -1), (2, 2)])
def test_distributed_loader_rejects_invalid_rank(peptide_splits, world_size, rank):
    folder, _ = peptide_splits
    with pytest.raises(ValueError):
        make_loaders(config(), ddp=True, dataset=str(folder), world_size=world_size, global_rank=rank)
