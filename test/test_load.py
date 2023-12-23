import sys
sys.path.append('../') 
from data import make_loaders,create_dataset,MaskAtom,MolGraphSet,Mol2HeteroGraph
from dgl.dataloading import GraphDataLoader
from omegaconf import DictConfig, OmegaConf
import pandas as pd
from rdkit import Chem
import torch
# def get_dataset(dataset, mask_edge=False):
#     transform = MaskAtom(num_atom_type=119, num_edge_type=5, mask_rate=0.1, mask_edge=mask_edge)
#     train_dataset = create_dataset('/mnt/data/xiuyuting/'+dataset+'/train.csv',transform=transform)
#     test_dataset = create_dataset('/mnt/data/xiuyuting/'+dataset+'/test.csv',transform=transform)
#     valid_dataset = create_dataset('/mnt/data/xiuyuting/'+dataset+'/valid.csv',transform=transform)
#     return train_dataset,test_dataset,valid_dataset

def test_dataloader_size():
    # train_loader = GraphDataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers = num_workers)
    # valid_loader = GraphDataLoader(valid_dataset, batch_size=batch_size, shuffle=True, num_workers = num_workers)
    # test_loader = GraphDataLoader(test_dataset, batch_size=batch_size, shuffle=True, num_workers = num_workers)
    # assert
    cfg = OmegaConf.load('../configs/pretrain_masking.yaml')
    dataloaders = make_loaders(ddp=cfg.mode.ddp,
                            dataset=cfg.train.dataset,
                            world_size=0,
                            global_rank=0,
                            batch_size=cfg.train.batch_size,
                            num_workers=cfg.train.num_workers,
                            transform=MaskAtom(num_atom_type=119, num_edge_type=5, mask_rate=cfg.train.mask_rate, mask_edge=cfg.train.mask_edge),
                            )
    assert len(dataloaders['train'])>0
    assert len(dataloaders['test'])>0
    assert len(dataloaders['valid'])>0
    
    assert len(dataloaders['train'].dataset) == 712397
    assert len(dataloaders['valid'].dataset) == 39578
    assert len(dataloaders['test'].dataset) == 39578
# train_dataset,test_dataset,valid_dataset = dataset_test('pep_test')


def test_graph():
    # train_loader = GraphDataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers = num_workers)
    # valid_loader = GraphDataLoader(valid_dataset, batch_size=batch_size, shuffle=True, num_workers = num_workers)
    # test_loader = GraphDataLoader(test_dataset, batch_size=batch_size, shuffle=True, num_workers = num_workers)
    # assert
    cfg = OmegaConf.load('../configs/pretrain_masking.yaml')
    transform_func = MaskAtom(num_atom_type=119, num_edge_type=5, mask_rate=cfg.train.mask_rate, mask_edge=cfg.train.mask_edge)
    dataloaders = make_loaders(ddp=cfg.mode.ddp,
                            dataset=cfg.train.dataset,
                            world_size=0,
                            global_rank=0,
                            batch_size=cfg.train.batch_size,
                            num_workers=cfg.train.num_workers,
                            transform= transform_func)
    data = pd.read_csv('/mnt/data/xiuyuting/pep_test/train.csv').head(10)
    assert isinstance(dataloaders['train'].dataset, MolGraphSet)
    # assert dataloaders['train'].dataset.data.equals(data)
    iterator = dataloaders['train']
    assert dataloaders['train'].batch_size==1
    for i, item in enumerate(iterator):
        smi = data.iloc[i]['smiles']
        # label = row[target].values.astype(float)
        mol = Chem.MolFromSmiles(smi)
        g = Mol2HeteroGraph(mol)
        g = transform_func(g)
        
        assert item.number_of_nodes('a')==g.number_of_nodes('a')
        assert torch.eq(item.nodes['a'].data['label'] ,g.nodes['a'].data['label']).all()
        assert torch.eq(item.nodes['a'].data['mask'] ,g.nodes['a'].data['mask']).all()
        
        # assert item == g, f"Dataset iterator should return the correct data at index {i}"
