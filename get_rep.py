import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from dgl.dataloading import GraphDataLoader
import torch
from omegaconf import OmegaConf
from PharmHGT.model.data import MolGraphSet
from PharmHGT.utils.utils import fix_random_seed, get_device, load_model_masking, load_model_contextpred
from PharmHGT.utils.std_logger import Logger 
import pandas as pd

def get_rep_masking():
    cfg = OmegaConf.load('./configs/pretrain_masking.yaml')
    
    random_seed = cfg.train.random_seed
    device = get_device(cfg)
    fix_random_seed(random_seed, cuda_deterministic=True)
    
    model, linear_pred_atoms, linear_pred_bonds = load_model_masking(cfg.inference.model_path, device)
    smiles_list = ['CC[C@H](C)[C@H](NC(=O)[C@H](Cc1ccccc1)NC(=O)[C@@H]1CCCN1C(=O)[C@@H](N)CC(=O)O)C(=O)N1CCC[C@H]1C(=O)N[C@@H](Cc1ccccc1)C(=O)N[C@@H](Cc1ccc(O)cc1)C(=O)N[C@H](C(=O)O)[C@@H](C)O',
'CSCC[C@H](NC(=O)[C@H](CC(N)=O)NC(=O)[C@H](CC(N)=O)NC(=O)[C@H](CC(C)C)NC(=O)[C@H](CCC(=O)O)NC(=O)[C@@H](N)CC(C)C)C(=O)N[C@H](C(=O)N[C@@H](CC(N)=O)C(=O)N[C@@H](CC(C)C)C(=O)O)C(C)C']
    dataset = MolGraphSet(smiles_list)
    dataloader = GraphDataLoader(dataset, batch_size=2, shuffle=False)

    with torch.no_grad():
        for step, batch in enumerate(dataloader):
            batch = batch.to(device)
            print(batch.device)
            print(next(model.parameters()).device)
            atom_rep,pharm_rep = model(batch)
            print(atom_rep.shape)
        
# def get_rep_contextpred():
#     cfg = OmegaConf.load('./configs/pretrain_contextpred.yaml')
    
#     random_seed = cfg.train.random_seed
#     device = get_device(cfg)
#     fix_random_seed(random_seed, cuda_deterministic=True)
    
#     model_substruct, model_context  = load_model_contextpred(cfg.inference.model_path, device)
#     smiles_list = ['CC[C@H](C)[C@H](NC(=O)[C@H](Cc1ccccc1)NC(=O)[C@@H]1CCCN1C(=O)[C@@H](N)CC(=O)O)C(=O)N1CCC[C@H]1C(=O)N[C@@H](Cc1ccccc1)C(=O)N[C@@H](Cc1ccc(O)cc1)C(=O)N[C@H](C(=O)O)[C@@H](C)O',
# 'CSCC[C@H](NC(=O)[C@H](CC(N)=O)NC(=O)[C@H](CC(N)=O)NC(=O)[C@H](CC(C)C)NC(=O)[C@H](CCC(=O)O)NC(=O)[C@@H](N)CC(C)C)C(=O)N[C@H](C(=O)N[C@@H](CC(N)=O)C(=O)N[C@@H](CC(C)C)C(=O)O)C(C)C']
#     l1 = cfg.train.num_layer - 1
#     l2 = l1 + cfg.train.csize
#     # dataset = SmilesDataset(smiles_list=smiles_list, transform=ExtractSubstructureContextPair(cfg.train.num_layer, l1, l2))
#     # dataloader = make_loaders(dataset, batch_size=1, shuffle=False)
#     dataset = MolGraphSet(smiles_list=smiles_list, transform=ExtractSubstructureContextPair(cfg.train.num_layer, l1, l2))
#     dataloader = GraphDataLoader(dataset,batch_size=1, num_workers = 0)
    
#     with torch.no_grad():
#         for step, batch in enumerate(dataloader):
#             batch.to(device)
#             substruct_rep = model_substruct(batch.x_substruct, batch.edge_index_substruct, batch.edge_attr_substruct)[batch.center_substruct_idx]
#             ### creating context representations
#             overlapped_node_rep = model_context(batch.x_context, batch.edge_index_context, batch.edge_attr_context)[batch.overlap_context_substruct_idx]
                
#             print(substruct_rep.shape)
#             print(overlapped_node_rep.shape)
    
if __name__ =='__main__':
    # get_rep_masking()
    get_rep_masking()
