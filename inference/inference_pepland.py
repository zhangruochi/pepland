import os
import sys
import mlflow
import torch.nn.functional as F
import torch
import torch.nn as nn
from omegaconf import OmegaConf
import numpy as np
import pandas as pd
import dgl

root_dir = os.path.dirname(os.path.abspath(__file__))

def load_model(cfg):
    model_path = os.path.join(root_dir, cfg.inference.model_path)
    sys.path.append(os.path.join(model_path, "code"))
    print("loading model from : {}".format(model_path))
    model = mlflow.pytorch.load_model(model_path, map_location="cpu")
    model.eval()
    return model

def split_batch(bg, ntype, field, device):
    hidden = bg.nodes[ntype].data[field]
    node_size = bg.batch_num_nodes(ntype)
    start_index = torch.cat([torch.tensor([0],device=device),torch.cumsum(node_size,0)[:-1]])
    max_num_node = max(node_size)
    # padding
    hidden_lst = []
    for i  in range(bg.batch_size):
        start, size = start_index[i],node_size[i]
        assert size != 0, size
        cur_hidden = hidden.narrow(0, start, size)
        cur_hidden = torch.nn.ZeroPad2d((0,0,0,max_num_node-cur_hidden.shape[0]))(cur_hidden)
        hidden_lst.append(cur_hidden.unsqueeze(0))
    hidden_lst = torch.cat(hidden_lst, 0)
    return hidden_lst

class Permute(nn.Module):

    def __init__(self):
        super(Permute, self).__init__()

    def forward(self, x):
        return torch.permute(x, (0, 2, 1))

class Squeeze(nn.Module):

    def __init__(self, dim):
        super(Squeeze, self).__init__()

        self.dim = dim

    def forward(self, x):
        return torch.squeeze(x, dim=self.dim)


if __name__ == "__main__":
    cfg = OmegaConf.load(os.path.join(root_dir, "../configs/inference.yaml"))
    pooling = cfg.inference.pool
    orig_cwd = os.path.dirname(__file__)

    device = torch.device("cuda:{}".format(cfg.inference.device_ids[0]
                                        ) if torch.cuda.is_available()
                        and len(cfg.inference.device_ids) > 0 else "cpu")

    model = load_model(cfg)
    model.to(device)

    ## Get the smiles list
    with open(cfg.inference.data, "r") as f:
        input_smiles = f.readlines()

    print("total smiles: {}".format(len(input_smiles)))
    print(input_smiles[0])
    
    graphs = []
    for i,smi in enumerate(input_smiles):
        try:
            sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
            from pepland.inference.process import Mol2HeteroGraph
            graph = Mol2HeteroGraph(smi)
            graphs.append(graph)
        except Exception as e:
                print(e,'invalid',smi)
                
    print("graph: ")
    print(graphs[0])
    if pooling == 'max':
        pool = nn.Sequential(Permute(), nn.AdaptiveMaxPool1d(output_size=1), Squeeze(dim=-1))
    elif pooling == 'avg':
        pool = nn.Sequential(Permute(), nn.AdaptiveAvgPool1d(output_size=1), Squeeze(dim=-1))
    index = cfg.inference.atom_index
    embeds = []
    bg = dgl.batch(graphs)
    
    bg = bg.to(device)
    atom_embed, frag_embed = model(bg)
    bg.nodes['a'].data['h'] = atom_embed
    bg.nodes['p'].data['h'] = frag_embed
    atom_rep = split_batch(bg,'a','h',device) 
    if cfg.inference.atom_index:
        embeds.append(atom_rep[:,index].detach().cpu().numpy())
    else:
        frag_rep = split_batch(bg,'p','h',device)
        embed = pool(torch.cat([atom_rep,frag_rep],dim=1)).detach().cpu().numpy()
        embeds.append(embed)
    embeds=np.vstack(embeds)
    print(embeds.shape)
    print(embeds)