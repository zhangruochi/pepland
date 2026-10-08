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
# Support both direct script execution and imports from the repository root.
sys.path.insert(0, os.path.dirname(root_dir))
from utils.readout import split_batch, pool_atom_fragment


def load_model(cfg):
    model_path = os.path.join(root_dir, cfg.inference.model_path)
    sys.path.append(os.path.join(model_path, "code"))
    print("loading model from : {}".format(model_path))
    model = mlflow.pytorch.load_model(model_path, map_location="cpu")
    model.eval()
    return model


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
    for i, smi in enumerate(input_smiles):
        try:
            # Use local import instead of package import
            from process import Mol2HeteroGraph
            graph = Mol2HeteroGraph(smi)
            graphs.append(graph)
        except Exception as e:
            print(e, 'invalid', smi)

    atom_index = cfg.inference.atom_index
    bg = dgl.batch(graphs)

    bg = bg.to(device)
    with torch.no_grad():
        atom_embed, frag_embed = model(bg)
    bg.nodes['a'].data['h'] = atom_embed
    bg.nodes['p'].data['h'] = frag_embed
    atom_rep = split_batch(bg, 'a', 'h', device)

    # if set atom index, only return the atom embedding with the index
    if atom_index:
        pep_embeds = atom_rep[:, atom_index].detach().cpu().numpy()
    else:

        # if not set atom index, return the whole peptide embedding (atom + fragment)
        frag_rep = split_batch(bg, 'p', 'h', device)
        embed = pool_atom_fragment(
            atom_rep, frag_rep, bg.batch_num_nodes('a'), bg.batch_num_nodes('p'),
            pooling=pooling, padding_mode=cfg.inference.get('padding_mode', 'exclude')
        ).detach().cpu().numpy()
        pep_embeds = embed

    print(pep_embeds.shape)
    print(pep_embeds)
