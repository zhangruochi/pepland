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
from utils.inference_config import atom_index as parse_atom_index, resolve_path
from pathlib import Path
import argparse


def load_model(cfg):
    model_path = str(resolve_path(cfg.inference.model_path, Path(root_dir).parent, Path(root_dir)))
    sys.path.append(os.path.join(model_path, "code"))
    print("loading model from : {}".format(model_path))
    model = mlflow.pytorch.load_model(model_path, map_location="cpu")
    padding_mode = cfg.inference.get("padding_mode", "exclude")
    if padding_mode not in ("exclude", "legacy"):
        raise ValueError("padding_mode must be exclude or legacy")
    for module in model.modules():
        if module.__class__.__name__ == "MVMP":
            module.empty_relation_mode = "per_graph" if padding_mode == "exclude" else "legacy"
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
    parser = argparse.ArgumentParser(description="Extract PepLand peptide embeddings")
    parser.add_argument("--config", default=os.path.join(root_dir, "../configs/inference.yaml"))
    args = parser.parse_args()
    cfg = OmegaConf.load(args.config)
    pooling = cfg.inference.pool
    orig_cwd = os.path.dirname(__file__)

    device = torch.device("cuda:{}".format(cfg.inference.device_ids[0]
                                           ) if torch.cuda.is_available()
                          and len(cfg.inference.device_ids) > 0 else "cpu")

    model = load_model(cfg)
    model.to(device)

    ## Get the smiles list
    data_path = resolve_path(cfg.inference.data, Path(root_dir).parent, Path(root_dir))
    with open(data_path, "r") as f:
        input_smiles = [line.strip() for line in f if line.strip()]
    if not input_smiles:
        raise ValueError("input_smiles must be nonempty")

    print("total smiles: {}".format(len(input_smiles)))
    print(input_smiles[0])

    graphs = []
    for i, smi in enumerate(input_smiles):
        try:
            # Use local import instead of package import
            from process import Mol2HeteroGraph
            graph = Mol2HeteroGraph(smi.strip())
            graphs.append(graph)
        except Exception as e:
            print(e, 'invalid', smi)

    atom_index = parse_atom_index(cfg.inference.get("atom_index", False))
    if not graphs:
        raise ValueError("input contains no valid molecules")
    bg = dgl.batch(graphs)

    bg = bg.to(device)
    with torch.no_grad():
        atom_embed, frag_embed = model(bg)
    bg.nodes['a'].data['h'] = atom_embed
    bg.nodes['p'].data['h'] = frag_embed
    atom_rep = split_batch(bg, 'a', 'h', device)

    # if set atom index, only return the atom embedding with the index
    if atom_index is not None:
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
