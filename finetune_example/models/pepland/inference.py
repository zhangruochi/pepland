import os
import sys
from .process import Mol2HeteroGraph

import mlflow
import torch.nn.functional as F
import torch
import torch.nn as nn
from omegaconf import OmegaConf
import numpy as np
import pandas as pd
import dgl
from typing import List, Union


def load_model(cfg):
    model_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                              cfg.inference.model_path)
    sys.path.append(os.path.join(model_path, "code"))
    print("loading model from : {}".format(model_path))
    model = mlflow.pytorch.load_model(model_path, map_location="cpu")
    model.eval()
    return model


def split_batch(bg, ntype, field, device):
    hidden = bg.nodes[ntype].data[field]
    node_size = bg.batch_num_nodes(ntype)
    start_index = torch.cat(
        [torch.tensor([0], device=device),
         torch.cumsum(node_size, 0)[:-1]])
    max_num_node = max(node_size)
    # padding
    hidden_lst = []
    for i in range(bg.batch_size):
        start, size = start_index[i], node_size[i]
        assert size != 0, size
        cur_hidden = hidden.narrow(0, start, size)
        cur_hidden = torch.nn.ZeroPad2d(
            (0, 0, 0, max_num_node - cur_hidden.shape[0]))(cur_hidden)
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


class FeatureExtractor():

    def __init__(self):
        cfg = OmegaConf.load(
        os.path.join(os.path.dirname(os.path.abspath(__file__)),
                     'inference.yaml'))
        self.cfg = cfg
        self.pooling = cfg.inference.pool
        self.device = torch.device(
            "cuda:{}".format(cfg.inference.device_ids[0]) if torch.cuda.
            is_available() and len(cfg.inference.device_ids) > 0 else "cpu")

        self.model = load_model(cfg)
        self.model.to(self.device)
        print("Model loaded successfully at device: {}".format(self.device))

    def __call__(self, input_smiles: List):
        graphs = []
        for i, smi in enumerate(input_smiles):
            try:
                graph = Mol2HeteroGraph(smi)
                graphs.append(graph)
            except Exception as e:
                print(e, 'invalid', smi)

        if self.pooling == 'max':
            pool = nn.Sequential(Permute(),
                                 nn.AdaptiveMaxPool1d(output_size=1),
                                 Squeeze(dim=-1))
        elif self.pooling == 'avg':
            pool = nn.Sequential(Permute(),
                                 nn.AdaptiveAvgPool1d(output_size=1),
                                 Squeeze(dim=-1))

        atom_index = self.cfg.inference.atom_index
        bg = dgl.batch(graphs)

        bg = bg.to(self.device)
        with torch.no_grad():
            atom_embed, frag_embed = self.model(bg)
        bg.nodes['a'].data['h'] = atom_embed
        bg.nodes['p'].data['h'] = frag_embed
        atom_rep = split_batch(bg, 'a', 'h', self.device)

        # if set atom index, only return the atom embedding with the index
        if atom_index:
            pep_embeds = atom_rep[:, atom_index].detach().cpu().numpy()
        else:
            # if not set atom index, return the whole peptide embedding (atom + fragment)
            frag_rep = split_batch(bg, 'p', 'h', self.device)
            embed = pool(torch.cat([atom_rep, frag_rep],
                                   dim=1)).detach().cpu().numpy()
            pep_embeds = embed
        return pep_embeds


class PepLandPredictor(nn.Module):

    def __init__(self,
                 model_path="./cpkt/model",
                 pool_type='avg'):
        super(PepLandPredictor, self).__init__()

        sys.path.append(os.path.join(model_path, "code"))
        print("loading model from : {}".format(model_path))
        print("pool type: {}".format(pool_type))
        self.hgnn = mlflow.pytorch.load_model(model_path, map_location="cpu")

        # remove all layer starts with readout and out
        for name, module in list(self.hgnn.named_children()):
            if name.startswith('readout') or name.startswith(
                    'out') or name.startswith('pool'):
                delattr(self.hgnn, name)

        # print all layers and shape

        self.pool_type = pool_type

        if self.pool_type == 'max':
            self.pool = nn.Sequential(Permute(),
                                      nn.AdaptiveMaxPool1d(output_size=1),
                                      Squeeze(dim=-1))
        elif self.pool_type == 'avg':
            self.pool = nn.Sequential(Permute(),
                                      nn.AdaptiveAvgPool1d(output_size=1),
                                      Squeeze(dim=-1))
        elif self.pool_type == 'gru':
            self.pool = nn.GRU(300,
                               300,
                               batch_first=True,
                               bidirectional=False,
                               num_layers=1)

    def tokenize(self, input_smiles: List):
        graphs = []
        for i, smi in enumerate(input_smiles):
            try:
                graph = Mol2HeteroGraph(smi)
                graphs.append(graph)
            except Exception as e:
                print(e, 'invalid', smi)

        bg = dgl.batch(graphs)
        bg = bg.to(self.device)
        return bg

    def forward(self, input_smiles: Union[List, dgl.DGLHeteroGraph]):

        if isinstance(input_smiles, list):
            bg = self.tokenize(input_smiles)
        else:
            bg = input_smiles

        atom_embed, frag_embed = self.hgnn(bg)

        # print(atom_embed.shape)

        bg.nodes['a'].data['h'] = atom_embed
        bg.nodes['p'].data['h'] = frag_embed
        atom_rep = split_batch(bg, 'a', 'h', self.device)
        frag_rep = split_batch(bg, 'p', 'h', self.device)
        graph_rep = torch.cat([atom_rep, frag_rep], dim=1)

        if self.pool_type == 'gru':
            graph_rep = self.pool(graph_rep)[0][:, -1, :]
        else:
            graph_rep = self.pool(graph_rep)

        return graph_rep

    @property
    def device(self):
        return next(self.parameters()).device


def pep_tokenize(input_smiles: List):
    graphs = []
    for i, smi in enumerate(input_smiles):
        try:
            graph = Mol2HeteroGraph(smi)
            graphs.append(graph)
        except Exception as e:
            print(e, 'invalid', smi)
    bg = dgl.batch(graphs)
    return bg
