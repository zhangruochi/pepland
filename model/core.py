#!/usr/bin/env python3
# -*- coding:utf-8 -*-
###
# File: /home/richard/projects/pepcorsser/models/core.py
# Project: /home/richard/projects/biology_llm_platform/mlm/pepland/model
# Created Date: Sunday, April 28th 2024, 12:03:00 am
# Author: Ruochi Zhang
# Email: zrc720@gmail.com
# -----
# Last Modified: Thu Nov 28 2024
# Modified By: Ruochi Zhang
# -----
# Copyright (c) 2024 Bodkin World Domination Enterprises
#
# MIT License
#
# Copyright (c) 2024 Ruochi Zhang
#
# Permission is hereby granted, free of charge, to any person obtaining a copy of
# this software and associated documentation files (the "Software"), to deal in
# the Software without restriction, including without limitation the rights to
# use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies
# of the Software, and to permit persons to whom the Software is furnished to do
# so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
# -----
###
import os
import sys
from ..utils.commons import load_model, split_batch, Permute, Squeeze, to_canonical_smiles
from ..utils.process import Mol2HeteroGraph
import torch
import torch.nn as nn
import numpy as np
from typing import List, Union
import dgl


class PepLandFeatureExtractor(nn.Module):

    def __init__(self, model_path, pooling: Union[str, None] = 'avg'):
        """ Initialize the PepLandInference class
            args:
                model_path: str, the path to the model directory
                device: torch.device
                pooling: str, the pooling method, either 'max
            
        """
        super(PepLandFeatureExtractor, self).__init__()

        self.model = load_model(model_path)
        if pooling == 'max':
            pooling_layer = nn.Sequential(Permute(),
                                          nn.AdaptiveMaxPool1d(output_size=1),
                                          Squeeze(dim=-1))
        elif pooling == 'avg':
            pooling_layer = nn.Sequential(Permute(),
                                          nn.AdaptiveAvgPool1d(output_size=1),
                                          Squeeze(dim=-1))
        else:
            pooling_layer = None

        self.pooling_layer = pooling_layer

    def tokenize(self, input_smiles: List[str]) -> List[str]:
        if isinstance(input_smiles, str):
            input_smiles = [input_smiles]

        input_smiles = to_canonical_smiles(input_smiles)

        graphs = []
        for i, smi in enumerate(input_smiles):
            try:
                graph = Mol2HeteroGraph(smi)
                graphs.append(graph)
            except Exception as e:
                raise ValueError(
                    "Error processing SMILES string: {}. {}".format(smi, e))
        return graphs

    def forward(
            self,
            input_smiles: List[str],
            atom_index: Union[int, List[int], None] = None) -> torch.Tensor:
        """ Extract the peptide embedding from the model
            args:
                input_smiles: List of SMILES strings
                atom_index: if set, only return the atom embedding with the index
            return:
                pep_embeds: torch.Tensor

            examples:
                input_smiles = ['CCO', 'CCN']
                atom_index = 1
                pep_embeds.shape == [2, 1, 300]

                input_smiles = ['CCO', 'CCN']
                atom_index = [1, 2]
                pep_embeds.shape == [2, 2, 300]

                input_smiles = ['CCO', 'CCN']
                atom_index = None
                pep_embeds.shape == [2, 300]
        """
        if isinstance(atom_index, int):
            atom_index = [atom_index]

        graphs = self.tokenize(input_smiles)
        bg = dgl.batch(graphs).to(self.device)

        atom_embed, frag_embed = self.model(bg)
        bg.nodes['a'].data['h'] = atom_embed
        bg.nodes['p'].data['h'] = frag_embed

        atom_rep = split_batch(bg, 'a', 'h', self.device)

        # if set atom index, only return the atom embedding with the index
        if atom_index:
            pep_embeds = atom_rep[:, atom_index]
        else:
            # if not set atom index, return the whole peptide embedding (atom + fragment)
            frag_rep = split_batch(bg, 'p', 'h', self.device)
            if self.pooling_layer is not None:
                embed = self.pooling_layer(
                    torch.cat([atom_rep, frag_rep], dim=1))
            pep_embeds = embed

        return pep_embeds

    @property
    def device(self):
        return next(self.parameters()).device


class PropertyPredictor(nn.Module):
    """ PropertyPredictor
        This model is used to predict the property of the peptide
        based on the peptide graph representation from pre-trained PepLand model.
    """

    def __init__(self,
                 model_path,
                 pooling="avg",
                 hidden_dims=[256, 128],
                 mlp_dropout=0.1):
        """ Initialize the PropertyPredictor class"""
        super(PropertyPredictor, self).__init__()

        self.feature_model = PepLandFeatureExtractor(model_path=model_path,
                                                     pooling=pooling)

        self.mlp = nn.Sequential()
        input_dim = 300
        for i, hidden_dim in enumerate(hidden_dims):
            self.mlp.add_module('linear_{}'.format(i),
                                nn.Linear(input_dim, hidden_dim))
            self.mlp.add_module('relu_{}'.format(i), nn.ReLU())
            self.mlp.add_module('dropout_{}'.format(i),
                                nn.Dropout(mlp_dropout))
            input_dim = hidden_dim

        self.mlp.add_module('output', nn.Linear(hidden_dim, 1))
        self.mlp.add_module('sigmoid', nn.Sigmoid())

    def tokenize(self, input_molecules: List[str]) -> List[dgl.DGLHeteroGraph]:
        """ Tokenize the input SMILES strings into DGLHeteroGraph
            args:
                input_molecules: List of SMILES strings
            return: List of dgl.DGLHeteroGraph
        """
        return self.feature_model.tokenize(input_molecules)

    def forward(
            self, input_molecules: Union[List[str],
                                         dgl.DGLHeteroGraph]) -> torch.Tensor:
        """ args:
            input_molecules: List of SMILES strings or dgl.DGLHeteroGraph
            return: torch.Tensor
        """

        graph_rep = self.feature_model(input_molecules)
        pred = self.mlp(graph_rep)

        return pred

    @property
    def device(self):
        return next(self.parameters()).device