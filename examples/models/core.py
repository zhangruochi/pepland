#!/usr/bin/env python3
# -*- coding:utf-8 -*-
###
# File: /home/richard/projects/pepcorsser/models/core.py
# Project: /home/richard/projects/pepland/finetune_example/models
# Created Date: Sunday, April 28th 2024, 12:03:00 am
# Author: Ruochi Zhang
# Email: zrc720@gmail.com
# -----
# Last Modified: Tue May 07 2024
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
import torch
import torch.nn as nn
from .pepland.inference import PepLandPredictor
import numpy as np
from typing import List, Union


class PropertyPredictor(nn.Module):

    def __init__(self, model_path, pool_type = "avg", hidden_dims = [256,128], mlp_dropout=0.1):
        super(PropertyPredictor, self).__init__()

        self.feature_model = PepLandPredictor(
            model_path=model_path,
            pool_type=pool_type)

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

    def forward(self,
                pep_graph):

        graph_rep = self.feature_model(pep_graph)
        pred = self.mlp(graph_rep)
        
        return pred

    @property
    def device(self):
        return next(self.parameters()).device
