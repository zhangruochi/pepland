#!/usr/bin/env python3
# -*- coding:utf-8 -*-
###
# File: /home/richard/projects/biology_llm_platform/mlm/pepland/inference.py
# Project: /home/richard/projects/biology_llm_platform/mlm/pepland
# Created Date: Thursday, November 28th 2024, 10:05:57 am
# Author: Ruochi Zhang
# Email: zrc720@gmail.com
# -----
# Last Modified: Sun Dec 01 2024
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

root_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(root_dir))
from pepland.model.core import PepLandFeatureExtractor
import mlflow
import torch
import torch.nn as nn
from omegaconf import OmegaConf
import dgl
from typing import List, Union

if __name__ == "__main__":
    cfg = OmegaConf.load(os.path.join(root_dir, "./configs/inference.yaml"))
    pooling = cfg.inference.pool
    model_path = os.path.join(root_dir, cfg.inference.model_path)
    device = torch.device("cuda:{}".format(cfg.inference.device_ids[0]
                                           ) if torch.cuda.is_available()
                          and len(cfg.inference.device_ids) > 0 else "cpu")
    data_path = os.path.join(root_dir, cfg.inference.data)

    model = PepLandFeatureExtractor(model_path, pooling)

    ## Get the smiles list
    with open(cfg.inference.data, "r") as f:
        input_smiles = f.readlines()

    with torch.no_grad():
        pep_embeds = model(input_smiles)

    print(pep_embeds)
