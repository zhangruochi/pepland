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

import argparse
import importlib.util
from pathlib import Path
import sys

# Direct execution works even when the checkout directory is renamed.
ROOT = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("pepland", ROOT / "__init__.py",
                                             submodule_search_locations=[str(ROOT)])
package = importlib.util.module_from_spec(spec)
sys.modules["pepland"] = package
spec.loader.exec_module(package)

import torch
from omegaconf import OmegaConf
from pepland.model.core import PepLandFeatureExtractor
from pepland.utils.inference_config import atom_index, resolve_path


def main(argv=None):
    parser = argparse.ArgumentParser(description="Extract PepLand peptide embeddings")
    parser.add_argument("--config", type=Path, default=ROOT / "configs" / "inference.yaml")
    args = parser.parse_args(argv)
    cfg = OmegaConf.load(args.config)
    device_ids = cfg.inference.get("device_ids", [])
    device = torch.device("cuda:{}".format(device_ids[0])
                          if torch.cuda.is_available() and device_ids else "cpu")
    model_path = resolve_path(cfg.inference.model_path, ROOT, ROOT / "inference")
    data_path = resolve_path(cfg.inference.data, ROOT, ROOT / "inference")
    model = PepLandFeatureExtractor(str(model_path), cfg.inference.pool,
                                   padding_mode=cfg.inference.get("padding_mode", "exclude"))
    model.to(device).eval()
    smiles = [line.strip() for line in data_path.read_text().splitlines() if line.strip()]
    with torch.no_grad():
        embeddings = model(smiles, atom_index=atom_index(cfg.inference.get("atom_index", False)))
    print(tuple(embeddings.shape))
    print(embeddings.cpu().numpy())


if __name__ == "__main__":
    main()
