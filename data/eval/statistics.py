#!/usr/bin/env python3
# -*- coding:utf-8 -*-
###
# File: /home/richard/projects/pepland/data/eval/statistics.py
# Project: /home/richard/projects/pepland/data/eval
# Created Date: Thursday, July 18th 2024, 11:32:56 am
# Author: Ruochi Zhang
# Email: zrc720@gmail.com
# -----
# Last Modified: Thu Jul 18 2024
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

import pandas as pd 
from collections import Counter


## ---- c-CPP ----
all_seqs = []
all_labels = []
with open("./c-CPP.txt", "r") as f:
    
    lines = f.readlines()
    for line in lines:
        seq, label = line.split(",")
        seq = seq.strip()
        label = int(label.strip())
        if seq:
            all_seqs.append(seq)
            all_labels.append(label)

print(Counter(all_labels))


## ---- nc-CPP ----

nc_cpp_df = pd.read_csv("./nc-CPP.csv")
print(nc_cpp_df.shape)



## ------ c-Binding ----
c_binding_df = pd.read_csv("./c-binding.csv", index_col=0)
print(c_binding_df.shape)


## ---- nc-Binding ----
nc_binding_df = pd.read_csv("./nc-binding.csv", index_col=0)
print(nc_binding_df.shape)



## ---- solubility ----
all_seqs = []
all_labels = []
with open("./c-Sol.txt", "r") as f:
    lines = f.readlines()
    for line in lines:
        seq, label = line.split(",")
        seq = seq.strip()
        label = int(label.strip())
        if seq:
            all_seqs.append(seq)
            all_labels.append(label)
print(Counter(all_labels))