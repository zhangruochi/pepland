# PepLand 

This repository contains the code for the paper [PepLand: a large-scale pre-trained peptide representation model for a comprehensive landscape of both canonical and non-canonical amino acids](https://arxiv.org/abs/2311.04419).

## Installation
```shell
conda env create -f environment.yaml
conda activate multiview
```

## Inference using pretrained model
```
cd inference 
python inference_pepland.py
```

## Fragment

In the config.yaml file, the train.fragment parameter is used to modify the fragment splitting strategy. Currently, there are two options: one that splits the side chains and another that does not. These options are represented by the lengths of the vocabulary. The option that splits the side chains is represented by "258", while the option that does not split is represented by "410". 
Please note that when changing the vocabulary, the address of the vocab file in model/data.py also needs to be modified.

## Masking Method
config.yaml
```shell
train.mask_pharm = True # random fragment masking
train.mask_rate = 0.8 # masking rate (random atom masking and random fragment masking)
train.mask_amino = 0.3 # or False, masking atoms of the same amino acid.
train.mask_pep = 0.8 # or False, masking atoms of side chains 
```

## Two-stage training
config.yaml
```shell
train.dataset = nnaa
train.model = fine-tune
```

## Model
config.yaml
```shell
train.model = PharmHGT # or fine-tune, HGT
```
## Train
```shell
python pretrain_masking.py
```

