# PepLand Supervised Minimal

This repository is a slimmed supervised-learning version of PepLand. It keeps
only the SMILES-to-heterograph workflow, the PepLand encoder, and a simple
binary classification training script.

## Files

- `train_supervised.py`: supervised binary training entrypoint.
- `model/data.py`: SMILES/RDKit molecule to PepLand heterogeneous graph.
- `model/model.py`: PepLand encoder.
- `model/supervised.py`: supervised dataset, collate function, pooling, and classifier head.
- `tokenizer/pep2fragments.py`: peptide fragmentation logic used by graph construction.
- `tokenizer/vocabs/`: fragment vocabularies.
- `environment.supervised.yaml`: minimal conda environment for this workflow.

## Data

Prepare two CSV files with at least these columns:

```csv
smiles,label
CC(C)C...,1
O=C(N...)...,0
```

`label` is expected to be binary: `0` or `1`.

## Configure

Edit the `CONFIG` block in `train_supervised.py`:

```python
"train_csv": "path/to/train.csv",
"test_csv": "path/to/test.csv",
"num_runs": 5,
"epochs": 20,
```

At runtime, 10% of `train_csv` is split into validation data using a
reproducible seed. The remaining 90% is used for training.

## Run

```bash
conda env create -f environment.supervised.yaml
conda activate pepland-supervised
python train_supervised.py
```

The script reports ACC, AUPRC, F1, Precision, and Recall for each run, then
prints the mean and standard deviation across runs.
