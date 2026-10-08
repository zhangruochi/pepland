# PepLand Inference

Generate peptide embeddings using the pretrained PepLand model.

## Quick Start

### 1. Create Conda Environment

```bash
conda env create -f environment.yaml
conda activate pepland-inference
```

### 2. Prepare Input Data

Create a `.smi` file with one SMILES string per line:

```
OC(=O)CC[C@@H](C(=O)N[C@@H](Cc1ccccc1)C...
[NH3+]CCCC[C@@H](C(=O)N[C@H](C(=O)N[C@H]...
```

### 3. Configure Inference

Edit `configs/inference.yaml`:

```yaml
mode:
  ddp: false

inference:
  device_ids: [0]              # GPU device IDs, empty [] for CPU
  data: '../data/example.smi'  # Path to input SMILES file
  model_path: "./cpkt/model"   # Path to model checkpoint
  pool: avg                    # Pooling method: avg or max
  atom_index: false            # false for peptide embedding, or index for atom embedding
```

### 4. Run Inference

```bash
cd inference
python inference_pepland.py
```

## Output

- **Peptide Embedding**: Shape `(N, 300)` where N is the number of input SMILES
- **Atom Embedding**: If `atom_index` is set, returns embedding for specific atom position

## Environment Details

| Package | Version | Purpose |
|---------|---------|---------|
| Python | 3.8 | Runtime |
| PyTorch | 1.11.0+cu113 | Deep Learning |
| DGL | 0.9.1 (CUDA 11.3) | Graph Neural Networks |
| RDKit | 2024.x | Molecule Processing |
| MLflow | 1.30.0 | Model Loading |
| OmegaConf | 2.2.x | Configuration |
| scikit-learn | 1.3.x | Model utilities |

## File Structure

```
inference/
├── environment.yaml       # Conda environment file
├── README.md              # This file
├── inference_pepland.py   # Main inference script
├── process.py             # Molecule to graph conversion
├── tokenizer/             # Peptide tokenization
│   ├── pep2fragments.py
│   └── vocabs/
│       └── Vocab_SIZE258.txt
└── cpkt/
    └── model/             # Pretrained model checkpoint
```

## Usage Example

```python
import torch
import dgl
from omegaconf import OmegaConf

# Import from inference directory
from process import Mol2HeteroGraph
from inference_pepland import load_model

# Load config and model
cfg = OmegaConf.load('../configs/inference.yaml')
model = load_model(cfg)
model.eval()

# Process SMILES
smiles = "CC(C)C[C@H](NC(=O)[C@H](CC(=O)O)NC(=O)C)C(=O)O"
graph = Mol2HeteroGraph(smiles)
bg = dgl.batch([graph])

# Get embeddings
with torch.no_grad():
    atom_embed, frag_embed = model(bg)
    print(f"Atom embedding shape: {atom_embed.shape}")      # (num_atoms, 300)
    print(f"Fragment embedding shape: {frag_embed.shape}")  # (num_fragments, 300)
```

## Batch Processing

```python
import torch
import dgl
import torch.nn as nn
from omegaconf import OmegaConf
from process import Mol2HeteroGraph
from inference_pepland import load_model, split_batch, pool_atom_fragment

# Load model
cfg = OmegaConf.load('../configs/inference.yaml')
model = load_model(cfg)
model.eval()
device = torch.device("cpu")

# Process multiple SMILES
smiles_list = [
    "CC(C)C[C@H](NC(=O)[C@H](CC(=O)O)NC(=O)C)C(=O)O",
    "CC[C@H](C)[C@H](NC(=O)CNC(=O)[C@H](CC(C)C)NC(=O)C)C(=O)O"
]

graphs = [Mol2HeteroGraph(smi) for smi in smiles_list]
bg = dgl.batch(graphs)

# Get embeddings with pooling
with torch.no_grad():
    atom_embed, frag_embed = model(bg)
    bg.nodes['a'].data['h'] = atom_embed
    bg.nodes['p'].data['h'] = frag_embed
    
    atom_rep = split_batch(bg, 'a', 'h', device)
    frag_rep = split_batch(bg, 'p', 'h', device)
    
    # Average over all real atom + fragment nodes, using graph counts.
    pep_embeds = pool_atom_fragment(
        atom_rep, frag_rep, bg.batch_num_nodes('a'), bg.batch_num_nodes('p'),
        pooling='avg', padding_mode='exclude'
    ).cpu().numpy()
    
    print(f"Peptide embeddings shape: {pep_embeds.shape}")  # (2, 300)
```

## Notes

- GPU inference requires mutually compatible Torch, CUDA and DGL builds
- For CPU-only inference, set `device_ids: []` in config
- Model outputs 300-dimensional embeddings
- The saved checkpoint records Torch 1.11.0; treat version warnings together with actual runtime validation, as described below

### Padding compatibility

Average/max readout now defaults to `padding_mode="exclude"`: the average is
`(sum(real atoms) + sum(real fragments)) / (A + F)`, preserving equal weight
per node; max ignores padding even for all-negative features. True zero feature
vectors remain real nodes. For fixed pre-pooling features, outputs are independent
of batch composition. This does not promise batch invariance for arbitrary models
in training mode (dropout, batch normalization, or other upstream operations).

Older readout divided by `Amax + Fmax` (the two maxima can belong to different
peptides). Thus corrected average features equal old features times
`(Amax + Fmax) / (A + F)` for the same node features. Downstream models trained on
old padded features may need retraining/recalibration; do not silently switch their
feature convention. Set `inference.padding_mode: legacy` in the config, or pass
`padding_mode="legacy"` to `PepLandFeatureExtractor`/`PropertyPredictor`, to retain
historical batch-dependent mean/max readout and absent-relation aggregation for a
fresh model call. Reproduce the old batch composition as well. Both modes now
avoid permanently removing a relation after an edgeless input. Default mode also
preserves singleton node state when another graph contributes a relation absent
from that molecule. Feature dimensions and pretrained backbone weights are unchanged.

The API's `gru` readout also uses real lengths by default; `legacy` retains its
old padded last-step behavior. It requires nonempty atom and fragment sequences.
`pooling=None` and `atom_index` continue to return padded node features. The public
`extract_atom_fragment_embedding` still returns two tensors by default; use
`return_counts=True` to also obtain atom and fragment counts. Empty batches or
graphs with no nodes of either type raise `ValueError` in average/max readout.
Old full serialized feature-extractor objects without `padding_mode` keep legacy
behavior; newly constructed objects use `exclude`. When loading a state dict,
choose the constructor's mode explicitly for the downstream feature convention.

### Readout regression tests

From the repository root, pure Torch tests need no model checkpoint, DGL, or
MLflow (their small fake graph checks exercise only splitting/readout):

```sh
CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=1 python -m unittest discover -s test -p test_readout.py -v
```

To explicitly run the bundled checkpoint and API checks in an environment with
compatible Torch, DGL, MLflow, RDKit and OmegaConf. IPython is optional
and is used only for notebook rendering:

```sh
PEPLAND_CHECKPOINT_TESTS=1 CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=1 python -m unittest discover -s test -p test_checkpoint_readout.py -v
```

These optional tests compare node features before pooling, then avg/max features
and PropertyPredictor outputs in CPU eval/no-grad mode with newly built graphs.
They cover three peptides, multiple batch sizes/orders, legacy readout and API
GRU lengths. They skip unless explicitly enabled; a skip is not checkpoint
verification. The bundled checkpoint's internal pretrained readout modules are
unchanged; its inference forward returns node embeddings before those modules.

Verified on CPU with Python 3.11, Torch 2.2.2+cpu, DGL 1.1.3, MLflow 2.22.2,
RDKit 2023.9.6 and NumPy 1.26.4. The checkpoint records Torch 1.11.0; its
version warning was present during these successful tests. Historical-runtime
and full GPU checkpoint inference remain unverified; pure Torch 1.11 readout and
CUDA helper checks are recorded in the [repository validation matrix](../README.md#reproducible-validation).
Run the complete suite with `PEPLAND_CHECKPOINT_TESTS=1 bash scripts/check.sh -q`
from a Git checkout. The GRU tests use the
real module with random parameters, PropertyPredictor uses an untrained head,
and old-object fallback is tested by simulating missing attributes rather than
loading a historical full serialized extractor artifact.
