# Frozen embeddings with supervised regression

This example fits a fixed-alpha Ridge head on frozen PepLand embeddings. It is
an ordinary usage example, not the paper's downstream training recipe or a
validated property model. It does not fine-tune the backbone.

Use the repository's [tested CPU environment](../README.md#reproducible-validation)
and a checkout named `pepland`, as required by the existing package API. Run from
the checkout root:

```sh
python examples/frozen_regression.py \
  --csv data/eval/nc-CPP.csv --smiles-column SMILES --target-column PAMPA \
  --drop-missing-target --max-molecules 40 --seed 0 \
  --output outputs/frozen-regression
```

This uses the public CSV's PAMPA column as supplied; it does not mix different
assay columns, convert units, or establish the data's measurement conventions.
The 40-molecule subset is a bounded interface demonstration. Its scores are not
evidence of predictive quality. Use an appropriate, documented target and
evaluation design for your own task.

For another CSV, explicitly select its SMILES and finite numeric target columns.
Pretraining CSVs contain only `smiles` and cannot be used as supervised labels.
Missing targets are dropped only with `--drop-missing-target`; malformed or
non-finite values still fail. Exact duplicate molecule/target/group records are
collapsed; repeated measurements with different targets stay together.

## Split and fitting contract

- Canonical isomeric SMILES identify molecules before sampling or splitting.
  The fixed seed operates on sorted components, independent of CSV row order.
- `--group-column` additionally keeps every supplied group in one split.
  Molecule and group links are combined transitively. Missing group IDs fail.
  The molecule limit keeps whole components; too few independent components fail.
- Train, validation and test must be nonempty and molecule/group disjoint.
  Related scaffolds, assay sources and unknown backbone-pretraining overlap are
  not controlled by molecular identity alone. Supply appropriate groups when
  needed; a valid split here is not proof against every form of leakage.
- Embeddings use CPU `eval()`, `no_grad()`, average pooling and
  `padding_mode="exclude"`. Backbone parameters are frozen.
- Feature centering/scaling, the fixed-alpha regression head and the mean-target
  baseline are fitted on training rows only. Test results select no parameters.
  `--alpha` is a configuration input, not an automatic tuning procedure.

The public `further_training` example files and the small archive linked from the
[maintainer's data announcement](https://github.com/zhangruochi/pepland/issues/9#issuecomment-3222235334)
have identical train/validation/test molecule sets. They illustrate the unlabeled
pretraining format and must not be treated as an independent held-out benchmark.
The external data's completeness, provenance and usage terms need separate
verification; this example does not redistribute that corpus.

## Outputs and verification

A new output directory contains `model.npz`, `config.json` and `metrics.json`.
Existing output directories are refused. Outputs under `outputs/` are ignored
by Git. The non-pickled model archive records feature dimension, pooling convention and
checkpoint fingerprint; it is reloaded to verify identical predictions. Use the
same checkpoint and extraction settings when applying the saved head. MAE and
RMSE are reported alongside the training-mean baseline, without an improvement
claim.

From the repository root:

```sh
python -m pytest -q test/test_frozen_regression.py
PEPLAND_CHECKPOINT_TESTS=1 python -m pytest -q test/test_frozen_regression.py
```

The ordinary tests use synthetic labels and injected features for interface and
leakage checks. The opt-in test separately exercises the real bundled CPU
checkpoint, parameter immutability, batch consistency and saved-head reload.
