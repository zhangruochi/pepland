"""Opt-in real-checkpoint CPU tests (no mocked model).

Run with PEPLAND_CHECKPOINT_TESTS=1 using the project inference dependencies.
The default checkpoint is the repository's inference/cpkt/model MLflow artifact.
"""
import importlib.util
import os
from pathlib import Path
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
ENABLED = os.environ.get("PEPLAND_CHECKPOINT_TESTS") == "1"


@unittest.skipUnless(ENABLED, "set PEPLAND_CHECKPOINT_TESTS=1 for real-checkpoint tests")
class CheckpointReadoutTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import torch
        cls.torch = torch
        torch.set_num_threads(1)
        spec = importlib.util.spec_from_file_location(
            "pepland", ROOT / "__init__.py", submodule_search_locations=[str(ROOT)])
        package = importlib.util.module_from_spec(spec)
        sys.modules["pepland"] = package
        spec.loader.exec_module(package)
        from pepland.model.core import PepLandFeatureExtractor, PropertyPredictor, Node_GRU
        from pepland.utils.process import Mol2HeteroGraph
        from pepland.utils.readout import pool_atom_fragment
        cls.extractor = PepLandFeatureExtractor(
            str(ROOT / "inference" / "cpkt" / "model"), pooling="avg").cpu().eval()
        cls.Predictor, cls.NodeGRU = PropertyPredictor, Node_GRU
        cls.graph_factory = staticmethod(Mol2HeteroGraph)
        cls.pool = staticmethod(pool_atom_fragment)
        examples = list(dict.fromkeys(line.strip() for line in (ROOT / "data" / "example.smi").read_text().splitlines() if line.strip()))
        # example.smi contains only two distinct peptides; include diglycine.
        cls.smiles = examples[:2] + ["NCC(=O)NCC(=O)O"]

    def graphs(self, indices):
        # Model forward mutates node data: never reuse an inferred graph.
        return [self.graph_factory(self.smiles[i]) for i in indices]

    def assertClose(self, actual, expected):
        self.torch.testing.assert_close(actual, expected, rtol=2e-5, atol=2e-5)

    def test_node_features_before_pooling_and_joint_pooling(self):
        t = self.torch
        with t.no_grad():
            single = [self.extractor.extract_atom_fragment_embedding(self.graphs([i]), return_counts=True)
                      for i in range(3)]
            for order in ([0, 1, 2], [2, 0, 1], [1, 0], [2]):
                a, f, ac, fc = self.extractor.extract_atom_fragment_embedding(self.graphs(order), return_counts=True)
                for row, index in enumerate(order):
                    sa, sf, sac, sfc = single[index]
                    self.assertEqual(ac[row].item(), sac[0].item())
                    self.assertEqual(fc[row].item(), sfc[0].item())
                    self.assertClose(a[row, :ac[row]], sa[0, :sac[0]])
                    self.assertClose(f[row, :fc[row]], sf[0, :sfc[0]])
                for mode in ("avg", "max"):
                    expected = t.cat([self.pool(*single[i], pooling=mode) for i in order])
                    self.assertClose(self.pool(a, f, ac, fc, pooling=mode), expected)

    def test_extractor_forward_and_legacy(self):
        t = self.torch
        extractor = self.extractor
        from pepland.utils.commons import Permute, Squeeze
        with t.no_grad():
            for mode in ("avg", "max"):
                extractor.pooling = mode
                reducer = t.nn.AdaptiveAvgPool1d(1) if mode == "avg" else t.nn.AdaptiveMaxPool1d(1)
                extractor.pooling_layer = t.nn.Sequential(Permute(), reducer, Squeeze(-1))
                extractor.padding_mode = "exclude"
                single = t.cat([extractor(self.graphs([i])) for i in range(3)])
                for order in ([0, 1, 2], [2, 0, 1], [1, 0]):
                    self.assertClose(extractor(self.graphs(order)), single[order])
                extractor.padding_mode = "legacy"
                a, f = extractor.extract_atom_fragment_embedding(self.graphs([0, 1, 2]))
                joined = t.cat([a, f], dim=1)
                expected = joined.mean(1) if mode == "avg" else joined.max(1).values
                self.assertClose(extractor(self.graphs([0, 1, 2])), expected)
                # Attribute fallback only: this is not an old serialized artifact.
                saved_padding, saved_pooling = extractor.padding_mode, extractor.pooling
                del extractor.padding_mode
                del extractor.pooling
                try:
                    self.assertClose(extractor(self.graphs([0, 1, 2])), expected)
                finally:
                    extractor.padding_mode, extractor.pooling = saved_padding, saved_pooling
        extractor.padding_mode = "exclude"

    def test_empty_input(self):
        with self.assertRaises(ValueError):
            self.extractor([])

    def test_property_predictor_real_checkpoint(self):
        predictor = self.Predictor(str(ROOT / "inference" / "cpkt" / "model"), hidden_dims=[4]).cpu().eval()
        with self.torch.no_grad():
            single = self.torch.cat([predictor(self.graphs([i])) for i in range(3)])
            self.assertClose(predictor(self.graphs([2, 0, 1])), single[[2, 0, 1]])
        with self.assertRaises(ValueError):
            predictor([])
        legacy_predictor = self.Predictor(
            str(ROOT / "inference" / "cpkt" / "model"), hidden_dims=[4],
            padding_mode="legacy").cpu().eval()
        self.assertEqual(legacy_predictor.feature_model.padding_mode, "legacy")

    def test_real_gru_lengths_legacy_and_parameter_shapes(self):
        t = self.torch
        t.manual_seed(12)
        gru = self.NodeGRU(hid_dim=2).cpu().eval()
        a, f = t.randn(2, 4, 2), t.randn(2, 3, 2)
        ac, fc = t.tensor([1, 4]), t.tensor([3, 1])
        invalid = [(t.tensor([1]), fc), (t.tensor([99, 99]), fc),
                   (t.tensor([1., 4.]), fc), (None, fc), (ac, None),
                   (t.tensor([0, 4]), fc)]
        for bad_ac, bad_fc in invalid:
            with self.subTest(atom_counts=bad_ac, frag_counts=bad_fc), self.assertRaises(ValueError):
                gru(a, f, bad_ac, bad_fc)
        shapes = {key: tuple(value.shape) for key, value in gru.state_dict().items()}
        # Original hid_dim=2 bidirectional GRU parameter contract, not merely
        # comparison with another instance of the current implementation.
        historical_shapes = {"projection.weight": (2, 8), "projection.bias": (2,)}
        for name in ("atom_gru", "frag_gru"):
            for suffix in ("", "_reverse"):
                historical_shapes.update({
                    f"{name}.weight_ih_l0{suffix}": (6, 2),
                    f"{name}.weight_hh_l0{suffix}": (6, 2),
                    f"{name}.bias_ih_l0{suffix}": (6,),
                    f"{name}.bias_hh_l0{suffix}": (6,),
                })
        self.assertEqual(shapes, historical_shapes)

        with t.no_grad():
            actual = gru(a, f, ac, fc)
            expected = t.cat([gru(a[i:i+1, :ac[i]], f[i:i+1, :fc[i]]) for i in range(2)])
            self.assertClose(actual, expected)
            ao, _ = gru.atom_gru(a)
            fo, _ = gru.frag_gru(f)
            legacy = gru.projection(t.cat([ao[:, -1], fo[:, -1]], dim=1))
            t.testing.assert_close(gru(a, f), legacy, rtol=0, atol=0)
            self.assertClose(gru(a.flip(0), f.flip(0), ac.flip(0), fc.flip(0)), actual.flip(0))
        self.assertEqual(shapes, {key: tuple(value.shape) for key, value in gru.state_dict().items()})
        clone = self.NodeGRU(hid_dim=2)
        clone.load_state_dict(gru.state_dict(), strict=True)


if __name__ == "__main__":
    unittest.main()
