"""Opt-in bundled pretraining reconstruction artifact checks.

These saved atom/bond/pharmacophore heads are not downstream property heads.
"""
import importlib.util
import os
from pathlib import Path
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]


@unittest.skipUnless(os.environ.get("PEPLAND_CHECKPOINT_TESTS") == "1",
                     "set PEPLAND_CHECKPOINT_TESTS=1 for saved artifact tests")
class SavedPretrainingHeadTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        spec = importlib.util.spec_from_file_location(
            "pepland", ROOT / "__init__.py", submodule_search_locations=[str(ROOT)])
        package = importlib.util.module_from_spec(spec)
        sys.modules["pepland"] = package
        spec.loader.exec_module(package)

    def test_saved_heads_load_state_and_real_node_batch_outputs(self):
        import mlflow.pytorch
        import torch

        torch.set_num_threads(1)
        spec = importlib.util.spec_from_file_location(
            "pepland", ROOT / "__init__.py", submodule_search_locations=[str(ROOT)])
        package = importlib.util.module_from_spec(spec)
        sys.modules["pepland"] = package
        spec.loader.exec_module(package)
        from pepland.model.core import PepLandFeatureExtractor
        from pepland.utils.process import Mol2HeteroGraph

        extractor = PepLandFeatureExtractor(str(ROOT / "inference/cpkt/model")).cpu().eval()
        smiles = ["NCC(=O)NCC(=O)O", "NCC(=O)O"]

        def features(order):
            return extractor.extract_atom_fragment_embedding(
                [Mol2HeteroGraph(smiles[i]) for i in order], return_counts=True)

        with torch.no_grad():
            singles = [features([i]) for i in range(2)]
            for name, size, kind in (("atoms", 119, 0), ("bonds", 4, 0), ("pharms", 264, 1)):
                with self.subTest(head=name):
                    saved = mlflow.pytorch.load_model(
                        str(ROOT / "cpkt" / ("linear_pred_" + name)), map_location="cpu").eval()
                    inference_saved = mlflow.pytorch.load_model(
                        str(ROOT / "inference/cpkt" / ("linear_pred_" + name)), map_location="cpu").eval()
                    self.assertIsInstance(saved, torch.nn.Linear)
                    self.assertEqual((saved.in_features, saved.out_features), (300, size))
                    state = saved.state_dict()
                    self.assertEqual(set(state), set(inference_saved.state_dict()))
                    for key in state:
                        torch.testing.assert_close(state[key], inference_saved.state_dict()[key], rtol=0, atol=0)
                    restored = torch.nn.Linear(300, size).eval()
                    restored.load_state_dict(state, strict=True)
                    for order in ([0, 1], [1, 0], [0], [1]):
                        batch = features(order)
                        for row, index in enumerate(order):
                            count = batch[kind + 2][row].item()
                            nodes = batch[kind][row, :count]
                            # Use actual graph bond endpoints; this checks saved
                            # head invocation, not reconstruction accuracy.
                            if name == "bonds":
                                source, target = Mol2HeteroGraph(smiles[index]).edges(etype=("a", "b", "a"))
                                nodes = nodes[source] + nodes[target]
                            expected_nodes = singles[index][kind][0, :count]
                            if name == "bonds":
                                expected_nodes = expected_nodes[source] + expected_nodes[target]
                            actual = saved(nodes)
                            self.assertEqual(actual.shape, (nodes.shape[0], size))
                            self.assertTrue(torch.isfinite(actual).all())
                            torch.testing.assert_close(actual, saved(expected_nodes), rtol=2e-5, atol=2e-5)
                            torch.testing.assert_close(actual, restored(nodes), rtol=0, atol=0)

    def test_original_singletons_and_fresh_legacy_batch_contract(self):
        import copy
        import types
        import torch
        from pepland.model.core import PepLandFeatureExtractor
        from pepland.utils.process import Mol2HeteroGraph

        path = str(ROOT / "inference/cpkt/model")
        smiles = ["NCC(=O)NCC(=O)O", "NCC(=O)O"]
        template = PepLandFeatureExtractor(path).cpu().eval()
        fixture = (ROOT / "test/fixtures/mvmp_original_forward.txt").read_text()

        def old_model():
            model = copy.deepcopy(template.model)
            for module in model.modules():
                if module.__class__.__name__ == "MVMP":
                    namespace = dict(module.forward.__func__.__globals__)
                    exec(compile(fixture, "frozen_original_mvmp", "exec"), namespace)
                    module.forward = types.MethodType(namespace["legacy_mvmp_forward"], module)
            return model

        with torch.no_grad():
            for order in ([0], [1], [0, 1], [1, 0]):
                original = old_model()
                import dgl
                expected = original(dgl.batch([Mol2HeteroGraph(smiles[i]) for i in order]))
                legacy = PepLandFeatureExtractor(path, padding_mode="legacy").cpu().eval()
                actual = legacy.extract_atom_fragment_embedding(
                    [Mol2HeteroGraph(smiles[i]) for i in order], return_counts=True)
                for kind in (0, 1):
                    rows = torch.cat([actual[kind][row, :actual[kind + 2][row].item()]
                                      for row in range(len(order))])
                    torch.testing.assert_close(rows, expected[kind], rtol=2e-5, atol=2e-5)
                if len(order) == 1:
                    corrected = template.extract_atom_fragment_embedding(
                        [Mol2HeteroGraph(smiles[order[0]])], return_counts=True)
                    for kind in (0, 1):
                        torch.testing.assert_close(corrected[kind][0, :corrected[kind + 2][0].item()],
                                                   expected[kind], rtol=2e-5, atol=2e-5)
            relation_list = list(template.model.mp.homo_etypes)
            template.extract_atom_fragment_embedding([Mol2HeteroGraph(smiles[1])])
            self.assertEqual(template.model.mp.homo_etypes, relation_list)
            template.padding_mode = "legacy"
            template.extract_atom_fragment_embedding([Mol2HeteroGraph(smiles[0])])
            self.assertEqual(template.model.mp.empty_relation_mode, "legacy")
            template.padding_mode = "exclude"
            template.extract_atom_fragment_embedding([Mol2HeteroGraph(smiles[0])])
            self.assertEqual(template.model.mp.empty_relation_mode, "per_graph")

    def test_edgeless_atom_and_fragment_graphs_singleton_mixed(self):
        import torch
        from pepland.model.core import PepLandFeatureExtractor
        from pepland.utils.process import Mol2HeteroGraph

        extractor = PepLandFeatureExtractor(str(ROOT / "inference/cpkt/model")).cpu().eval()
        smiles = ["NCC(=O)NCC(=O)O", "C", "[NH4+]"]

        def features(order):
            return extractor.extract_atom_fragment_embedding(
                [Mol2HeteroGraph(smiles[i]) for i in order], return_counts=True)

        with torch.no_grad():
            singles = [features([i]) for i in range(3)]
            relations = list(extractor.model.mp.homo_etypes)
            for order in ([0, 1, 2], [2, 1, 0], [1, 0], [2, 0], [1], [2]):
                actual = features(order)
                self.assertEqual(extractor.model.mp.homo_etypes, relations)
                for row, index in enumerate(order):
                    for kind in (0, 1):
                        count = actual[kind + 2][row].item()
                        self.assertEqual(count, singles[index][kind + 2][0].item())
                        nodes = actual[kind][row, :count]
                        self.assertTrue(torch.isfinite(nodes).all())
                        torch.testing.assert_close(nodes, singles[index][kind][0, :count],
                                                   rtol=2e-5, atol=2e-5)
