"""Opt-in actual MLflow checkpoint CUDA tests; no training or property accuracy.

Set PEPLAND_GPU_CHECKPOINT_TESTS=1 on one idle GPU. Enabled tests fail if CUDA
is unavailable or the selected GPU has foreign compute PIDs before allocation.
"""
import copy
import importlib.util
import os
from pathlib import Path
import subprocess
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
ENABLED = os.environ.get("PEPLAND_GPU_CHECKPOINT_TESTS") == "1"
RTOL = ATOL = 1e-4


def require_idle_gpu():
    rows = subprocess.check_output(["nvidia-smi", "--query-gpu=index,uuid", "--format=csv,noheader,nounits"], text=True, timeout=15)
    identities = dict(tuple(part.strip() for part in row.split(",", 1)) for row in rows.splitlines() if row.strip())
    selected = os.environ.get("CUDA_VISIBLE_DEVICES", "0").split(",")[0].strip()
    identity = identities.get(selected, selected)
    if identity not in identities.values():
        raise RuntimeError("Cannot resolve selected visible GPU: " + selected)
    rows = subprocess.check_output(["nvidia-smi", "--query-compute-apps=gpu_uuid,pid", "--format=csv,noheader,nounits"], text=True, timeout=15)
    foreign = []
    for row in rows.splitlines():
        if row.strip():
            gpu, pid = [part.strip() for part in row.split(",", 1)]
            if gpu == identity and int(pid) != os.getpid():
                foreign.append(pid)
    if foreign:
        raise RuntimeError("Selected GPU has foreign compute PIDs; refusing allocation: " + ",".join(foreign))


@unittest.skipUnless(ENABLED, "set PEPLAND_GPU_CHECKPOINT_TESTS=1 on one idle CUDA GPU")
class GPUCheckpointTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        require_idle_gpu()  # No CUDA tensor allocation occurs before this check.
        import torch
        if not torch.cuda.is_available():
            raise RuntimeError("GPU checkpoint tests enabled but CUDA is unavailable")
        torch.set_num_threads(1)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        cls.torch = torch
        spec = importlib.util.spec_from_file_location("pepland", ROOT / "__init__.py", submodule_search_locations=[str(ROOT)])
        package = importlib.util.module_from_spec(spec)
        sys.modules["pepland"] = package
        spec.loader.exec_module(package)
        from pepland.model.core import PepLandFeatureExtractor
        from pepland.utils.process import Mol2HeteroGraph
        from pepland.utils.readout import pool_atom_fragment
        cls.graph_factory = staticmethod(Mol2HeteroGraph)
        cls.pool = staticmethod(pool_atom_fragment)
        path = str(ROOT / "inference" / "cpkt" / "model")
        cls.cpu = PepLandFeatureExtractor(path, pooling="avg").cpu().eval()
        cls.gpu = PepLandFeatureExtractor(path, pooling="avg").cpu().eval()
        # CPU artifact loading takes time; recheck ownership before allocation.
        require_idle_gpu()
        cls.gpu.cuda()
        if not all(param.is_cuda for param in cls.gpu.model.parameters()):
            raise AssertionError("Checkpoint parameters must reside on CUDA")
        cls.smiles = list(dict.fromkeys(line.strip() for line in (ROOT / "data" / "example.smi").read_text().splitlines() if line.strip()))[:2]
        cls.smiles += ["NCC(=O)O", "NCC(=O)NCC(=O)O", "C", "[NH4+]"]
        cls.original_relations = {name: tuple(module.homo_etypes) for name, module in cls.gpu.model.named_modules() if hasattr(module, "homo_etypes")}
        def assert_cuda_graph(module, args):
            graph = args[0]
            for kind in graph.ntypes:
                if any(not graph.nodes[kind].data[key].is_cuda for key in graph.nodes[kind].data.keys()):
                    raise AssertionError("Backbone input node features must reside on CUDA")
            for kind in graph.canonical_etypes:
                if any(not graph.edges[kind].data[key].is_cuda for key in graph.edges[kind].data.keys()):
                    raise AssertionError("Backbone input edge features must reside on CUDA")
        cls.cuda_graph_hook = cls.gpu.model.register_forward_pre_hook(assert_cuda_graph)

    @classmethod
    def tearDownClass(cls):
        cls.cuda_graph_hook.remove()
        del cls.gpu
        cls.torch.cuda.empty_cache()

    def graphs(self, order):
        return [self.graph_factory(self.smiles[i]) for i in order]

    def close(self, actual, expected):
        self.torch.testing.assert_close(actual.detach().cpu(), expected.detach().cpu(), rtol=RTOL, atol=ATOL)

    def cuda_finite(self, value):
        self.assertTrue(value.is_cuda)
        self.assertTrue(self.torch.isfinite(value).all().item())

    def extract(self, model, order):
        graphs = self.graphs(order)  # Fresh graphs for every inference call.
        snapshots = [(graph, {kind: {key: graph.nodes[kind].data[key].clone() for key in graph.nodes[kind].data.keys()} for kind in graph.ntypes},
                      {kind: {key: graph.edges[kind].data[key].clone() for key in graph.edges[kind].data.keys()} for kind in graph.canonical_etypes}) for graph in graphs]
        relations = {name: tuple(module.homo_etypes) for name, module in model.model.named_modules() if hasattr(module, "homo_etypes")}
        result = model.extract_atom_fragment_embedding(graphs, return_counts=True)
        after = {name: tuple(module.homo_etypes) for name, module in model.model.named_modules() if hasattr(module, "homo_etypes")}
        self.assertEqual(relations, after)
        for graph, nodes, edges in snapshots:
            for kind, values in nodes.items():
                self.assertEqual(set(values), set(graph.nodes[kind].data.keys()))
                for key, value in values.items():
                    self.torch.testing.assert_close(graph.nodes[kind].data[key], value, rtol=0, atol=0)
            for kind, values in edges.items():
                self.assertEqual(set(values), set(graph.edges[kind].data.keys()))
                for key, value in values.items():
                    self.torch.testing.assert_close(graph.edges[kind].data[key], value, rtol=0, atol=0)
        return result

    def test_cuda_node_features_joint_pooling_and_cpu_parity(self):
        t = self.torch
        self.gpu.padding_mode = "exclude"
        with t.no_grad():
            singles = [self.extract(self.gpu, [i]) for i in range(len(self.smiles))]
            for order in ([0, 1, 2, 3, 4, 5], [5, 3, 1, 4, 2, 0], [4, 0], [2], [5]):
                a, f, ac, fc = self.extract(self.gpu, order)
                ca, cf, cac, cfc = self.extract(self.cpu, order)
                self.cuda_finite(a)
                self.cuda_finite(f)
                self.assertTrue(t.equal(ac.cpu(), cac.cpu()))
                self.assertTrue(t.equal(fc.cpu(), cfc.cpu()))
                for row, index in enumerate(order):
                    sa, sf, sac, sfc = singles[index]
                    self.assertEqual(ac[row].item(), sac[0].item())
                    self.assertEqual(fc[row].item(), sfc[0].item())
                    self.close(a[row, :ac[row]], sa[0, :sac[0]])
                    self.close(f[row, :fc[row]], sf[0, :sfc[0]])
                    self.close(a[row, :ac[row]], ca[row, :cac[row]])
                    self.close(f[row, :fc[row]], cf[row, :cfc[row]])
                for mode in ("avg", "max"):
                    pooled = self.pool(a, f, ac, fc, pooling=mode)
                    self.cuda_finite(pooled)
                    direct = []
                    for row in range(len(order)):
                        real = t.cat([a[row, :ac[row]], f[row, :fc[row]]])
                        direct.append(real.mean(0) if mode == "avg" else real.max(0).values)
                    self.close(pooled, t.stack(direct))
                    self.close(pooled, t.cat([self.pool(*singles[i], pooling=mode) for i in order]))
                    self.close(pooled, self.pool(ca, cf, cac, cfc, pooling=mode))
        current = {name: tuple(module.homo_etypes) for name, module in self.gpu.model.named_modules() if hasattr(module, "homo_etypes")}
        self.assertEqual(current, self.original_relations)

    def test_actual_api_cuda_forward_and_legacy(self):
        from pepland.utils.commons import Permute, Squeeze
        t = self.torch
        try:
            with t.no_grad():
                for mode in ("avg", "max"):
                    reducer = t.nn.AdaptiveAvgPool1d(1) if mode == "avg" else t.nn.AdaptiveMaxPool1d(1)
                    self.gpu.pooling, self.gpu.pooling_layer = mode, t.nn.Sequential(Permute(), reducer, Squeeze(-1)).cuda()
                    self.gpu.padding_mode = "exclude"
                    singles = t.cat([self.gpu(self.graphs([i])) for i in range(len(self.smiles))])
                    order = [5, 2, 0, 4, 3, 1]
                    actual = self.gpu(self.graphs(order))
                    self.cuda_finite(actual)
                    self.close(actual, singles[order])
                    self.gpu.padding_mode = "legacy"
                    # Reduce legacy's own historical backbone features.
                    a, f, _, _ = self.extract(self.gpu, order)
                    joined = t.cat([a, f], dim=1)
                    reference = joined.mean(1) if mode == "avg" else joined.max(1).values
                    legacy = self.gpu(self.graphs(order))
                    self.cuda_finite(legacy)
                    self.close(legacy, reference)
        finally:
            self.gpu.padding_mode = "exclude"

    def head_features(self, name, a, f, ac, fc, row, index):
        if name == "linear_pred_pharms":
            return f[row, :fc[row]]
        atom = a[row, :ac[row]]
        if name == "linear_pred_atoms":
            return atom
        source, target = self.graph_factory(self.smiles[index]).edges(etype="b")
        # trainer.py's saved bond-head contract: sum endpoint embeddings.
        return atom[source.cuda()] + atom[target.cuda()]

    def test_three_saved_reconstruction_heads_cuda_strict_and_batch_parity(self):
        import mlflow.pytorch
        t = self.torch
        dimensions = {"linear_pred_atoms": 119, "linear_pred_bonds": 4, "linear_pred_pharms": 264}
        try:
            with t.no_grad():
                for name, out_dim in dimensions.items():
                    artifact = str(ROOT / "inference" / "cpkt" / name)
                    cpu_head = mlflow.pytorch.load_model(artifact, map_location="cpu").eval()
                    gpu_head = mlflow.pytorch.load_model(artifact, map_location="cpu").cuda().eval()
                    clone = copy.deepcopy(cpu_head)
                    clone.load_state_dict(gpu_head.state_dict(), strict=True)
                    self.assertEqual(cpu_head.in_features, 300)
                    self.assertEqual(cpu_head.out_features, out_dim)
                    self.assertTrue(all(param.is_cuda for param in gpu_head.parameters()))
                    self.gpu.padding_mode = "exclude"
                    batch = self.extract(self.gpu, [0, 3])
                    for row, index in enumerate([0, 3]):
                        single = self.extract(self.gpu, [index])
                        features = self.head_features(name, *batch, row, index)
                        singleton = self.head_features(name, *single, 0, index)
                        output = gpu_head(features)
                        self.cuda_finite(output)
                        self.assertEqual(tuple(output.shape), (features.shape[0], out_dim))
                        self.close(output, gpu_head(singleton))
                        self.close(output, cpu_head(features.cpu()))
                    self.gpu.padding_mode = "legacy"
                    single = self.extract(self.gpu, [0])
                    legacy_features = self.head_features(name, *single, 0, 0)
                    legacy_output = gpu_head(legacy_features)
                    self.cuda_finite(legacy_output)
                    self.close(legacy_output, cpu_head(legacy_features.cpu()))
                    del gpu_head
        finally:
            self.gpu.padding_mode = "exclude"


if __name__ == "__main__":
    unittest.main()
