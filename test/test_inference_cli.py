"""Configuration checks and opt-in real-checkpoint CLI regression tests."""
import importlib.util
import os
from pathlib import Path
import re
import subprocess
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location("inference_config", ROOT / "utils" / "inference_config.py")
_config = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_config)


class InferenceConfigTests(unittest.TestCase):
    def test_index_zero_is_not_false(self):
        self.assertEqual(_config.atom_index(0), 0)
        self.assertIsNone(_config.atom_index(False))
        self.assertIsNone(_config.atom_index(None))
        self.assertEqual(_config.atom_index([0, 2]), [0, 2])
        for invalid in (True, [], [False], 1.2, "0"):
            with self.subTest(invalid=invalid), self.assertRaisesRegex(ValueError, "atom_index"):
                _config.atom_index(invalid)

    def test_paths_canonical_legacy_and_missing(self):
        expected = (ROOT / "data" / "example.smi").resolve()
        self.assertEqual(_config.resolve_path("data/example.smi", ROOT, ROOT / "inference"), expected)
        self.assertEqual(_config.resolve_path("../data/example.smi", ROOT, ROOT / "inference"), expected)
        self.assertEqual(_config.resolve_path(expected, ROOT), expected)
        with self.assertRaisesRegex(FileNotFoundError, "Configured path does not exist"):
            _config.resolve_path("__nonexistent_issue12_data__", ROOT)


@unittest.skipUnless(os.environ.get("PEPLAND_CHECKPOINT_TESTS") == "1",
                     "set PEPLAND_CHECKPOINT_TESTS=1 for real-checkpoint CLI tests")
class InferenceCLITests(unittest.TestCase):
    scripts = (ROOT / "inference.py", ROOT / "inference" / "inference_pepland.py")

    def run_cli(self, script, data="data/example.smi", model="inference/cpkt/model", index="false"):
        with tempfile.TemporaryDirectory(prefix="pepland-cli-") as directory:
            cfg = Path(directory) / "inference.yaml"
            cfg.write_text("inference:\n  device_ids: []\n  data: " + repr(str(data)) +
                           "\n  model_path: " + repr(model) +
                           "\n  pool: avg\n  padding_mode: exclude\n  atom_index: " + index + "\n")
            env = dict(os.environ, CUDA_VISIBLE_DEVICES="", OMP_NUM_THREADS="1", DGLBACKEND="pytorch")
            result = subprocess.run([sys.executable, str(script), "--config", str(cfg)],
                                    cwd=directory, env=env, capture_output=True, text=True, timeout=180)
            return result

    def embeddings_text(self, result):
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        match = re.search(r"\(4, 300\)\s*(\[\[.*)", result.stdout, re.S)
        self.assertIsNotNone(match, result.stdout + result.stderr)
        return match.group(1).strip()

    def test_canonical_and_legacy_paths_both_entrypoints(self):
        # Root API canonicalizes SMILES; standalone preserves raw atom order.
        # Compare each established pipeline with itself, not across conventions.
        for script in self.scripts:
            canonical = self.embeddings_text(self.run_cli(script))
            legacy = self.embeddings_text(self.run_cli(script, "../data/example.smi", "./cpkt/model"))
            self.assertEqual(canonical, legacy)

    def test_atom_index_zero_distinct_from_pooling(self):
        # Root API canonicalizes SMILES; standalone preserves raw atom order.
        # Compare each established pipeline with itself, not across conventions.
        for script in self.scripts:
            pooled = self.embeddings_text(self.run_cli(script, index="false"))
            first_node = self.embeddings_text(self.run_cli(script, index="0"))
            self.assertNotEqual(first_node, pooled)
            self.assertEqual(first_node, self.embeddings_text(self.run_cli(script, index="0")))

    def test_invalid_index_and_no_valid_graphs_fail_clearly(self):
        for script in self.scripts:
            result = self.run_cli(script, index="true")
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("atom_index must", result.stderr)
            with tempfile.TemporaryDirectory(prefix="pepland-invalid-") as directory:
                data = Path(directory) / "invalid.smi"
                data.write_text("this-is-not-a-smiles\n")
                result = self.run_cli(script, data=data)
                self.assertNotEqual(result.returncode, 0)
                self.assertRegex(result.stderr, "ValueError:.*(no valid|Error processing)")
                data.write_text("")
                result = self.run_cli(script, data=data)
                self.assertNotEqual(result.returncode, 0)
                self.assertRegex(result.stderr, "ValueError:.*(no valid|nonempty)")

    def test_tokenizer_headless_without_ipython(self):
        # Import the real tokenizer with an import hook blocking notebook-only
        # dependencies. This does not mock either graph construction or checkpoint.
        code = '''import builtins, importlib.util, pathlib, sys
root = pathlib.Path(sys.argv[1])
original_import = builtins.__import__
def without_notebook(name, *args, **kwargs):
    if name == "IPython" or name.startswith("IPython."):
        raise ImportError("IPython deliberately unavailable for headless regression")
    return original_import(name, *args, **kwargs)
builtins.__import__ = without_notebook
spec = importlib.util.spec_from_file_location("pepland", root / "__init__.py", submodule_search_locations=[str(root)])
package = importlib.util.module_from_spec(spec)
sys.modules["pepland"] = package
spec.loader.exec_module(package)
import pepland.tokenizer.pep2fragments as tokenizer
assert tokenizer.IPythonConsole is None
assert tokenizer.SVG is None
from pepland.utils.process import Mol2HeteroGraph
graph = Mol2HeteroGraph("NCC(=O)NCC(=O)O")
assert graph.num_nodes("a") > 0
print("HEADLESS_TOKENIZER_OK")
'''
        with tempfile.TemporaryDirectory(prefix="pepland-headless-") as directory:
            result = subprocess.run([sys.executable, "-c", code, str(ROOT)], cwd=directory,
                                    env=dict(os.environ, CUDA_VISIBLE_DEVICES="", OMP_NUM_THREADS="1", DGLBACKEND="pytorch"),
                                    text=True, capture_output=True, timeout=90)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertIn("HEADLESS_TOKENIZER_OK", result.stdout)


if __name__ == "__main__":
    unittest.main()
