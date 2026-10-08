"""Actual edge permutation helpers work for empty pairs and autograd."""
import ast
from pathlib import Path
import unittest

import torch

ROOT = Path(__file__).resolve().parents[1]


class ReverseEdgeTests(unittest.TestCase):
    def test_all_runtime_and_archived_helpers(self):
        paths = [ROOT / "model/model.py", *ROOT.glob("cpkt/*/code/model/model.py"),
                 *ROOT.glob("inference/cpkt/*/code/model/model.py")]
        self.assertEqual(len(paths), 9)
        for path in paths:
            with self.subTest(source=str(path.relative_to(ROOT))):
                function = next(node for node in ast.parse(path.read_text()).body
                                if isinstance(node, ast.FunctionDef) and node.name == "reverse_edge")
                namespace = {"torch": torch}
                exec(compile(ast.Module(body=[function], type_ignores=[]), str(path), "exec"), namespace)
                reverse = namespace["reverse_edge"]
                empty = torch.empty((0, 3), requires_grad=True)
                result = reverse(empty)
                self.assertEqual(result.shape, (0, 3))
                self.assertEqual(torch.autograd.grad(result.sum(), empty)[0].shape, (0, 3))
                nodes = torch.arange(12., dtype=torch.float32).reshape(4, 3).requires_grad_()
                expected = nodes[[1, 0, 3, 2]]
                torch.testing.assert_close(reverse(nodes), expected, rtol=0, atol=0)
                torch.testing.assert_close(reverse(reverse(nodes)), nodes, rtol=0, atol=0)
                weights = torch.arange(12.).reshape(4, 3)
                gradient = torch.autograd.grad((reverse(nodes) * weights).sum(), nodes)[0]
                torch.testing.assert_close(gradient, weights[[1, 0, 3, 2]], rtol=0, atol=0)
                with self.assertRaises(ValueError):
                    reverse(torch.zeros(3, 2))
