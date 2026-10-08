"""CPU-only readout regression tests; fake graphs are not checkpoint validation."""
import importlib.util
from pathlib import Path
import unittest

import torch

_spec = importlib.util.spec_from_file_location(
    "pepland_readout", Path(__file__).resolve().parents[1] / "utils" / "readout.py"
)
_readout = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_readout)
split_batch = _readout.split_batch
pool_atom_fragment = _readout.pool_atom_fragment


class FakeGraph:
    def __init__(self, features, counts):
        self.counts = torch.tensor(counts, dtype=torch.long)
        self.batch_size = len(counts)
        self.nodes = {"a": type("Nodes", (), {"data": {"h": features}})()}

    def batch_num_nodes(self, ntype):
        return self.counts


def padded(samples):
    width = max(len(x) for x in samples)
    result = torch.zeros(len(samples), width, samples[0].shape[-1], dtype=samples[0].dtype)
    for i, sample in enumerate(samples):
        result[i, :len(sample)] = sample
    return result


class ReadoutTests(unittest.TestCase):
    def assertTensor(self, actual, expected):
        torch.testing.assert_close(actual, expected, rtol=1e-6, atol=1e-7)

    def test_joint_weighting_cross_type_maxima_and_batch_invariance(self):
        atoms = [torch.tensor([[1., 2.], [3., 4.], [5., 6.]]),
                 torch.tensor([[7., 8.]]), torch.tensor([[0., 0.], [9., 10.]])]
        fragments = [torch.tensor([[11., 12.]]),
                     torch.tensor([[13., 14.], [15., 16.], [17., 18.], [19., 20.]]),
                     torch.tensor([[21., 22.], [23., 24.]])]
        for mode in ("avg", "max"):
            expected = torch.stack([getattr(torch.cat([a, f]), "mean" if mode == "avg" else "amax")(dim=0)
                                    for a, f in zip(atoms, fragments)])
            for order in ([0, 1, 2], [2, 0, 1], [1, 0], [0], [1], [2]):
                a, f = [atoms[i] for i in order], [fragments[i] for i in order]
                result = pool_atom_fragment(padded(a), padded(f),
                                            torch.tensor([len(x) for x in a]),
                                            torch.tensor([len(x) for x in f]), pooling=mode)
                self.assertTensor(result, expected[order])

    def test_equal_lengths_and_true_zero_nodes(self):
        a = torch.tensor([[[0., 0.], [6., 3.]], [[2., 4.], [4., 8.]]])
        f = torch.tensor([[[0., 0.]], [[3., 6.]]])
        self.assertTensor(pool_atom_fragment(a, f, torch.tensor([2, 2]), torch.tensor([1, 1])),
                          torch.tensor([[2., 1.], [3., 6.]]))

    def test_negative_max_padding_never_wins(self):
        a = torch.tensor([[[-4., -8.], [0., 0.]], [[-3., -7.], [-2., -6.]]])
        f = torch.tensor([[[-5., -9.], [-6., -10.]], [[-1., -5.], [0., 0.]]])
        self.assertTensor(pool_atom_fragment(a, f, torch.tensor([1, 2]), torch.tensor([2, 1]), pooling="max"),
                          torch.tensor([[-4., -8.], [-1., -5.]]))

    def test_legacy_matches_original_padded_reduction(self):
        a = torch.tensor([[[2.], [4.]], [[-2.], [0.]]])
        f = torch.tensor([[[6.], [0.], [0.]], [[-3.], [-4.], [-5.]]])
        for mode in ("avg", "max"):
            joined = torch.cat([a, f], dim=1)
            expected = joined.mean(1) if mode == "avg" else joined.amax(1)
            self.assertTensor(pool_atom_fragment(a, f, torch.tensor([2, 1]), torch.tensor([1, 3]),
                                                pooling=mode, padding_mode="legacy"), expected)

    def test_zero_nodes_of_one_type(self):
        a, f = torch.empty(2, 0, 2), torch.tensor([[[0., -1.]], [[2., 3.]]])
        for mode in ("avg", "max"):
            self.assertTensor(pool_atom_fragment(a, f, torch.tensor([0, 0]), torch.tensor([1, 1]), pooling=mode), f[:, 0])

    def test_gradients_exclude_padding_and_preserve_joint_weights(self):
        a = torch.tensor([[[2.], [99.]], [[3.], [4.]]], requires_grad=True)
        f = torch.tensor([[[5.]], [[6.]]], requires_grad=True)
        pool_atom_fragment(a, f, torch.tensor([1, 2]), torch.tensor([1, 1])).sum().backward()
        self.assertTensor(a.grad, torch.tensor([[[.5], [0.]], [[1/3], [1/3]]]))
        self.assertTensor(f.grad, torch.tensor([[[.5]], [[1/3]]]))

    def test_invalid_counts_and_empty_inputs(self):
        a, f = torch.ones(2, 2, 3), torch.ones(2, 1, 3)
        bad = [(torch.tensor([1]), torch.tensor([1, 1])),
               (torch.tensor([3, 1]), torch.tensor([1, 1])),
               (torch.tensor([-1, 1]), torch.tensor([1, 1])),
               (torch.tensor([1., 1.]), torch.tensor([1, 1])),
               (torch.tensor([[1, 1]]), torch.tensor([1, 1])),
               (torch.tensor([0, 1]), torch.tensor([0, 1]))]
        for ac, fc in bad:
            with self.subTest(ac=ac, fc=fc), self.assertRaises(ValueError):
                pool_atom_fragment(a, f, ac, fc)
        with self.assertRaises(ValueError):
            pool_atom_fragment(a[:0], f[:0], torch.tensor([], dtype=torch.long), torch.tensor([], dtype=torch.long))
        for kw in ({"pooling": "sum"}, {"padding_mode": "unknown"}):
            with self.assertRaises(ValueError):
                pool_atom_fragment(a, f, torch.tensor([1, 1]), torch.tensor([1, 1]), **kw)
        for aa, ff in ((a[0], f), (a, f[:1]), (a, torch.ones(2, 1, 4))):
            with self.assertRaises(ValueError):
                pool_atom_fragment(aa, ff, torch.tensor([1, 1]), torch.tensor([1, 1]))

    def test_split_batch_preserves_real_zero_and_gradients(self):
        h = torch.tensor([[0., 0.], [1., 2.], [3., 4.]], requires_grad=True)
        result = split_batch(FakeGraph(h, [1, 2]), "a", "h", device="cpu")
        self.assertTensor(result, torch.tensor([[[0., 0.], [0., 0.]], [[1., 2.], [3., 4.]]]))
        result.sum().backward()
        self.assertTensor(h.grad, torch.ones_like(h))

    def test_split_batch_zero_and_invalid_counts(self):
        h = torch.tensor([[1., 2.]])
        self.assertTensor(split_batch(FakeGraph(h, [0, 1]), "a", "h"),
                          torch.tensor([[[0., 0.]], [[1., 2.]]]))
        for counts in ([], [2], [-1, 2]):
            with self.subTest(counts=counts), self.assertRaises(ValueError):
                split_batch(FakeGraph(h, counts), "a", "h")


if __name__ == "__main__":
    unittest.main()
