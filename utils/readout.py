"""Padding-aware readout using graph counts, never embedding values."""
import torch
from torch.nn.utils.rnn import pad_sequence


def _counts(counts, batch_size, max_nodes, device):
    counts = torch.as_tensor(counts, device=device)
    if counts.ndim != 1 or counts.numel() != batch_size:
        raise ValueError("node counts must contain one entry per graph")
    if counts.dtype == torch.bool or counts.is_floating_point() or counts.is_complex():
        raise ValueError("node counts must be integers")
    if torch.any(counts < 0) or torch.any(counts > max_nodes):
        raise ValueError("node counts are outside the tensor bounds")
    return counts.to(dtype=torch.long)


def split_batch(bg, ntype, field, device=None):
    """Return padded node features, retaining real zeros and empty node types.

    ``device`` remains accepted for compatibility; features determine placement.
    The graph's explicit batch counts are the sole source of lengths.
    """
    hidden = bg.nodes[ntype].data[field]
    if hidden.ndim != 2:
        raise ValueError("node features must be a two-dimensional tensor")
    if bg.batch_size == 0:
        raise ValueError("cannot split an empty batch")
    counts = _counts(bg.batch_num_nodes(ntype), bg.batch_size,
                     hidden.shape[0], hidden.device)
    sizes = counts.tolist()
    if sum(sizes) != hidden.shape[0]:
        raise ValueError("node counts do not match the number of feature rows")
    return pad_sequence(hidden.split(sizes), batch_first=True)


def pool_atom_fragment(atom_rep, frag_rep, atom_counts, frag_counts,
                       pooling='avg', padding_mode='exclude'):
    """Pool the union of real atom and fragment nodes for each graph.

    ``legacy`` deliberately includes zero padding, reproducing historical
    batch-dependent feature scales for downstream models trained with it.
    """
    if pooling not in ('avg', 'max'):
        raise ValueError("pooling must be 'avg' or 'max'")
    if padding_mode not in ('exclude', 'legacy'):
        raise ValueError("padding_mode must be 'exclude' or 'legacy'")
    if atom_rep.ndim != 3 or frag_rep.ndim != 3:
        raise ValueError("node representations must have shape [batch, nodes, features]")
    if (atom_rep.shape[0] != frag_rep.shape[0]
            or atom_rep.shape[2] != frag_rep.shape[2]
            or atom_rep.device != frag_rep.device
            or atom_rep.dtype != frag_rep.dtype):
        raise ValueError("atom and fragment representations must share batch, features, device and dtype")
    if atom_rep.shape[0] == 0 or atom_rep.shape[2] == 0:
        raise ValueError("batch and feature dimensions must be nonempty")
    if not atom_rep.is_floating_point():
        raise ValueError("node representations must be floating point")
    ac = _counts(atom_counts, atom_rep.shape[0], atom_rep.shape[1], atom_rep.device)
    fc = _counts(frag_counts, frag_rep.shape[0], frag_rep.shape[1], frag_rep.device)
    total = ac + fc
    if torch.any(total == 0):
        raise ValueError("each graph must have at least one atom or fragment node")
    nodes = torch.cat((atom_rep, frag_rep), dim=1)
    if padding_mode == 'legacy':
        return nodes.mean(dim=1) if pooling == 'avg' else nodes.max(dim=1).values
    mask = torch.cat((torch.arange(atom_rep.shape[1], device=nodes.device)[None] < ac[:, None],
                      torch.arange(frag_rep.shape[1], device=nodes.device)[None] < fc[:, None]), dim=1)
    if pooling == 'avg':
        return nodes.masked_fill(~mask.unsqueeze(-1), 0).sum(dim=1) / total.to(nodes.dtype).unsqueeze(-1)
    return nodes.masked_fill(~mask.unsqueeze(-1), float('-inf')).max(dim=1).values
