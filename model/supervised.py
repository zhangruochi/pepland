import dgl
import pandas as pd
import torch
from rdkit import Chem
from torch import nn
from torch.utils.data import Dataset

from model.data import Mol2HeteroGraph


class SupervisedMolGraphDataset(Dataset):
    def __init__(self, csv_path, smiles_col="smiles", label_col="label", frag="258"):
        self.df = pd.read_csv(csv_path)
        self.smiles_col = smiles_col
        self.label_col = label_col
        self.frag = frag
        self.unfound_fragments = set()

        missing = {smiles_col, label_col} - set(self.df.columns)
        if missing:
            raise ValueError(f"{csv_path} is missing columns: {sorted(missing)}")

        self.df = self.df[[smiles_col, label_col]].dropna().reset_index(drop=True)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        row = self.df.iloc[index]
        smi = str(row[self.smiles_col])
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            raise ValueError(f"Invalid SMILES at row {index}: {smi}")

        graph = Mol2HeteroGraph(mol,
                                frag=self.frag,
                                unfound_fragments=self.unfound_fragments)
        label = torch.tensor(row[self.label_col])
        return graph, label

    def get_unfound_fragments(self):
        return set(self.unfound_fragments)


def supervised_collate(samples):
    graphs, labels = map(list, zip(*samples))
    return dgl.batch(graphs), torch.stack(labels)


def split_batch(bg, ntype, field, device):
    hidden = bg.nodes[ntype].data[field]
    node_size = bg.batch_num_nodes(ntype)
    start_index = torch.cat([
        torch.tensor([0], device=device),
        torch.cumsum(node_size, 0)[:-1]
    ])
    max_num_node = max(node_size)

    hidden_lst = []
    for i in range(bg.batch_size):
        start, size = start_index[i], node_size[i]
        cur_hidden = hidden.narrow(0, start, size)
        cur_hidden = torch.nn.ZeroPad2d(
            (0, 0, 0, max_num_node - cur_hidden.shape[0]))(cur_hidden)
        hidden_lst.append(cur_hidden.unsqueeze(0))

    return torch.cat(hidden_lst, 0)


class PepLandClassifier(nn.Module):
    def __init__(self, encoder, hid_dim=300, num_classes=1, pool="avg", dropout=0.1):
        super().__init__()
        if pool not in {"avg", "max"}:
            raise ValueError("pool must be 'avg' or 'max'")

        self.encoder = encoder
        self.pool = pool
        self.classifier = nn.Sequential(
            nn.Linear(hid_dim, hid_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hid_dim, num_classes),
        )

    def forward(self, bg):
        atom_embed, frag_embed = self.encoder(bg)
        bg.nodes["a"].data["h"] = atom_embed
        bg.nodes["p"].data["h"] = frag_embed

        atom_rep = split_batch(bg, "a", "h", bg.device)
        frag_rep = split_batch(bg, "p", "h", bg.device)
        node_rep = torch.cat([atom_rep, frag_rep], dim=1)

        if self.pool == "avg":
            graph_rep = node_rep.mean(dim=1)
        else:
            graph_rep = node_rep.max(dim=1)[0]

        return self.classifier(graph_rep)
