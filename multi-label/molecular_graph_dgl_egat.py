import csv
import os
from rdkit import Chem
from rdkit.Chem import rdchem
import numpy as np
import dgl
import torch as th
from torch import nn
from torch.nn import init
import pandas as pd
from tqdm import tqdm

def get_atom_features(atom):
    atom_feature_dim = 69

    features = np.zeros((atom_feature_dim,))


    atom_type_idx = {
        "C": 0, "N": 1, "O": 2, "S": 3, "F": 4, "Si": 5, "P": 6, "Cl": 7,
        "Br": 8, "Mg": 9, "Na": 10, "Ca": 11, "Fe": 12, "As": 13, "Al": 14,
        "I": 15, "B": 16, "V": 17, "K": 18, "Tl": 19, "Yb": 20, "Sb": 21,
        "Sn": 22, "Ag": 23, "Pd": 24, "Co": 25, "Se": 26, "Ti": 27, "Zn": 28,
        "H": 29, "Li": 30, "Ge": 31, "Cu": 32, "Au": 33, "Ni": 34, "Cd": 35,
        "In": 36, "Mn": 37, "Zr": 38, "Cr": 39, "Pt": 40, "Hg": 41, "Pb": 42,
        "other": 43
    }.get(atom.GetSymbol(), 43)
    features[atom_type_idx] = 1


    hybridization_idx = {
        rdchem.HybridizationType.SP: 44,
        rdchem.HybridizationType.SP2: 45,
        rdchem.HybridizationType.SP3: 46,
        rdchem.HybridizationType.SP3D: 47
    }.get(atom.GetHybridization(), 48)
    features[hybridization_idx] = 1


    degree = atom.GetDegree()
    if degree < 6:
        features[49 + degree] = 1


    num_h = atom.GetTotalNumHs()
    if num_h < 6:
        features[55 + num_h] = 1


    implicit_valence = atom.GetImplicitValence()
    if implicit_valence < 6:
        features[61 + implicit_valence] = 1


    if atom.IsInRing():
        features[67] = 1


    if atom.GetIsAromatic():
        features[68] = 1

    return features

def get_bond_features(bond):
    bond_feature_dim = 6

    features = np.zeros((bond_feature_dim,))


    bond_type_idx = {
        rdchem.BondType.SINGLE: 0,
        rdchem.BondType.DOUBLE: 1,
        rdchem.BondType.TRIPLE: 2,
        rdchem.BondType.AROMATIC: 3
    }.get(bond.GetBondType(), -1)
    if bond_type_idx >= 0:
        features[bond_type_idx] = 1


    if bond.GetIsConjugated():
        features[4] = 1


    if bond.IsInRing():
        features[5] = 1

    return features


def smiles_to_dgl_graph(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"Invalid SMILES: {smiles}")

    G = dgl.graph([])
    num_atoms = mol.GetNumAtoms()
    num_bonds = mol.GetNumBonds()


    G.add_nodes(num_atoms)


    node_feats = np.array([get_atom_features(atom) for atom in mol.GetAtoms()])
    node_feats = th.tensor(node_feats, dtype=th.float32)


    edges = []
    edge_feats = []
    for bond in mol.GetBonds():
        src = bond.GetBeginAtomIdx()
        dst = bond.GetEndAtomIdx()
        edges.append((src, dst))
        edges.append((dst, src))

        bond_features = get_bond_features(bond)
        edge_feats.append(bond_features)
        edge_feats.append(bond_features)
    G.add_edges(*zip(*edges))


    edge_feats = np.array(edge_feats)
    edge_feats = th.tensor(edge_feats, dtype=th.float32)


    G.ndata['node_feats'] = node_feats
    G.edata['edge_feats'] = edge_feats

    return G, node_feats, edge_feats, num_atoms, num_bonds




class EGATConv(nn.Module):

    def __init__(self,
                 in_node_feats,
                 in_edge_feats,
                 out_node_feats,
                 out_edge_feats,
                 num_heads,
                 **kw_args):
        super().__init__()
        self._num_heads = num_heads
        self._out_node_feats = out_node_feats
        self._out_edge_feats = out_edge_feats
        self.fc_nodes = nn.Linear(in_node_feats, out_node_feats * num_heads, bias=True)
        self.fc_edges = nn.Linear(in_edge_feats + 2 * in_node_feats, out_edge_feats * num_heads, bias=True)
        self.fc_attn = nn.Linear(out_edge_feats, num_heads, bias=False)
        self.reset_parameters()

    def reset_parameters(self):
        """
        Reinitialize learnable parameters.
        """
        gain = init.calculate_gain('relu')
        init.xavier_normal_(self.fc_nodes.weight, gain=gain)
        init.xavier_normal_(self.fc_edges.weight, gain=gain)
        init.xavier_normal_(self.fc_attn.weight, gain=gain)

    def edge_attention(self, edges):
        # extract features
        h_src = edges.src['h']
        h_dst = edges.dst['h']
        f = edges.data['f']
        # stack h_i | f_ij | h_j
        stack = th.cat([h_src, f, h_dst], dim=-1)
        # apply FC and activation
        f_out = self.fc_edges(stack)
        f_out = nn.functional.leaky_relu(f_out)
        f_out = f_out.view(-1, self._num_heads, self._out_edge_feats)
        # apply FC to reduce edge_feats to scalar
        a = self.fc_attn(f_out).sum(-1).unsqueeze(-1)
        return {'a': a, 'f': f_out}

    def message_func(self, edges):
        return {'h': edges.src['h'], 'a': edges.data['a']}

    def reduce_func(self, nodes):
        alpha = nn.functional.softmax(nodes.mailbox['a'], dim=1)
        h = th.sum(alpha * nodes.mailbox['h'], dim=1)
        return {'h': h}

    def forward(self, graph, nfeats, efeats):
        with graph.local_scope():
            ##TODO allow node src and dst feats
            graph.edata['f'] = efeats
            graph.ndata['h'] = nfeats

            graph.apply_edges(self.edge_attention)

            nfeats_ = self.fc_nodes(nfeats)
            nfeats_ = nfeats_.view(-1, self._num_heads, self._out_node_feats)

            graph.ndata['h'] = nfeats_
            graph.update_all(message_func=self.message_func,
                             reduce_func=self.reduce_func)


            nfeats_ = graph.ndata.pop('h')
            nfeats_ = nfeats_.view(-1,
                                   self._num_heads * self._out_node_feats)


            efeats_ = graph.edata.pop('f')
            efeats_ = efeats_.view(-1,
                                   self._num_heads * self._out_edge_feats)

            return nfeats_, efeats_

class EGATModule(nn.Module):
    def __init__(self, in_node_feats, in_edge_feats, out_node_feats, out_edge_feats, num_heads, num_layers=4):
        super(EGATModule, self).__init__()
        self.num_layers = num_layers

        self.egat_layers = nn.ModuleList()

        self.egat1 = EGATConv(in_node_feats, in_edge_feats, out_node_feats, out_edge_feats, num_heads)
        self.egat_layers.append(self.egat1)

        shared_layer = EGATConv(out_node_feats * num_heads, out_edge_feats * num_heads, out_node_feats, out_edge_feats,
                                num_heads)
        for _ in range(1, num_layers):
            self.egat_layers.append(shared_layer)



    def forward(self, graph, nfeats, efeats):

        for i in range(self.num_layers):
            nfeats, efeats = self.egat_layers[i](graph, nfeats, efeats)

        graph_feature = th.mean(nfeats, dim=0)

        return graph_feature




model = EGATModule(in_node_feats=69, in_edge_feats=6, out_node_feats=128, out_edge_feats=128, num_heads=4, num_layers=4)


def process_drug_smiles(input_csv, output_csv, model):

    data = pd.read_csv(input_csv)

    drugs = data.iloc[:, 0]
    smiles_list = data.iloc[:, 1]


    results = []


    model.eval()

    with th.no_grad():
        for drug, smiles in tqdm(zip(drugs, smiles_list), total=len(drugs)):
            try:

                graph, nfeats, efeats, num_atoms, num_bonds = smiles_to_dgl_graph(smiles)


                device = th.device("cuda" if th.cuda.is_available() else "cpu")
                graph = graph.to(device)
                nfeats = nfeats.to(device)
                efeats = efeats.to(device)
                model = model.to(device)


                graph_feature = model(graph, nfeats, efeats)
                graph_feature = graph_feature.cpu().numpy()


                feature_str = ",".join(map(str, graph_feature))
                results.append([drug, feature_str])

            except ValueError as e:
                print(f"Error processing SMILES for drug {drug}: {e}")
                continue


    with open(output_csv, mode="w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["drug", "graph_feature"])
        writer.writerows(results)

    print(f"Graph features saved to {output_csv}")


input_csv = "drug_smiles.csv"
output_csv = "drug_graph_features.csv"

# 调用函数
process_drug_smiles(input_csv, output_csv, model)

