import random
from torch_geometric.loader import DataLoader
import numpy as np
from datahandling import DataHandler
from torch_geometric.data import Data, Batch
from rdkit import Chem
from rdkit.Chem import AllChem
import torch


class RankingDataset(torch.utils.data.Dataset):

    def __init__(self, data_dict, list_size=1000):
        self.data_dict = data_dict
        self.list_size = list_size
        self.queries = list(data_dict.keys())
        self.cell_wise = True

    def __len__(self):
        return len(self.queries)

    def __getitem__(self, idx):
        query = self.queries[idx]
        query_data = self.data_dict[query]

        query_feat = torch.tensor(query_data['query_feat'], dtype=torch.float32)

        docs_feat   = query_data['docs']['docs_feat']
        docs_label = query_data['docs']['responses']

        if len(docs_feat) > self.list_size:
            indices = random.sample(range(len(docs_feat)), self.list_size)
            docs_feat  = [docs_feat[i] for i in indices]
            docs_label = [docs_label[i] for i in indices]

        docs_label = np.array(docs_label)
        max_val = docs_label.max()
        docs_label = -docs_label + max_val
        docs_label = docs_label.tolist()
        if self.cell_wise:
            docs_feat = [self.smiles_to_graph(smiles) for smiles in docs_feat]

        if len(docs_feat) < self.list_size:
            doc_feat_dummy, doc_label_dummy = self.dummydata()
            docs_feat = docs_feat + [doc_feat_dummy for i in range(self.list_size - len(docs_feat))]
            docs_label = docs_label + [doc_label_dummy for i in range(self.list_size - len(docs_label))]

        docs_label = torch.tensor(docs_label, dtype=torch.float32)
        return query_feat, docs_feat, docs_label

    @staticmethod
    def smiles_to_graph(smiles):
        # Convert SMILES to RDKit molecule
        mol = Chem.MolFromSmiles(smiles)


        # Initialize feature matrices
        atom_features = []
        edge_index = []
        edge_features = []

        # Atom features
        for atom in mol.GetAtoms():
            atom_feature = [
                atom.GetAtomicNum(),
                atom.GetTotalDegree(),
                atom.GetFormalCharge(),
                atom.GetTotalNumHs(),
                atom.GetNumRadicalElectrons(),
                atom.GetIsAromatic(),
                atom.IsInRing(),
                atom.GetMass(),
                atom.GetExplicitValence(),
                atom.GetImplicitValence(),
                *[int(atom.IsInRingSize(size)) for size in range(3, 8)],  # Is in ring of size 3-7
                *[int(atom.GetHybridization() == x) for x in [
                    Chem.rdchem.HybridizationType.SP,
                    Chem.rdchem.HybridizationType.SP2,
                    Chem.rdchem.HybridizationType.SP3,
                    Chem.rdchem.HybridizationType.SP3D,
                    Chem.rdchem.HybridizationType.SP3D2
                ]],
                atom.GetChiralTag(),
            ]
            atom_features.append(atom_feature)

        # Edge features
        for bond in mol.GetBonds():
            i = bond.GetBeginAtomIdx()
            j = bond.GetEndAtomIdx()
            edge_index.extend([[i, j], [j, i]])

            bond_feature = [
                bond.GetBondTypeAsDouble(),
                bond.GetIsConjugated(),
                bond.IsInRing(),
                bond.GetIsAromatic(),
                bond.GetStereo(),
            ]
            edge_features.extend([bond_feature, bond_feature])  # Add for both directions

        # Convert to PyTorch tensors
        x = torch.tensor(atom_features, dtype=torch.float)
        edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
        edge_attr = torch.tensor(edge_features, dtype=torch.float)

        # Create PyG Data object
        data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr)
        return data


    def dummydata(self):
        if self.cell_wise:
            num_dum_atoms = 10
            atom_feat_len = 21
            edge_feat_len = 5
            edge_index = []

            for i in range(num_dum_atoms - 1):
                edge_index.append([i, i+1])
                edge_index.append([i, i + 1])

            doc_feat_dummy = Data(x=torch.zeros((num_dum_atoms, atom_feat_len)),
                             edge_index=torch.tensor(edge_index, dtype=torch.int32),
                             edge_attr=torch.zeros((len(edge_index), edge_feat_len)))
            doc_label_dummy = -1.0

        return doc_feat_dummy, doc_label_dummy

