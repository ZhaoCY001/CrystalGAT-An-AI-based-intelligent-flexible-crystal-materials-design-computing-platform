# -*- coding: utf-8 -*-
"""
Created on Sun Oct 19 15:25:49 2025

@author: User
"""

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch_geometric.data import Data as PyGData
from torch_geometric.loader import DataLoader as PyGDataLoader
from torch_geometric.nn import GATConv, global_mean_pool
from sklearn.preprocessing import MinMaxScaler
from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem
import itertools
from tqdm import tqdm
import os

# -------------------- Config --------------------
class Config:
    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.mod_params = {
            "ring_groups": [("CC(=O)O", 0), ("OC", 0)],   # Example placeholders
            "chain_groups": [("CC(=O)O", 0), ("C(O)O", 0)],
            "max_output": 20,
            "similarity_threshold": 0.1,
            "max_modification_depth": 2
        }
        self.paths = {
            "model": "best_model.pth",
            "new_data": "processed_data.xlsx",
            "output_file": "modified_molecules.xlsx"
        }

# -------------------- GAT Model --------------------
class EnhancedGAT(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_heads=4):
        super().__init__()
        self.conv1 = GATConv(input_dim, hidden_dim, heads=num_heads, edge_dim=1)
        self.bn1 = nn.BatchNorm1d(hidden_dim * num_heads)
        self.conv2 = GATConv(hidden_dim * num_heads, hidden_dim, heads=1, edge_dim=1)
        self.bn2 = nn.BatchNorm1d(hidden_dim)
        self.fc = nn.Linear(hidden_dim, output_dim)
        self.dropout = nn.Dropout(0.5)
        
    def forward(self, data):
        x, edge_index, edge_attr, batch = data.x, data.edge_index, data.edge_attr, data.batch
        x = self.conv1(x, edge_index, edge_attr=edge_attr)
        x = self.bn1(x)
        x = torch.relu(x)
        x = self.dropout(x)

        x = self.conv2(x, edge_index, edge_attr=edge_attr)
        x = self.bn2(x)
        x = torch.relu(x)

        x = global_mean_pool(x, batch)
        return self.fc(x)

# -------------------- Molecular Processor --------------------
class MolecularProcessor:
    def __init__(self):
        self.scaler = MinMaxScaler()
        self.atom_feature_dim = 12

    def smiles_to_graph(self, smiles):
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None

        # Atom features (simplified)
        atom_features = [[atom.GetAtomicNum()] for atom in mol.GetAtoms()]
        atom_features = self.scaler.fit_transform(atom_features).astype(np.float32)

        # Build edge index and attributes (simplified)
        adj = np.zeros((mol.GetNumAtoms(), mol.GetNumAtoms()), dtype=np.float32)
        for bond in mol.GetBonds():
            i, j = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
            adj[i, j] = adj[j, i] = 1.0

        rows, cols = np.nonzero(adj)
        edge_values = adj[rows, cols]

        return PyGData(
            x=torch.tensor(atom_features, dtype=torch.float),
            edge_index=torch.tensor(np.column_stack((rows, cols)).T, dtype=torch.long),
            edge_attr=torch.tensor(edge_values, dtype=torch.float).view(-1, 1),
            smiles=smiles
        )

# -------------------- Modifier Engine --------------------
class MolecularModifierEngine:
    def __init__(self, config):
        self.config = config
        self.processor = MolecularProcessor()
        self.model = self._load_model()

    def _load_model(self):
        model = EnhancedGAT(
            input_dim=12,
            hidden_dim=512,
            output_dim=2,
            num_heads=8
        ).to(self.config.device)

        checkpoint = torch.load(self.config.paths["model"], map_location=self.config.device)
        model.load_state_dict(checkpoint["model_state_dict"])
        model.eval()
        return model

    def _generate_modified_molecules(self, smiles):
        # Placeholder: generate candidate structures
        mol = Chem.MolFromSmiles(smiles)
        if not mol:
            return []
        return [mol]  # simplified placeholder

    def _filter_molecules(self, candidates, orig_smiles):
        # Placeholder: filtering logic
        results = []
        orig_mol = Chem.MolFromSmiles(orig_smiles)
        fp1 = AllChem.GetMorganFingerprint(orig_mol, 2)
        for mol in candidates:
            canon_smiles = Chem.MolToSmiles(mol, isomericSmiles=True, canonical=True)
            fp2 = AllChem.GetMorganFingerprint(mol, 2)
            sim = DataStructs.TanimotoSimilarity(fp1, fp2)
            if sim < self.config.mod_params["similarity_threshold"]:
                continue
            results.append({"smiles": canon_smiles, "similarity": sim, "probability": 0.9})
        return results[:self.config.mod_params["max_output"]]

    def process(self):
        df = pd.read_excel(self.config.paths["new_data"])
        results = []

        for _, row in tqdm(df.iterrows(), total=len(df)):
            orig_smiles = row["SMILES"]
            candidates = self._generate_modified_molecules(orig_smiles)
            filtered = self._filter_molecules(candidates, orig_smiles)
            for item in filtered:
                results.append({
                    "Original_SMILES": orig_smiles,
                    "Modified_SMILES": item["smiles"],
                    "Similarity": item["similarity"],
                    "Probability": item["probability"]
                })

        result_df = pd.DataFrame(results)
        result_df.to_excel(self.config.paths["output_file"], index=False)
        print(f"Finished. Total valid molecules: {len(result_df)}")

# -------------------- Main --------------------
if __name__ == "__main__":
    config = Config()
    engine = MolecularModifierEngine(config)
    engine.process()
