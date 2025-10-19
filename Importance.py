# -*- coding: utf-8 -*-


import os
import torch
import pandas as pd
import numpy as np
from rdkit import Chem
from rdkit.Chem import Draw
from torch_geometric.data import Data as PyGData
from torch_geometric.nn import GATConv, global_mean_pool
from torch_geometric.loader import DataLoader as PyGDataLoader
from torch.utils.data import Dataset
from collections import defaultdict
import matplotlib.pyplot as plt
from tqdm import tqdm
import logging

# -------------------- Configuration --------------------
CONFIG = {
    "device": "cuda" if torch.cuda.is_available() else "cpu",
    "model_params": {
        "input_dim": 12,
        "hidden_dim": 512,
        "output_dim": 2,
        "num_heads": 8
    },
    "analysis": {
        "radius": 2,
        "top_atoms": 3,
        "top_groups": 25
    },
    "paths": {
        "model": "best_model.pth",
        "train_data": "train_data.xlsx",
        "output_csv": "results/important_groups.csv",
        "output_img": "results/group_structures/"
    }
}

# -------------------- Model --------------------
class EnhancedGAT(torch.nn.Module):
    """GAT-based classifier for molecular graphs (structure only)."""
    def __init__(self, input_dim, hidden_dim, output_dim, num_heads=4):
        super().__init__()
        self.conv1 = GATConv(input_dim, hidden_dim, heads=num_heads, edge_dim=1)
        self.conv2 = GATConv(hidden_dim * num_heads, hidden_dim, heads=1, edge_dim=1)
        self.fc = torch.nn.Linear(hidden_dim, output_dim)
    
    def forward(self, data):
        x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr
        batch = data.batch
        x = self.conv1(x, edge_index, edge_attr=edge_attr)
        x = torch.relu(x)
        x = self.conv2(x, edge_index, edge_attr=edge_attr)
        x = torch.relu(x)
        x = global_mean_pool(x, batch)
        return self.fc(x)

# -------------------- Molecular Graph Processing --------------------
class MolecularProcessor:
    """Convert SMILES to PyG graph (details omitted)."""
    def __init__(self):
        self.atom_feature_dim = 12

    def smiles_to_graph(self, smiles):
        # TODO: Implement feature extraction & adjacency matrix construction
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None
        # Placeholder example
        x = torch.randn((mol.GetNumAtoms(), self.atom_feature_dim), dtype=torch.float)
        edge_index = torch.zeros((2, 0), dtype=torch.long)
        edge_attr = torch.zeros((0, 1), dtype=torch.float)
        return PyGData(x=x, edge_index=edge_index, edge_attr=edge_attr, smiles=smiles)

class NewSMILESDataset(Dataset):
    """Dataset wrapper for SMILES strings."""
    def __init__(self, smiles_list):
        self.smiles_list = smiles_list
        self.processor = MolecularProcessor()

    def __len__(self):
        return len(self.smiles_list)
    
    def __getitem__(self, idx):
        return self.processor.smiles_to_graph(self.smiles_list[idx])

# -------------------- Group Analysis --------------------
class ImportantGroupAnalyzer:
    """Identify important functional groups (structure only)."""
    def __init__(self, model, processor):
        self.model = model
        self.processor = processor
        self.group_counter = defaultdict(float)
        self.radius = CONFIG['analysis']['radius']
        self.top_atoms = CONFIG['analysis']['top_atoms']

    def analyze_molecule(self, smiles):
        # TODO: Implement attention-based atom importance & environment extraction
        pass

    def get_important_groups(self, top_groups=10):
        # TODO: Implement clustering & selection of functional groups
        return list(self.group_counter.keys())[:top_groups]

# -------------------- Visualization --------------------
def visualize_groups(groups, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    fig, axes = plt.subplots(5, 5, figsize=(20, 12))
    for idx, (smarts, ax) in enumerate(zip(groups, axes.flatten())):
        try:
            mol = Chem.MolFromSmarts(smarts)
            img = Draw.MolToImage(mol, size=(300, 300))
            ax.imshow(img)
            ax.set_title(f"Group {idx+1}", fontsize=8)
        except:
            ax.set_title("Invalid", fontsize=8)
        ax.axis('off')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "important_groups.png"), dpi=300)
    plt.close()

# -------------------- Main --------------------
def main():
    processor = MolecularProcessor()
    model = EnhancedGAT(**CONFIG['model_params']).to(CONFIG['device'])
    model.load_state_dict(torch.load(CONFIG['paths']['model'], map_location=CONFIG['device'])["model_state_dict"])
    model.eval()

    train_df = pd.read_excel(CONFIG['paths']['train_data'])
    dataset = [d for d in NewSMILESDataset(train_df['SMILES'].tolist()) if d is not None]
    loader = PyGDataLoader(dataset, batch_size=32, shuffle=False)

    positive_smiles = []
    for data in tqdm(loader, desc="Predicting"):
        with torch.no_grad():
            data = data.to(CONFIG['device'])
            preds = torch.softmax(model(data), dim=1)[:, 1].cpu().numpy()
            positive_smiles.extend([s for s, p in zip(data.smiles, preds) if p > 0.5])

    analyzer = ImportantGroupAnalyzer(model, processor)
    for smi in tqdm(positive_smiles, desc="Analyzing"):
        analyzer.analyze_molecule(smi)

    groups = analyzer.get_important_groups(CONFIG['analysis']['top_groups'])
    pd.DataFrame({"SMARTS": groups}).to_csv(CONFIG['paths']['output_csv'], index=False)
    visualize_groups(groups, CONFIG['paths']['output_img'])

if __name__ == "__main__":
    main()
