import pandas as pd
import numpy as np
from rdkit import Chem
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import torch
from torch.utils.data import Dataset
import torch.nn as nn
from torch_geometric.data import Data as PyGData
from torch_geometric.loader import DataLoader as PyGDataLoader
from torch_geometric.nn import GATConv, global_mean_pool

def load_data(file_path):
    data = pd.read_excel(file_path)
    data = data[["SMILES", "Label"]].dropna()
    return data

def smiles_to_graph(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"Invalid SMILES: {smiles}")
    atom_features = []
    for atom in mol.GetAtoms():
        features = [
            atom.GetAtomicNum(),
            atom.GetTotalNumHs(),
            atom.GetDegree(),
            int(atom.GetHybridization()),
            atom.GetIsAromatic(),
            atom.GetFormalCharge(),
            atom.IsInRing(),
            int(atom.GetChiralTag()),
            atom.GetTotalValence(),
            atom.GetMass() / 100.0,
            atom.GetNumRadicalElectrons(),
            len(atom.GetNeighbors()) > 2
        ]
        atom_features.append(features)
    scaler = MinMaxScaler()
    atom_features = scaler.fit_transform(atom_features).astype(np.float32)
    adj = np.zeros((mol.GetNumAtoms(), mol.GetNumAtoms()), dtype=np.float32)
    for bond in mol.GetBonds():
        i, j = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        bond_val = {
            Chem.BondType.SINGLE: 1,
            Chem.BondType.DOUBLE: 2,
            Chem.BondType.TRIPLE: 3,
            Chem.BondType.AROMATIC: 1.5
        }.get(bond.GetBondType(), 0)
        adj[i, j] = bond_val
        adj[j, i] = bond_val
    rows, cols = np.nonzero(adj)
    edge_values = adj[rows, cols]
    return atom_features, (rows, cols, edge_values), mol

class SMILESGraphDataset(Dataset):
    def __init__(self, df):
        self.df = df
    def __len__(self):
        return len(self.df)
    def __getitem__(self, idx):
        smiles = self.df.iloc[idx]["SMILES"]
        label = self.df.iloc[idx]["Label"]
        try:
            atom_features, (rows, cols, edge_attr), mol = smiles_to_graph(smiles)
            edge_index = torch.tensor(np.column_stack((rows, cols)).T, dtype=torch.long)
            x = torch.tensor(atom_features, dtype=torch.float)
            edge_attr = torch.tensor(edge_attr, dtype=torch.float).unsqueeze(1)
            y = torch.tensor([label], dtype=torch.long)
            return PyGData(x=x, edge_index=edge_index, edge_attr=edge_attr, y=y, smiles=smiles)
        except Exception:
            return None

class EnhancedGAT(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_heads=8):
        super().__init__()
        self.conv1 = GATConv(input_dim, hidden_dim, heads=num_heads, edge_dim=1)
        self.bn1 = nn.BatchNorm1d(hidden_dim * num_heads)
        self.conv2 = GATConv(hidden_dim * num_heads, hidden_dim, heads=1, edge_dim=1)
        self.bn2 = nn.BatchNorm1d(hidden_dim)
        self.fc = nn.Linear(hidden_dim, output_dim)
        self.dropout = nn.Dropout(0.5)
    def forward(self, data):
        x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr
        batch = data.batch
        x = self.conv1(x, edge_index, edge_attr=edge_attr)
        x = self.bn1(x)
        x = torch.relu(x)
        x = self.dropout(x)
        x = self.conv2(x, edge_index, edge_attr=edge_attr)
        x = self.bn2(x)
        x = torch.relu(x)
        x = global_mean_pool(x, batch)
        return self.fc(x)

def train_model(model, loader, optimizer, criterion, device):
    model.train()
    total_loss = 0.0
    count = 0
    for data in loader:
        if data is None:
            continue
        data = data.to(device)
        optimizer.zero_grad()
        out = model(data)
        loss = criterion(out, data.y.view(-1))
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        count += 1
    return total_loss / max(1, count)

if __name__ == "__main__":
    config = {
        "input_dim": 12,
        "hidden_dim": 512,
        "output_dim": 2,
        "num_heads": 8,
        "lr": 1e-5,
        "epochs": 1000,
        "batch_size": 32
    }
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    raw_data = load_data("data.xlsx")
    full_data = pd.concat([raw_data], ignore_index=True)
    train_df, _ = train_test_split(full_data, test_size=0.2, stratify=full_data["Label"], random_state=42)
    train_dataset = SMILESGraphDataset(train_df)
    train_loader = PyGDataLoader([d for d in train_dataset if d is not None], batch_size=config["batch_size"], shuffle=True)
    model = EnhancedGAT(input_dim=config["input_dim"], hidden_dim=config["hidden_dim"], output_dim=config["output_dim"], num_heads=config["num_heads"]).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config["lr"])
    criterion = nn.CrossEntropyLoss()
    for epoch in range(1, config["epochs"] + 1):
        train_loss = train_model(model, train_loader, optimizer, criterion, device)
        print(f"Epoch {epoch}: Train Loss: {train_loss:.6f}")
    print("Training completed")
