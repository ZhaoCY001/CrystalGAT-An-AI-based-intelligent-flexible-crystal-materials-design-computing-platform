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
from torch_geometric.nn import TransformerConv, global_mean_pool
from collections import deque

def load_data(file_path):
    data = pd.read_excel(file_path)
    data = data[["SMILES", "Label"]].dropna()
    return data

def smiles_to_graph(smiles, k_eig=8, K=4):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"Invalid SMILES: {smiles}")
    n = mol.GetNumAtoms()
    atom_feats = []
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
        atom_feats.append(features)
    atom_feats = np.array(atom_feats, dtype=np.float32) if n > 0 else np.zeros((0, 12), dtype=np.float32)
    adj = np.zeros((n, n), dtype=np.float32)
    for bond in mol.GetBonds():
        i, j = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        bond_val = {
            Chem.BondType.SINGLE: 1,
            Chem.BondType.DOUBLE: 2,
            Chem.BondType.TRIPLE: 3,
            Chem.BondType.AROMATIC: 1.5
        }.get(bond.GetBondType(), 1.0)
        adj[i, j] = bond_val
        adj[j, i] = bond_val
    A_bin = (adj > 0).astype(np.float32)
    if n <= 1:
        eig_feats = np.zeros((n, k_eig), dtype=np.float32)
    else:
        deg = np.diag(A_bin.sum(axis=1))
        L = deg - A_bin
        try:
            eigvals, eigvecs = np.linalg.eigh(L)
            start_idx = 1
            vecs = eigvecs[:, start_idx:start_idx + k_eig] if eigvecs.shape[1] > start_idx else np.zeros((n, k_eig))
            if vecs.shape[1] < k_eig:
                pad = np.zeros((n, k_eig - vecs.shape[1]), dtype=np.float32)
                eig_feats = np.hstack((vecs.astype(np.float32), pad))
            else:
                eig_feats = vecs[:, :k_eig].astype(np.float32)
            col_min = eig_feats.min(axis=0, keepdims=True)
            col_max = eig_feats.max(axis=0, keepdims=True)
            eig_feats = (eig_feats - col_min) / (col_max - col_min + 1e-8)
        except Exception:
            eig_feats = np.zeros((n, k_eig), dtype=np.float32)
    dist_matrix = np.full((n, n), fill_value=999, dtype=np.int32)
    for src in range(n):
        q = deque([src])
        dist_matrix[src, src] = 0
        while q:
            u = q.popleft()
            for v in np.where(A_bin[u] > 0)[0]:
                if dist_matrix[src, v] == 999:
                    dist_matrix[src, v] = dist_matrix[src, u] + 1
                    q.append(v)
    dist_hist = np.zeros((n, K), dtype=np.float32)
    for i in range(n):
        denom = max(1, n - 1)
        for d in range(1, K + 1):
            count = int(((dist_matrix[i] == d)).sum())
            dist_hist[i, d - 1] = count / denom
    combined = np.concatenate([atom_feats, eig_feats, dist_hist], axis=1) if n > 0 else np.zeros((0, 12 + k_eig + K), dtype=np.float32)
    try:
        scaler = MinMaxScaler()
        combined = scaler.fit_transform(combined).astype(np.float32)
    except Exception:
        combined = combined.astype(np.float32)
    rows, cols = np.nonzero(adj)
    edge_values = adj[rows, cols].astype(np.float32)
    return combined, (rows, cols, edge_values), mol

class SMILESGraphDataset(Dataset):
    def __init__(self, df, k_eig=8, K=4):
        self.df = df.reset_index(drop=True)
        self.k_eig = k_eig
        self.K = K
    def __len__(self):
        return len(self.df)
    def __getitem__(self, idx):
        smiles = self.df.iloc[idx]["SMILES"]
        label = self.df.iloc[idx]["Label"]
        try:
            atom_features, (rows, cols, edge_attr), mol = smiles_to_graph(smiles, k_eig=self.k_eig, K=self.K)
            edge_index = torch.tensor(np.column_stack((rows, cols)).T, dtype=torch.long)
            x = torch.tensor(atom_features, dtype=torch.float)
            edge_attr = torch.tensor(edge_attr, dtype=torch.float).unsqueeze(1)
            y = torch.tensor([label], dtype=torch.long)
            return PyGData(x=x, edge_index=edge_index, edge_attr=edge_attr, y=y, smiles=smiles)
        except Exception:
            return None

class EnhancedTransformer(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_heads=8, edge_dim=1, dropout=0.3):
        super().__init__()
        self.conv1 = TransformerConv(in_channels=input_dim, out_channels=hidden_dim, heads=num_heads, concat=True, edge_dim=edge_dim)
        self.ln1 = nn.LayerNorm(hidden_dim * num_heads)
        self.dropout1 = nn.Dropout(dropout)
        self.conv2 = TransformerConv(in_channels=hidden_dim * num_heads, out_channels=hidden_dim, heads=1, concat=False, edge_dim=edge_dim)
        self.ln2 = nn.LayerNorm(hidden_dim)
        self.dropout2 = nn.Dropout(dropout)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, output_dim)
        )
    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        edge_attr = getattr(data, "edge_attr", None)
        batch = data.batch
        x = self.conv1(x, edge_index, edge_attr=edge_attr) if edge_attr is not None else self.conv1(x, edge_index)
        x = self.ln1(x)
        x = torch.relu(x)
        x = self.dropout1(x)
        x = self.conv2(x, edge_index, edge_attr=edge_attr) if edge_attr is not None else self.conv2(x, edge_index)
        x = self.ln2(x)
        x = torch.relu(x)
        x = self.dropout2(x)
        x = global_mean_pool(x, batch)
        out = self.mlp(x)
        return out

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
    k_eig = 8
    K = 4
    config = {
        "input_dim": 12 + k_eig + K,
        "hidden_dim": 512,
        "output_dim": 2,
        "num_heads": 8,
        "edge_dim": 1,
        "lr": 1e-5,
        "epochs": 1000,
        "batch_size": 32,
        "k_eig": k_eig,
        "K": K
    }
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    raw_data = load_data("data.xlsx")
    full_data = pd.concat([raw_data], ignore_index=True)
    train_df, _ = train_test_split(full_data, test_size=0.2, stratify=full_data["Label"], random_state=42)
    train_dataset = SMILESGraphDataset(train_df, k_eig=config["k_eig"], K=config["K"])
    train_loader = PyGDataLoader([d for d in train_dataset if d is not None], batch_size=config["batch_size"], shuffle=True)
    model = EnhancedTransformer(input_dim=config["input_dim"], hidden_dim=config["hidden_dim"], output_dim=config["output_dim"]).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config["lr"])
    criterion = nn.CrossEntropyLoss()
    for epoch in range(1, config["epochs"] + 1):
        train_loss = train_model(model, train_loader, optimizer, criterion, device)
        print(f"Epoch {epoch}: Train Loss: {train_loss:.6f}")
    print("Training completed")
