# layers.py
# -*- coding: utf-8 -*-

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# ============== RDKit (可选) =============
try:
    from rdkit import Chem
    _HAS_RDKIT = True
except Exception:
    _HAS_RDKIT = False
    print("Warning: RDKit not found. DrugAtomEncoder will use simple SMILES tokenization.")


# ================== 1. Local: GAT Layer ==================
class GATLayer(nn.Module):
    """
    Local GAT: 捕捉细胞间的 CNV 图拓扑结构
    """
    def __init__(self, in_dim, out_dim, nheads=4, dropout=0.1, alpha=0.2):
        super().__init__()
        self.dropout = dropout
        self.nheads = nheads
        self.out_dim = out_dim

        assert out_dim % nheads == 0, "out_dim must be divisible by nheads"
        self.head_dim = out_dim // nheads

        self.W = nn.Linear(in_dim, out_dim, bias=False)
        self.a_src = nn.Parameter(torch.Tensor(1, nheads, self.head_dim))
        self.a_dst = nn.Parameter(torch.Tensor(1, nheads, self.head_dim))
        self.leakyrelu = nn.LeakyReLU(alpha)

        nn.init.xavier_uniform_(self.W.weight)
        nn.init.xavier_uniform_(self.a_src)
        nn.init.xavier_uniform_(self.a_dst)

    def forward(self, h, adj):
        N = h.size(0)
        h_prime = self.W(h).view(N, self.nheads, self.head_dim)

        attn_src = (h_prime * self.a_src).sum(dim=-1).unsqueeze(-1)
        attn_dst = (h_prime * self.a_dst).sum(dim=-1).unsqueeze(-1)
        attn_scores = attn_src.view(N, 1, self.nheads) + attn_dst.view(1, N, self.nheads)
        attn_scores = self.leakyrelu(attn_scores)

        if adj.is_sparse:
            adj = adj.to_dense()

        adj = adj.unsqueeze(-1)
        zero_vec = -9e15 * torch.ones_like(attn_scores)
        attention = torch.where(adj > 0, attn_scores, zero_vec)

        attention = F.softmax(attention, dim=1)
        attention = F.dropout(attention, self.dropout, training=self.training)

        h_out = torch.matmul(attention.permute(2, 0, 1), h_prime.permute(1, 0, 2))
        h_out = h_out.permute(1, 0, 2).contiguous().view(N, self.out_dim)

        return F.elu(h_out)


# ================== 2. Global: Standard Transformer Block (No Gate) ==================
class GlobalTransformerBlock(nn.Module):
    """
    Global MHSA: 捕捉 Pathway 间的全局长程依赖
    (No Gate, No Mask)
    """
    def __init__(self, in_dim, nheads=4, dropout=0.1):
        super().__init__()
        # 1. Global Self-Attention
        self.mha = nn.MultiheadAttention(embed_dim=in_dim, num_heads=nheads, dropout=dropout, batch_first=True)

        # 2. Norm & FFN
        self.ln1 = nn.LayerNorm(in_dim)
        self.ln2 = nn.LayerNorm(in_dim)
        self.ffn = nn.Sequential(
            nn.Linear(in_dim, in_dim * 4),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(in_dim * 4, in_dim)
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # x: [N_cells, Dim] -> [1, N_cells, Dim] (Batch size = 1)
        x_in = x.unsqueeze(0)

        # Global MHSA
        attn_out, _ = self.mha(x_in, x_in, x_in)
        attn_out = attn_out.squeeze(0)  # [N, D]

        # Residual + Norm (Direct)
        x = self.ln1(x + self.dropout(attn_out))

        # FFN
        ffn_out = self.ffn(x)
        x = self.ln2(x + self.dropout(ffn_out))
        return x


# ================== 3. Drug Micro Encoder (GNN) ==================
ATOM_VOCAB = ['B', 'C', 'N', 'O', 'F', 'Si', 'P', 'S', 'Cl', 'Br', 'I', 'H', '*']
ATOM2IDX = {a: i for i, a in enumerate(ATOM_VOCAB)}

def _atom_one_hot(sym: str):
    idx = ATOM2IDX.get(sym, len(ATOM_VOCAB) - 1)
    return F.one_hot(torch.tensor(idx), num_classes=len(ATOM_VOCAB)).float()

def _tokenize_smiles(smi: str):
    i, toks = 0, []
    while i < len(smi):
        if i + 1 < len(smi) and smi[i:i + 2] in ('Cl', 'Br', 'Si'):
            toks.append(smi[i:i + 2]); i += 2
        else:
            toks.append(smi[i]); i += 1
    return toks

def smiles_to_graph(smiles: str):
    atoms_list = []
    adj_mat = torch.eye(1)
    n = 1
    if _HAS_RDKIT and smiles:
        try:
            mol = Chem.MolFromSmiles(smiles)
            if mol is not None:
                mol = Chem.AddHs(mol)
                atoms_list_rdkit = [atom.GetSymbol() for atom in mol.GetAtoms()]
                atoms_list = [a if a in ATOM2IDX else '*' for a in atoms_list_rdkit]
                n = len(atoms_list)
                if n > 0:
                    adj_mat = torch.zeros((n, n), dtype=torch.float32)
                    for bond in mol.GetBonds():
                        i = bond.GetBeginAtomIdx(); j = bond.GetEndAtomIdx()
                        adj_mat[i, j] = 1.0; adj_mat[j, i] = 1.0
                    adj_mat = adj_mat + torch.eye(n)
                else:
                    atoms_list = ['*']; n = 1; adj_mat = torch.eye(1)
            else:
                atoms_list = ['*']
        except Exception:
            atoms_list = ['*']
    else:
        toks = _tokenize_smiles(smiles or "")
        atoms_list = []
        for t in toks:
            if t in ATOM2IDX: atoms_list.append(t)
            elif t.isalpha(): atoms_list.append('*')
        if len(atoms_list) == 0: atoms_list = ['*']
        n = len(atoms_list)
        adj_mat = torch.eye(n)
        if n > 1:
            for i in range(n - 1):
                adj_mat[i, i + 1] = 1.0; adj_mat[i + 1, i] = 1.0

    if not atoms_list: atoms_list = ['*']; n = 1; adj_mat = torch.eye(1)

    X_oh = torch.stack([_atom_one_hot(a) for a in atoms_list], dim=0)
    pos_feat = torch.stack([
        torch.linspace(0, 1, n),
        torch.ones(n) * (2.0 / max(n, 2)),
        torch.tensor([1.0 if i in (0, n - 1) else 0.0 for i in range(n)], dtype=torch.float32)
    ], dim=1)
    X = torch.cat([X_oh, pos_feat], dim=1)
    if X.shape[0] != adj_mat.shape[0]: adj_mat = torch.eye(X.shape[0])
    return X, adj_mat

class MLP(nn.Module):
    def __init__(self, in_dim, hid, out, dropout=0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hid), nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hid, out)
        )
    def forward(self, x): return self.net(x)

class GINELike(nn.Module):
    def __init__(self, dim, dropout=0.1):
        super().__init__()
        self.msg = MLP(dim, dim, dim, dropout)
        self.upd = MLP(dim, dim, dim, dropout)
        self.eps = nn.Parameter(torch.Tensor([0.0]))
        self.ln = nn.LayerNorm(dim)
    def forward(self, x, adj):
        m = self.msg(x)
        m = adj @ m
        out = self.upd((1.0 + self.eps) * x + m)
        return self.ln(F.relu(out))

class VirtualNodeBlock(nn.Module):
    def __init__(self, dim, dropout=0.1):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(dim, dim), nn.ReLU(inplace=True),
            nn.Dropout(dropout), nn.Linear(dim, dim)
        )
        self.ln = nn.LayerNorm(dim)
        self.attn_gate = nn.Linear(dim, 1)
    def forward(self, x):
        scores = self.attn_gate(x)
        attn_weights = F.softmax(scores, dim=0)
        g = (x * attn_weights).sum(dim=0, keepdim=True)
        return self.ln(x + self.proj(g))

class DrugAtomEncoder(nn.Module):
    def __init__(self, in_dim, hidden, depth=3, dropout=0.1):
        super().__init__()
        self.input = nn.Linear(in_dim, hidden)
        self.layers = nn.ModuleList([GINELike(hidden, dropout) for _ in range(depth)])
        self.vn = nn.ModuleList([VirtualNodeBlock(hidden, dropout) for _ in range(max(0, depth - 1))])
    def forward_graph(self, X, A):
        dev = next(self.parameters()).device
        X = X.to(dev).float(); A = A.to(dev).float()
        h = F.relu(self.input(X))
        for i, conv in enumerate(self.layers):
            h = conv(h, A)
            if i < len(self.vn):
                h = self.vn[i](h)
        return h
    def encode_smiles(self, smi: str):
        X, A = smiles_to_graph(smi)
        return self.forward_graph(X, A)