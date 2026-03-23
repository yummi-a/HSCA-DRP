# model.py
# -*- coding: utf-8 -*-

import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from layers import (
    GATLayer, GlobalTransformerBlock,
    DrugAtomEncoder
)

try:
    from tqdm import tqdm
except Exception:
    def tqdm(x, **k):
        return x


# ============== KAN 回归头 ==============
class KANHead(nn.Module):
    def __init__(self, in_dim: int, n_bins: int = 10, hidden: int = None,
                 dropout: float = 0.3, shortcut_gamma: float = 0.3,
                 use_gelu: bool = True):
        super().__init__()
        self.in_dim = in_dim
        self.n_bins = n_bins
        self.hidden = in_dim if hidden is None else hidden
        self.ln_in = nn.LayerNorm(in_dim)
        centers = torch.linspace(-2.0, 2.0, steps=n_bins).repeat(in_dim, 1)
        self.centers = nn.Parameter(centers)
        self.width = nn.Parameter(torch.ones(in_dim))
        self.softplus = nn.Softplus()
        basis_dim = in_dim * n_bins
        self.fc1 = nn.Linear(basis_dim, self.hidden)
        self.act = nn.GELU() if use_gelu else nn.ReLU(inplace=True)
        self.drop = nn.Dropout(dropout)
        self.head_mean = nn.Linear(self.hidden, 1)
        self.head_var = nn.Linear(self.hidden, 1)
        self.short = nn.Linear(in_dim, 1)
        self.gamma = nn.Parameter(torch.tensor(float(shortcut_gamma)))

    def _tri_basis(self, x: torch.Tensor) -> torch.Tensor:
        x = x.unsqueeze(-1)
        c = self.centers.unsqueeze(0)
        w = self.softplus(self.width).view(1, -1, 1) + 1e-6
        phi = torch.relu(1.0 - torch.abs((x - c) / w))
        return phi.reshape(phi.size(0), -1)

    def forward(self, x: torch.Tensor) -> (torch.Tensor, torch.Tensor):
        x_n = self.ln_in(x)
        z = self._tri_basis(x_n)
        h_kan = self.drop(self.act(self.fc1(z)))
        y_kan_mean = self.head_mean(h_kan)
        y_lin_mean = self.short(x_n)
        mean = y_kan_mean + torch.relu(self.gamma) * y_lin_mean
        log_var = self.head_var(h_kan)
        return mean, log_var


# ============== 纵向药物编码器 (Pure Hierarchical) ==============
class DrugEncoder(nn.Module):
    def __init__(self, ndrug, ndrugfeat, npharmfeat, D, drug_nheads, drug_smiles_list, device,
                 n_pharm_tokens=8):
        super().__init__()
        self.device = device
        self.D = D
        self.n_pharm_tokens = n_pharm_tokens

        atom_in_dim = len(['B', 'C', 'N', 'O', 'F', 'Si', 'P', 'S', 'Cl', 'Br', 'I', 'H', '*']) + 3
        # 1. Micro View: Atoms (GNN)
        self.drug_atom = DrugAtomEncoder(in_dim=atom_in_dim, hidden=D, depth=3, dropout=0.3).to(self.device)

        # 2. Meso View: Pharmacophores (Hierarchy Level 1)
        self.pharm_proj = nn.Linear(npharmfeat, n_pharm_tokens * D)
        self.atom_to_pharm_attn = nn.MultiheadAttention(embed_dim=D, num_heads=drug_nheads, batch_first=True)
        self.pharm_ln = nn.LayerNorm(D)

        # 3. Macro View: Physicochemical (Hierarchy Level 2)
        self.mol_feat_proj = nn.Sequential(
            nn.Linear(ndrugfeat, D), nn.ReLU(), nn.Linear(D, D)
        )
        self.pharm_to_macro_attn = nn.MultiheadAttention(embed_dim=D, num_heads=drug_nheads, batch_first=True)

        # Output Projection
        self.out_proj = nn.Sequential(
            nn.LayerNorm(D),
            nn.Linear(D, D),
            nn.GELU(),
            nn.LayerNorm(D)
        )

        self._init_drug_bank(ndrug, D, drug_smiles_list)

    def _init_drug_bank(self, ndrug, D, drug_smiles_list):
        os.makedirs("./output", exist_ok=True)
        cache_atom = "./output/hierdrp_drug_bank_seq.pt"
        bank_atom = None
        bank_mask = None

        if os.path.exists(cache_atom):
            try:
                ckpt = torch.load(cache_atom, map_location="cpu")
                if ckpt.get("D") == D and ckpt.get("N") == ndrug:
                    bank_atom = ckpt["bank"]
                    bank_mask = ckpt["mask"]
            except Exception:
                bank_atom = None

        if bank_atom is None:
            with torch.no_grad():
                vecs = []
                if drug_smiles_list is not None:
                    for smi in tqdm(drug_smiles_list, desc="Encode Atoms (HierDRP)"):
                        emb = self.drug_atom.encode_smiles(smi)
                        vecs.append(emb.cpu())
                    if len(vecs) > 0:
                        lens = [v.size(0) for v in vecs]
                        max_len = max(lens)
                        padded_vecs = []
                        masks = []
                        for v in vecs:
                            n = v.size(0)
                            pad_len = max_len - n
                            if pad_len > 0:
                                pv = F.pad(v, (0, 0, 0, pad_len))
                                m = torch.cat([torch.zeros(n, dtype=torch.bool), torch.ones(pad_len, dtype=torch.bool)])
                            else:
                                pv = v
                                m = torch.zeros(n, dtype=torch.bool)
                            padded_vecs.append(pv)
                            masks.append(m)
                        bank_atom = torch.stack(padded_vecs)
                        bank_mask = torch.stack(masks)
                        torch.save({"bank": bank_atom, "mask": bank_mask, "N": ndrug, "D": D}, cache_atom)
                else:
                    bank_atom = torch.zeros(ndrug, 1, D)
                    bank_mask = torch.zeros(ndrug, 1, dtype=torch.bool)

        self.register_buffer("_drug_bank_atom", bank_atom.to(self.device))
        self.register_buffer("_drug_bank_mask", bank_mask.to(self.device))

    def forward(self, d_idx, drug_mol_feat, drug_pharm_feat):
        # 1. Micro View
        atom_feats = self._drug_bank_atom[d_idx]
        atom_mask = self._drug_bank_mask[d_idx]

        if drug_mol_feat.shape[0] == self._drug_bank_atom.shape[0]:
            batch_mol = drug_mol_feat[d_idx]
            batch_pharm = drug_pharm_feat[d_idx]
        else:
            batch_mol = drug_mol_feat
            batch_pharm = drug_pharm_feat

        # --- Hierarchy Logic (Hardcoded) ---

        # Step 1: Meso (Pharm) query Micro (Atom)
        pharm_tokens = self.pharm_proj(batch_pharm)
        pharm_query = pharm_tokens.view(-1, self.n_pharm_tokens, self.D)

        meso_feats, _ = self.atom_to_pharm_attn(
            query=pharm_query, key=atom_feats, value=atom_feats, key_padding_mask=atom_mask
        )
        meso_feats = self.pharm_ln(pharm_query + meso_feats)

        # Step 2: Macro (PhysChem) query Meso (Pharm)
        macro_query = self.mol_feat_proj(batch_mol).unsqueeze(1)

        global_repr, _ = self.pharm_to_macro_attn(
            query=macro_query, key=meso_feats, value=meso_feats
        )
        final_repr = (macro_query + global_repr).squeeze(1)

        return self.out_proj(final_repr)


# ============== 主模型 (HierDRP) - Final Pure ==============
class HierDRP(nn.Module):
    def __init__(self, ncell, ndrug, ncellfeat, ndrugfeat, npharmfeat, nhid, nheads,
                 drug_nheads: int = 4,
                 n_pharm_tokens: int = 8,
                 use_global_cd_attn: bool = False,
                 global_cd_topk_hint: int = 32,
                 global_cd_bidirectional: bool = True,
                 drug_smiles_list=None,
                 device="cpu"):
        super().__init__()
        self.device = device
        self.ncell = int(ncell)
        self.ndrug = int(ndrug)
        D = nhid * nheads

        # ================== 1. 细胞侧 (No Gate) ==================
        self.cell_proj = nn.Linear(ncellfeat, D)

        # Layer 1: Local GAT + Global MHSA
        self.cell_gat1 = GATLayer(in_dim=D, out_dim=D, nheads=nheads)
        self.cell_norm1 = nn.LayerNorm(D)
        self.cell_attn1 = GlobalTransformerBlock(in_dim=D, nheads=nheads)

        # Layer 2: Local GAT + Global MHSA
        self.cell_gat2 = GATLayer(in_dim=D, out_dim=D, nheads=nheads)
        self.cell_norm2 = nn.LayerNorm(D)
        self.cell_attn2 = GlobalTransformerBlock(in_dim=D, nheads=nheads)

        self.final_ln = nn.LayerNorm(D)

        # ================== 2. 药物侧 (Hardcoded Hierarchical) ==================
        self.drug_encoder = DrugEncoder(
            ndrug=ndrug, ndrugfeat=ndrugfeat, npharmfeat=npharmfeat, D=D,
            drug_nheads=drug_nheads,
            drug_smiles_list=drug_smiles_list, device=device,
            n_pharm_tokens=n_pharm_tokens
        )

        # --- Head ---
        self.proj_c = nn.Linear(D, D)
        self.proj_d = nn.Linear(D, D)
        self.norm_c = nn.LayerNorm(D)
        self.norm_d = nn.LayerNorm(D)

        self.head = KANHead(in_dim=2 * D, n_bins=8, hidden=2 * D, dropout=0.3, shortcut_gamma=0.3)

        # ================== 3. 全局 Cell-Drug 调制 (Edge-Masked Cross-Attn) ==================
        # 目标：让所有细胞 token 与所有药物 CEO token 在二部图上做 cross-attn。
        # 为避免噪声，cross-attn 使用外部设置的 edge mask（仅在“有意义的边”上允许注意力）。
        self.use_global_cd_attn = bool(use_global_cd_attn)
        self.global_cd_bidirectional = bool(global_cd_bidirectional)
        self.global_cd_topk_hint = int(global_cd_topk_hint)
        if self.use_global_cd_attn:
            # cells <- drugs
            self.cd_attn = nn.MultiheadAttention(embed_dim=D, num_heads=nheads, batch_first=True)
            self.cd_ln = nn.LayerNorm(D)
            self.cd_drop = nn.Dropout(0.1)
            # optional: drugs <- cells
            if self.global_cd_bidirectional:
                self.dc_attn = nn.MultiheadAttention(embed_dim=D, num_heads=nheads, batch_first=True)
                self.dc_ln = nn.LayerNorm(D)
                self.dc_drop = nn.Dropout(0.1)
            else:
                self.dc_attn = None
                self.dc_ln = None
                self.dc_drop = None

            # 注意力 mask：float tensor，shape [N_cells, N_drugs]，masked 位置为 -inf
            #（MultiheadAttention 的 float mask 是 additive mask）
            self._cd_attn_mask = None
            self._dc_attn_mask = None

        if self.use_global_cd_attn:
            extra = f" + Global Cell↔Drug CrossAttn(masked, bidir={self.global_cd_bidirectional}, topk_hint={self.global_cd_topk_hint})"
        else:
            extra = ""
        print(f"[Model] HierDRP Final. Cell: Global MHSA (No Gate). Drug: Hierarchical.{extra}")
        if self.use_global_cd_attn:
            bi = "bi" if self.global_cd_bidirectional else "uni"
            print(f"[Model] Global Cell-Drug Modulation: ON ({bi}-directional, topk_hint={self.global_cd_topk_hint}).")

    def encode_cells(self, cell_features, cell_adj):
        x = self.cell_proj(cell_features)

        # Layer 1
        h_gat = self.cell_gat1(x, cell_adj)
        x = self.cell_norm1(x + h_gat)
        x = self.cell_attn1(x)

        # Layer 2
        h_gat2 = self.cell_gat2(x, cell_adj)
        x = self.cell_norm2(x + h_gat2)
        x = self.cell_attn2(x)

        return self.final_ln(x)

    @torch.no_grad()
    def set_global_edge_mask(self, allow_edges: torch.Tensor):
        """设置全局 cell-drug cross-attn 的允许边。

        Parameters
        ----------
        allow_edges : torch.Tensor
            bool tensor of shape [N_cells, N_drugs]. True 表示允许注意力。

        Notes
        -----
        MultiheadAttention 的 attn_mask：
          - float mask: additive（masked 位置加 -inf）
          - bool mask: True 表示 masked
        这里我们转成 float additive mask，避免不同版本 PyTorch 的 bool 语义差异。
        """
        if not self.use_global_cd_attn:
            return
        if allow_edges.dtype != torch.bool:
            allow_edges = allow_edges.bool()
        if allow_edges.dim() != 2:
            raise ValueError(f"allow_edges must be [N_cells, N_drugs], got {tuple(allow_edges.shape)}")
        if allow_edges.size(0) != self.ncell or allow_edges.size(1) != self.ndrug:
            raise ValueError(
                f"allow_edges shape mismatch: expected ({self.ncell},{self.ndrug}) got {tuple(allow_edges.shape)}"
            )

        dev = next(self.parameters()).device
        allow_edges = allow_edges.to(dev)

        # float additive mask: 0 for allowed, -inf for masked
        neg_inf = torch.finfo(torch.float32).min
        cd_mask = torch.zeros((self.ncell, self.ndrug), device=dev, dtype=torch.float32)
        cd_mask[~allow_edges] = neg_inf
        self._cd_attn_mask = cd_mask
        if self.global_cd_bidirectional:
            dc_mask = torch.zeros((self.ndrug, self.ncell), device=dev, dtype=torch.float32)
            dc_mask[~allow_edges.t()] = neg_inf
            self._dc_attn_mask = dc_mask

    def forward(self, cell_features, cell_adj, drug_mol_feat, drug_pharm_feat, idx_cell_drug, device):
        cell_emb_all = self.encode_cells(cell_features, cell_adj)

        # ===== 全局 Cell-Drug 调制（edge-masked cross-attn） =====
        if self.use_global_cd_attn:
            # 1) 计算所有 drug 的 CEO token（[N_drugs, D]）
            all_d = torch.arange(self.ndrug, device=cell_emb_all.device, dtype=torch.long)
            drug_emb_all = self.drug_encoder(all_d, drug_mol_feat, drug_pharm_feat)

            # 2) cells <- drugs
            q = cell_emb_all.unsqueeze(0)       # [1, N_cells, D]
            k = drug_emb_all.unsqueeze(0)       # [1, N_drugs, D]
            attn_mask = self._cd_attn_mask  # [N_cells, N_drugs] or None
            ctx, _ = self.cd_attn(q, k, k, attn_mask=attn_mask)
            cell_emb_all = self.cd_ln(q + self.cd_drop(ctx)).squeeze(0)

            # 3) optional: drugs <- cells
            if self.global_cd_bidirectional:
                qd = drug_emb_all.unsqueeze(0)          # [1, N_drugs, D]
                kc = cell_emb_all.unsqueeze(0)          # [1, N_cells, D]
                ctxd, _ = self.dc_attn(qd, kc, kc, attn_mask=self._dc_attn_mask)
                drug_emb_all = self.dc_ln(qd + self.dc_drop(ctxd)).squeeze(0)
        else:
            drug_emb_all = None

        if isinstance(idx_cell_drug, np.ndarray):
            idx = torch.from_numpy(idx_cell_drug).long().to(cell_emb_all.device)
        else:
            idx = torch.as_tensor(idx_cell_drug, dtype=torch.long, device=cell_emb_all.device)

        c = cell_emb_all[idx[:, 0]]
        if drug_emb_all is None:
            d = self.drug_encoder(idx[:, 1], drug_mol_feat, drug_pharm_feat)
        else:
            d = drug_emb_all[idx[:, 1]]

        pair = torch.cat([self.norm_c(self.proj_c(c)), self.norm_d(self.proj_d(d))], dim=-1)
        mean, log_var = self.head(pair)
        return mean.squeeze(-1), log_var.squeeze(-1)