# run_HierDRP_ensemble.py
# -*- coding: utf-8 -*-

import os, argparse, time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data as Dataset
from torch.autograd import Variable
from torch.optim.lr_scheduler import ReduceLROnPlateau
from sklearn.model_selection import KFold, train_test_split
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from scipy.stats import pearsonr

try:
    from tqdm import tqdm
except Exception:
    def tqdm(x, **k):
        return x

from utils import multiomics_data
from model import HierDRP

# ===== Args =====
parser = argparse.ArgumentParser(description="HierDRP Auto-Ensemble (Macro + Micro)")
parser.add_argument('--seed', type=int, default=0)
parser.add_argument('--epochs', type=int, default=100)
parser.add_argument('--lr', type=float, default=1e-4)
parser.add_argument('--wd', type=float, default=1e-5)
parser.add_argument('--batch', type=int, default=128)
parser.add_argument('--hidden', type=int, default=128)
parser.add_argument('--nb_heads', type=int, default=4)
parser.add_argument('--patience', type=int, default=15)
parser.add_argument('--drug_nheads', type=int, default=4)
parser.add_argument('--n_pharm_tokens', type=int, default=4)
parser.add_argument('--folds', type=int, default=5)

# --- Ensemble Configs ---
parser.add_argument('--k_macro', type=int, default=10, help="Expand K for Macro Model")
parser.add_argument('--sim_macro', type=int, default=10, help="Sim K for Macro Model")

parser.add_argument('--k_micro', type=int, default=4, help="Expand K for Micro Model")
parser.add_argument('--sim_micro', type=int, default=4, help="Sim K for Micro Model")

parser.add_argument('--rank_coef', type=float, default=0.2, help="Ranking loss coefficient")

args = parser.parse_args()

WARMUP_EPOCHS = 20
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Set Global Seed
np.random.seed(args.seed)
torch.manual_seed(args.seed)
if torch.cuda.is_available(): torch.cuda.manual_seed(args.seed)


# ===== 辅助函数 =====
def compute_metrics(trues, preds):
    trues, preds = np.array(trues), np.array(preds)
    rmse = np.sqrt(mean_squared_error(trues, preds))
    mae = mean_absolute_error(trues, preds)
    pcc = pearsonr(trues, preds)[0] if np.std(preds) > 1e-9 else 0.0
    r2 = r2_score(trues, preds)
    return rmse, mae, pcc, r2


def _row_normalize(mat, eps=1e-12):
    return mat / mat.sum(dim=1, keepdim=True).clamp_min(eps)


def _topk_filter(sim, k):
    n = sim.size(0)
    k = min(max(k, 1), n)
    vals, idx = torch.topk(sim, k=k, dim=1)
    out = torch.zeros_like(sim)
    rows = torch.arange(n, device=sim.device).unsqueeze(1).expand_as(idx)
    out[rows, idx] = vals
    out.fill_diagonal_(1.0)
    return out


def build_edge_mask(train_pairs, cell_adj, drug_mol, drug_pharm, expand_k, sim_k):
    # (复用之前的逻辑)
    train_pairs = train_pairs.detach().cpu()
    cell_adj = cell_adj.detach().cpu()
    drug_mol = drug_mol.detach().cpu()
    drug_pharm = drug_pharm.detach().cpu()

    ncell, ndrug = int(cell_adj.size(0)), int(drug_mol.size(0))
    A = torch.zeros((ncell, ndrug), dtype=torch.bool)
    A[train_pairs[:, 0].long(), train_pairs[:, 1].long()] = True

    S_cell = _row_normalize(torch.relu(cell_adj.float()))

    Xm = F.normalize(drug_mol.float(), dim=1)
    Xp = F.normalize(drug_pharm.float(), dim=1)
    S_drug = _topk_filter(torch.relu(0.5 * (Xm @ Xm.t() + Xp @ Xp.t())), k=sim_k)
    S_drug = _row_normalize(S_drug)

    score = S_cell @ A.float() @ S_drug
    k = min(max(expand_k, 1), ndrug)
    topk_idx = torch.topk(score, k=k, dim=1).indices
    allow = A.clone()
    allow.scatter_(1, topk_idx, True)

    col_sum = allow.sum(dim=0)
    if (col_sum == 0).any():
        best_cells = torch.argmax(score, dim=0)
        zeros = torch.nonzero(col_sum == 0, as_tuple=False).view(-1)
        allow[best_cells[zeros], zeros] = True

    return allow


def compute_ranking_loss(preds, targets, drug_ids):
    drug_ids = drug_ids.view(-1, 1)
    matches = (drug_ids == drug_ids.T)
    matches.fill_diagonal_(False)
    if not matches.any(): return torch.tensor(0.0, device=preds.device)
    y_diff = targets.view(-1, 1) - targets.view(1, -1)
    p_diff = preds.view(-1, 1) - preds.view(1, -1)
    valid_mask = matches & (torch.abs(y_diff) > 1e-4)
    if not valid_mask.any(): return torch.tensor(0.0, device=preds.device)
    s_ij = torch.sign(y_diff)
    loss = torch.log1p(torch.exp(-s_ij * p_diff))
    return loss[valid_mask].mean()


# ===== 核心训练函数 (带进度打印版) =====
def train_single_model(name, expand_k, sim_k,
                       train_loader, val_loader, test_loader,
                       train_pairs_for_mask, fold_idx,
                       cell_feats, cell_adj, drug_mol, drug_pharm, drug_smiles):
    print(f"  > Training [{name}] (TopK={expand_k}, SimK={sim_k}) ...")

    model = HierDRP(ncell=cell_feats.shape[0], ndrug=len(drug_smiles),
                    ncellfeat=cell_feats.shape[1], ndrugfeat=drug_mol.shape[1],
                    npharmfeat=drug_pharm.shape[1], nhid=args.hidden, nheads=args.nb_heads,
                    drug_nheads=args.drug_nheads, n_pharm_tokens=args.n_pharm_tokens,
                    use_global_cd_attn=True,
                    global_cd_topk_hint=expand_k,
                    global_cd_bidirectional=True,
                    drug_smiles_list=drug_smiles, device=device).to(device)

    # 修正后的参数传递
    allow_edges = build_edge_mask(train_pairs_for_mask, cell_adj, drug_mol, drug_pharm,
                                  expand_k=expand_k, sim_k=sim_k)
    model.set_global_edge_mask(allow_edges)

    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.wd)
    scheduler = ReduceLROnPlateau(optimizer, 'min', factor=0.5, patience=3)

    best_rmse, best_ep = 1e9, 0
    save_path = f"./output/ensemble_{name}_fold{fold_idx}.pkl"

    for epoch in range(args.epochs):
        model.train()
        for idx_b, y_b in train_loader:
            y_b = y_b.to(device);
            idx_b = idx_b.to(device)
            mean, logvar = model(cell_feats, cell_adj, drug_mol, drug_pharm, idx_b, device)
            p = mean.view(-1);
            y = y_b.view(-1)

            if epoch < WARMUP_EPOCHS:
                loss = F.mse_loss(p, y)
            else:
                var = torch.exp(torch.clamp(logvar, -5, 5))
                loss = torch.mean(0.5 * logvar + ((y - p) ** 2) / (2 * var + 1e-6))

            loss += args.rank_coef * compute_ranking_loss(p, y, idx_b[:, 1])

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # Val
        model.eval()
        v_preds, v_trues = [], []
        with torch.no_grad():
            for idx_b, y_b in val_loader:
                idx_b = idx_b.to(device)
                p, _ = model(cell_feats, cell_adj, drug_mol, drug_pharm, idx_b, device)
                v_preds.extend(p.cpu().numpy());
                v_trues.extend(y_b.cpu().numpy())

        val_rmse = np.sqrt(mean_squared_error(v_trues, v_preds))
        scheduler.step(val_rmse)

        # 进度打印
        if (epoch + 1) % 5 == 0:
            print(f"    Ep {epoch + 1:03d} | Val RMSE: {val_rmse:.4f}")

        if val_rmse < best_rmse:
            best_rmse, best_ep = val_rmse, epoch + 1
            torch.save(model.state_dict(), save_path)
        elif epoch + 1 - best_ep >= args.patience:
            print(f"    Early stop at Ep {epoch + 1}")
            break

    model.load_state_dict(torch.load(save_path, map_location=device))
    model.eval()
    t_preds, t_trues = [], []
    with torch.no_grad():
        for idx_b, y_b in test_loader:
            idx_b = idx_b.to(device)
            p, _ = model(cell_feats, cell_adj, drug_mol, drug_pharm, idx_b, device)
            t_preds.extend(p.cpu().numpy());
            t_trues.extend(y_b.cpu().numpy())

    return np.array(t_preds), np.array(t_trues), best_rmse


# ===== 主程序 =====
print("[info] Loading data...", flush=True)
cell_features, cell_adj, drug_mol_feat, drug_pharm_feat, drug_smiles, sample_set = multiomics_data()

# Move static data to GPU
cell_features = Variable(cell_features).to(device)
cell_adj = Variable(cell_adj).to(device)
drug_mol_feat = Variable(drug_mol_feat).to(device)
drug_pharm_feat = Variable(drug_pharm_feat).to(device)
sample_set_cpu = sample_set.cpu().numpy()

kf = KFold(n_splits=args.folds, shuffle=True, random_state=args.seed)
results = {'Macro': [], 'Micro': [], 'Ensemble': []}

print(f"\n{'=' * 15} Auto-Ensemble: Macro(k={args.k_macro}) + Micro(k={args.k_micro}) {'=' * 15}")

for fold, (train_val_idx, test_idx) in enumerate(kf.split(sample_set_cpu)):
    print(f"\n>>> Fold {fold + 1} / {args.folds}")

    train_idx, valid_idx = train_test_split(train_val_idx, test_size=0.1, random_state=args.seed)


    # Build Loaders
    def get_loader(idx, shuf):
        d = torch.FloatTensor(sample_set_cpu[idx])
        return Dataset.DataLoader(Dataset.TensorDataset(d[:, :2], d[:, 2]), batch_size=args.batch, shuffle=shuf)


    tra_loader = get_loader(train_idx, True)
    val_loader = get_loader(valid_idx, False)
    te_loader = get_loader(test_idx, False)

    train_pairs_for_mask = torch.FloatTensor(sample_set_cpu[train_idx])[:, :2]

    # --- 1. Train Macro Model ---
    pred_A, y_true, val_rmse_A = train_single_model(
        "Macro", args.k_macro, args.sim_macro,
        tra_loader, val_loader, te_loader, train_pairs_for_mask, fold + 1,
        cell_features, cell_adj, drug_mol_feat, drug_pharm_feat, drug_smiles
    )

    # --- 2. Train Micro Model ---
    pred_B, _, val_rmse_B = train_single_model(
        "Micro", args.k_micro, args.sim_micro,
        tra_loader, val_loader, te_loader, train_pairs_for_mask, fold + 1,
        cell_features, cell_adj, drug_mol_feat, drug_pharm_feat, drug_smiles
    )

    # --- 3. Ensemble ---
    pred_Ens = (pred_A + pred_B) / 2.0

    # Metrics
    m_A = compute_metrics(y_true, pred_A)
    m_B = compute_metrics(y_true, pred_B)
    m_E = compute_metrics(y_true, pred_Ens)

    results['Macro'].append(m_A)
    results['Micro'].append(m_B)
    results['Ensemble'].append(m_E)

    # 【修改点1】这里打印每一个 Fold 的详细指标
    print(f"\n[Fold {fold + 1} Summary]")
    print(f"  Macro (Top{args.k_macro}) : RMSE={m_A[0]:.4f} MAE={m_A[1]:.4f} PCC={m_A[2]:.4f} R2={m_A[3]:.4f}")
    print(f"  Micro (Top{args.k_micro})  : RMSE={m_B[0]:.4f} MAE={m_B[1]:.4f} PCC={m_B[2]:.4f} R2={m_B[3]:.4f}")
    print(f"  >> ENSEMBLE  : RMSE={m_E[0]:.4f} MAE={m_E[1]:.4f} PCC={m_E[2]:.4f} R2={m_E[3]:.4f}")

# ===== Final Summary =====
# 【修改点2】这里打印带 ± 的最终汇总
print(f"\n{'=' * 20} 5-Fold Final Report {'=' * 20}")
print(f"{'Model':<10} | {'RMSE':<20} | {'MAE':<20} | {'PCC':<20} | {'R2':<20}")
print("-" * 100)

for name in ['Macro', 'Micro', 'Ensemble']:
    res = np.array(results[name])
    means = np.mean(res, axis=0)
    stds = np.std(res, axis=0)

    print(
        f"{name:10s} | {means[0]:.4f} ± {stds[0]:.4f} | {means[1]:.4f} ± {stds[1]:.4f} | {means[2]:.4f} ± {stds[2]:.4f} | {means[3]:.4f} ± {stds[3]:.4f}")
print("=" * 100)