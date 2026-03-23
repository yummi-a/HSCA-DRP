# utils.py
# -*- coding: utf-8 -*-

import time
import numpy as np
import torch
import pandas as pd
from pathlib import Path
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# RDKit 导入
try:
    from rdkit import Chem
    from rdkit.Chem.Pharm2D import Generate, Gobbi_Pharm2D

    _HAS_RDKIT = True
except ImportError:
    _HAS_RDKIT = False

# ---------- 路径配置 ----------
DATA_DIR = Path("./data")
F_EXPR = DATA_DIR / "GEP.csv"
F_COPY = DATA_DIR / "CNV.csv"
F_SMILES = DATA_DIR / "Drug_smiles.csv"
F_FP = DATA_DIR / "drug_fingerprints.csv"
F_MOR = DATA_DIR / "drug_physchem.csv"
F_IC50 = DATA_DIR / "ic50_Filtered.csv"

TOPK = 10
PCA_D = 128
GENE_D = 1000  # 【保持】使用 1000 个高变基因 Z-score


# ---------- 辅助函数 ----------
def _norm_cell_id(s: pd.Series): return s.astype(str).str.strip().str.upper()


def _norm_cid(s: pd.Series): return s.apply(pd.to_numeric, errors='coerce').fillna(0).astype(int)


def _pca(X: np.ndarray, d=PCA_D):
    X = np.asarray(X, dtype=np.float32)
    X = np.nan_to_num(X, 0.0, 0.0, 0.0)
    min_dim = min(X.shape)
    if min_dim < 2: return X
    if min_dim < d: d = min_dim
    X = StandardScaler().fit_transform(X)
    if X.shape[1] > d:
        pca = PCA(n_components=d, random_state=0)
        X = pca.fit_transform(X)
    return X


def _zscore_top_var(X: np.ndarray, n=GENE_D):
    """
    【保持】GEP 专用：选方差最大的 n 个基因做 Z-score
    """
    X = np.asarray(X, dtype=np.float32)
    X = np.nan_to_num(X, 0.0, 0.0, 0.0)

    if X.shape[1] <= n:
        return StandardScaler().fit_transform(X)

    vars = np.var(X, axis=0)
    top_indices = np.argpartition(vars, -n)[-n:]
    top_indices = np.sort(top_indices)

    X_sub = X[:, top_indices]
    print(f"[Data] Selected top {n} high-variance genes from {X.shape[1]} total.")
    return StandardScaler().fit_transform(X_sub)


def _cosine_topk_weighted(X: np.ndarray, k: int = TOPK) -> np.ndarray:
    X = np.asarray(X, dtype=np.float32)
    norm = np.linalg.norm(X, axis=1, keepdims=True)
    norm[norm == 0] = 1e-12
    Xn = X / norm
    sim = Xn @ Xn.T
    n = sim.shape[0]
    adj = np.zeros((n, n), dtype=np.float32)
    actual_k = min(k, n - 1)
    if actual_k <= 0: actual_k = n
    if actual_k >= n - 1:
        adj = sim
    else:
        topk = np.argpartition(sim, -actual_k, axis=1)[:, -actual_k:]
        rows = np.repeat(np.arange(n), actual_k)
        cols = topk.reshape(-1)
        adj[rows, cols] = sim[rows, cols]
    np.fill_diagonal(adj, 1.0)
    return adj


def _read_feat_with_header(csv_path: Path, id_col_name="CID"):
    df = pd.read_csv(csv_path, header=0, low_memory=False)
    if id_col_name in df.columns:
        cid = _norm_cid(df[id_col_name]).tolist()
        feat_df = df.drop(columns=[id_col_name])
    else:
        cid = _norm_cid(df.iloc[:, 0]).tolist()
        feat_df = df.iloc[:, 1:]
    mat = feat_df.apply(pd.to_numeric, errors="coerce").fillna(0.0).values.astype(np.float32)
    if len(cid) != len(set(cid)):
        keep = ~pd.Series(cid).duplicated(keep="first").values
        cid = [c for c, k in zip(cid, keep) if k]
        mat = mat[keep]
    return cid, mat


def _drug_screen_by_fp_mor(fp_ids, fp_mat, mor_ids, mor_mat, base_ids):
    base = set(base_ids)
    inter = sorted(base & set(fp_ids) & set(mor_ids))
    if not inter: raise RuntimeError("药物特征交集为空")

    fp_i = {c: i for i, c in enumerate(fp_ids)}
    mor_i = {c: i for i, c in enumerate(mor_ids)}
    fp = np.stack([fp_mat[fp_i[c]] for c in inter], axis=0).astype(np.float32)
    mor = np.stack([mor_mat[mor_i[c]] for c in inter], axis=0).astype(np.float32)
    X = np.concatenate([fp, mor], axis=1)

    if X.shape[0] > 5:
        Xz = StandardScaler().fit_transform(np.nan_to_num(X, 0.0, 0.0, 0.0))
        r = np.linalg.norm(Xz, axis=1)
        q1, q3 = np.percentile(r, [25, 75])
        iqr = q3 - q1
        low, high = q1 - 1.5 * iqr, q3 + 1.5 * iqr
        keep = (r >= low) & (r <= high)
        kept = [c for c, k in zip(inter, keep) if k]
    else:
        kept = inter

    print(f"[drug-screen] matched drugs: {len(inter)} -> kept: {len(kept)}", flush=True)
    return kept


def get_pharm_fingerprints(smiles_list):
    if not _HAS_RDKIT: return np.zeros((len(smiles_list), 128), dtype=np.float32)
    factory = Gobbi_Pharm2D.factory
    fps = []
    for s in smiles_list:
        try:
            mol = Chem.MolFromSmiles(s)
        except:
            mol = None
        if mol is None:
            fps.append(np.zeros(factory.GetSigSize(), dtype=np.int8))
        else:
            try:
                fp = Generate.Gen2DFingerprint(mol, factory)
                arr = np.frombuffer(fp.ToBitString().encode(), 'u1') - ord('0')
                fps.append(arr)
            except:
                fps.append(np.zeros(factory.GetSigSize(), dtype=np.int8))
    return np.stack(fps)


def multiomics_data():
    t0 = time.time()
    print(f"[info] Loading data from {DATA_DIR} ...")

    # 1. 细胞
    expr_df = pd.read_csv(F_EXPR, header=0, low_memory=False)
    cpy_df = pd.read_csv(F_COPY, header=0, low_memory=False)
    cell_col_expr = "cell_line" if "cell_line" in expr_df.columns else expr_df.columns[0]
    cell_col_cnv = "cell_line" if "cell_line" in cpy_df.columns else cpy_df.columns[0]
    expr_df["norm_id"] = _norm_cell_id(expr_df[cell_col_expr])
    cpy_df["norm_id"] = _norm_cell_id(cpy_df[cell_col_cnv])
    cells = sorted(set(expr_df["norm_id"]) & set(cpy_df["norm_id"]))
    if not cells: raise RuntimeError("GEP/CNV ID mismatch")
    expr_df = expr_df[expr_df["norm_id"].isin(cells)].sort_values("norm_id")
    cpy_df = cpy_df[cpy_df["norm_id"].isin(cells)].sort_values("norm_id")
    cell_ids = expr_df["norm_id"].tolist()
    drop_cols = ["CellName", "cell_line", "CosmicId", "Unnamed: 0", "Sequencing", "IsDefault", "norm_id"]

    def get_values(df, drops):
        return df.drop(columns=[c for c in drops if c in df.columns]).apply(pd.to_numeric, errors="coerce").fillna(
            0.0).values

    # 【GEP -> Z-score 1000】
    raw_expr = get_values(expr_df, drop_cols)
    cell_feat = _zscore_top_var(raw_expr, GENE_D)

    # 【CNV -> PCA -> Graph】
    raw_cnv = get_values(cpy_df, drop_cols)
    cpy_128 = _pca(raw_cnv, PCA_D)
    cell_adj = _cosine_topk_weighted(cpy_128, TOPK)

    # 2. 药物
    smi = pd.read_csv(F_SMILES, header=0, low_memory=False)
    smi_col = "Canonical_SMILES" if "Canonical_SMILES" in smi.columns else "CanonicalSMILES"
    smi["pubchem_cid"] = _norm_cid(smi["CID"])
    smi["smiles"] = smi[smi_col].astype(str)
    if "DRUG_NAME" in smi.columns:
        name_to_cid = pd.Series(smi.pubchem_cid.values, index=smi.DRUG_NAME.astype(str).str.strip()).to_dict()
    else:
        name_to_cid = {}
    smi = smi.dropna(subset=["pubchem_cid", "smiles"]).drop_duplicates(subset=["pubchem_cid"])
    base_ids = smi["pubchem_cid"].tolist()
    fp_ids, fp_mat = _read_feat_with_header(F_FP, id_col_name="CID")
    mor_ids, mor_mat = _read_feat_with_header(F_MOR, id_col_name="CID")

    kept_cids = _drug_screen_by_fp_mor(fp_ids, fp_mat, mor_ids, mor_mat, base_ids)

    smi = smi[smi["pubchem_cid"].isin(kept_cids)].sort_values("pubchem_cid")
    drug_ids = smi["pubchem_cid"].tolist()
    drug_smiles = smi["smiles"].tolist()
    fp_idx = {c: i for i, c in enumerate(fp_ids)}
    fp_take = np.stack([fp_mat[fp_idx[c]] for c in drug_ids], axis=0)
    fp_128 = _pca(fp_take, PCA_D)
    pharm_128 = _pca(get_pharm_fingerprints(drug_smiles), PCA_D)

    # 3. 标签
    lab = pd.read_csv(F_IC50, header=0, low_memory=False)
    lab_cell_col = "cell_line" if "cell_line" in lab.columns else "cell_name"
    lab_drug_col = "drug" if "drug" in lab.columns else "cid"
    lab["cell_id"] = _norm_cell_id(lab[lab_cell_col])

    def map_drug(val):
        if isinstance(val, (int, float)): return int(val)
        v = str(val).strip()
        return int(v) if v.isdigit() else name_to_cid.get(v, 0)

    lab["cid_norm"] = lab[lab_drug_col].apply(map_drug) if lab[lab_drug_col].dtype == object else _norm_cid(
        lab[lab_drug_col])
    lab["y"] = pd.to_numeric(lab["IC50"], errors="coerce")
    lab = lab[(lab["cid_norm"].isin(drug_ids)) & (lab["cell_id"].isin(cell_ids))].dropna(subset=["y"])

    y_raw = lab["y"].values.astype(float)
    cid2i = {c: i for i, c in enumerate(cell_ids)}
    did2i = {c: i for i, c in enumerate(drug_ids)}

    pairs = []
    for c, d, v in zip(lab["cell_id"], lab["cid_norm"], y_raw):
        pairs.append([cid2i[c], did2i[d], v])
    pairs = np.array(pairs, dtype=np.float32)

    cell_features = torch.from_numpy(cell_feat.astype(np.float32))
    cell_adj_t = torch.from_numpy(cell_adj.astype(np.float32))
    drug_mol_feat = torch.from_numpy(fp_128.astype(np.float32))
    drug_pharm_feat = torch.from_numpy(pharm_128.astype(np.float32))
    sample_set = torch.from_numpy(pairs)

    print(f"[data] Done. {time.time() - t0:.1f}s")
    return cell_features, cell_adj_t, drug_mol_feat, drug_pharm_feat, drug_smiles, sample_set