import os
import copy
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from torch_geometric.data import DataLoader, Batch
from torch.serialization import safe_globals
from torch_geometric.data.data import DataEdgeAttr, DataTensorAttr, GlobalStorage
from tqdm import tqdm
from sklearn.metrics import r2_score, precision_recall_curve, auc
import warnings

warnings.filterwarnings('ignore')

from M import MGraphDTA
from utils import AverageMeter
from metrics import get_cindex

# ============================================================
# Dataset
# ============================================================
class MultiGraphDataset(torch.utils.data.Dataset):
    def __init__(self, root, csv_path, split='test'):
        with safe_globals([DataEdgeAttr, DataTensorAttr, GlobalStorage]):
            self.atomic_list = torch.load(os.path.join(root, 'processed', f'atomic_{split}.pt'), weights_only=True)
            self.brics_list = torch.load(os.path.join(root, 'processed', f'brics_{split}.pt'), weights_only=True)
            self.protein_list = torch.load(os.path.join(root, 'processed', f'protein_{split}.pt'), weights_only=True)

        df = pd.read_csv(csv_path)
        self.smiles_list = df["compound_iso_smiles"].tolist()
        self.protein_seq_list = df["target_sequence"].tolist()

        assert len(self.atomic_list) == len(self.smiles_list), "processed 数据与 CSV 行数不一致"

    def __len__(self):
        return len(self.atomic_list)

    def __getitem__(self, idx):
        atomic_data = self.atomic_list[idx]
        brics_data = self.brics_list[idx]
        target, y = self.protein_list[idx]
        smiles = self.smiles_list[idx]
        protein_seq = self.protein_seq_list[idx]
        return atomic_data, brics_data, target, y, smiles, protein_seq


# ============================================================
# collate_fn
# ============================================================
def collate_fn(batch):
    atomic_list, brics_list, target_list, y_list, smiles_list, protein_list = zip(*batch)
    atomic_batch = Batch.from_data_list(atomic_list)
    brics_batch = Batch.from_data_list(brics_list)
    target_batch = torch.cat(target_list, dim=0)
    y_batch = torch.cat(y_list, dim=0)
    return atomic_batch, brics_batch, target_batch, y_batch, smiles_list, protein_list


# ============================================================
# CSV 保存函数
# ============================================================
def save_predictions_csv(smiles, proteins, y_true, y_pred, threshold, save_path):
    y_true_array = np.array(y_true)
    y_pred_array = np.array(y_pred)
    y_true_binary = (y_true_array > threshold).astype(int)
    y_pred_binary = (y_pred_array > threshold).astype(int)

    df = pd.DataFrame({
        "compound_iso_smiles": smiles,
        "target_sequence": proteins,
        "y_true": y_true,
        "y_pred": y_pred,
        "y_true_binary": y_true_binary.tolist(),
        "y_pred_binary": y_pred_binary.tolist()
    })
    df.to_csv(save_path, index=False)



# ============================================================
# 通用评估函数
# ============================================================
def evaluate(models, dataloader, device, threshold=12.1, save_csv_path=None):
    """
    models: list of nn.Module, 如果是单模型也可以传 [model]
    """
    for m in models:
        m.eval()

    criterion = nn.MSELoss()
    running_loss = AverageMeter()
    y_true, y_pred = [], []
    all_smiles, all_proteins = [], []

    with torch.no_grad():
        for atomic_batch, brics_batch, target_batch, y_batch, smiles_list, protein_list in tqdm(dataloader):
            atomic_batch = atomic_batch.to(device)
            brics_batch = brics_batch.to(device)
            target_batch = target_batch.to(device)
            y_batch = y_batch.to(device)

            preds = []
            for m in models:
                atomic_copy = copy.deepcopy(atomic_batch)
                brics_copy = copy.deepcopy(brics_batch)
                p = m(atomic_copy, brics_copy, target_batch).view(-1)
                preds.append(p)

            pred_mean = torch.stack(preds, dim=0).mean(dim=0)
            loss = criterion(pred_mean, y_batch.view(-1))
            running_loss.update(loss.item(), y_batch.size(0))

            y_true.extend(y_batch.cpu().numpy().reshape(-1))
            y_pred.extend(pred_mean.cpu().numpy().reshape(-1))
            all_smiles.extend(smiles_list)
            all_proteins.extend(protein_list)

    # 指标计算
    y_true_array = np.array(y_true)
    y_pred_array = np.array(y_pred)
    mse = running_loss.get_average()
    ci = get_cindex(y_true_array, y_pred_array)
    r2 = r2_score(y_true_array, y_pred_array)

    y_true_binary = (y_true_array >= threshold).astype(int)
    precision, recall, _ = precision_recall_curve(y_true_binary, y_pred_array)
    aupr = auc(recall, precision)

    # CSV 保存
    if save_csv_path is not None:
        save_predictions_csv(all_smiles, all_proteins, y_true, y_pred, threshold, save_csv_path)

    return mse, ci, r2, aupr


# ============================================================
# Main
# ============================================================
if __name__ == "__main__":
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # ----------------------------
    # 配置（直接相对路径）
    # ----------------------------

    dataset = "davis"
    csv_path = f"data/{dataset}/raw/data_test.csv"

    batch_size = 512
    embedding_size = 128
    filter_num = 32
    out_dim = 1
    n_folds = 5

    model_dir = f"data/save/{dataset}/model"

    save_csv_dir = f"save/{dataset}_predictions_csv"
    os.makedirs(save_csv_dir, exist_ok=True)

    # ----------------------------
    # 测试集
    # ----------------------------
    test_set = MultiGraphDataset(f"data/{dataset}", csv_path, split="test")
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)

    # ----------------------------
    # 单模型测试
    # ----------------------------
    fold_models = []
    fold_results = []
    threshold = 7.0  # 二分类阈值

    print("\n========== Fold Models Test ==========")
    print("=" * 60)
    print(f"{'Fold':<10} {'MSE':<12} {'CI':<12} {'R²':<12} {'AUPR':<12}")
    print("-" * 60)

    for k in range(1, n_folds + 1):
        model_path = os.path.join(model_dir, f"fold{k}_best.pt")
        model = MGraphDTA(3, 25 + 1, embedding_size=embedding_size, filter_num=filter_num, out_dim=out_dim).to(device)
        model.load_state_dict(torch.load(model_path, map_location=device))
        fold_models.append(model)

        csv_save_path = os.path.join(save_csv_dir, f"fold{k}_predictions.csv")
        mse, ci, r2, aupr = evaluate([model], test_loader, device, threshold, csv_save_path)
        fold_results.append((mse, ci, r2, aupr))

        print(f"{f'Fold{k}':<10} {mse:<12.3f} {ci:<12.3f} {r2:<12.3f} {aupr:<12.3f}")

    # ----------------------------
    # 平均折指标
    # ----------------------------
    avg_mse = np.mean([x[0] for x in fold_results])
    avg_ci = np.mean([x[1] for x in fold_results])
    avg_r2 = np.mean([x[2] for x in fold_results])
    avg_aupr = np.mean([x[3] for x in fold_results])

    std_mse = np.std([x[0] for x in fold_results])
    std_ci = np.std([x[1] for x in fold_results])
    std_r2 = np.std([x[2] for x in fold_results])
    std_aupr = np.std([x[3] for x in fold_results])

    print("\n" + "=" * 60)
    print("[Average of 5 folds (single-model)]")
    print("-" * 60)
    print(f"{'Metric':<10} {'Mean':<12} {'Std':<12}")
    print("-" * 60)
    print(f"{'MSE':<10} {avg_mse:<12.3f} {std_mse:<12.3f}")
    print(f"{'CI':<10} {avg_ci:<12.3f} {std_ci:<12.3f}")
    print(f"{'R²':<10} {avg_r2:<12.3f} {std_r2:<12.3f}")
    print(f"{'AUPR':<10} {avg_aupr:<12.3f} {std_aupr:<12.3f}")

    # ----------------------------
    # Ensemble 测试
    # ----------------------------
    ensemble_csv_path = os.path.join(save_csv_dir, "ensemble_predictions.csv")
    ens_mse, ens_ci, ens_r2, ens_aupr = evaluate(fold_models, test_loader, device, threshold, ensemble_csv_path)

    print("\n" + "=" * 60)
    print("[Ensemble (mean prediction of 5 models)]")
    print("-" * 60)
    print(f"{'MSE':<10} {ens_mse:<12.3f}")
    print(f"{'CI':<10} {ens_ci:<12.3f}")
    print(f"{'R²':<10} {ens_r2:<12.3f}")
    print(f"{'AUPR':<10} {ens_aupr:<12.3f}")
    print("=" * 60)

    # ----------------------------
    # 保存汇总结果
    # ----------------------------
    summary_csv_path = os.path.join(save_csv_dir, "summary_results.csv")
    summary_data = []

    for k in range(1, n_folds + 1):
        mse, ci, r2, aupr = fold_results[k - 1]
        summary_data.append({"Model": f"Fold{k}", "MSE": mse, "CI": ci, "R2": r2, "AUPR": aupr})

    summary_data.append({"Model": "Average", "MSE": avg_mse, "CI": avg_ci, "R2": avg_r2, "AUPR": avg_aupr})
    summary_data.append({"Model": "Ensemble", "MSE": ens_mse, "CI": ens_ci, "R2": ens_r2, "AUPR": ens_aupr})

    pd.DataFrame(summary_data).to_csv(summary_csv_path, index=False)
    print(f"\n[INFO] 汇总结果已保存到: {summary_csv_path}")