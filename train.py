import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

import random
import warnings
import torch
import torch.nn as nn
import torch.optim as optim
from torch_geometric.data import Batch
from torch_geometric.loader import DataLoader
from torch.serialization import safe_globals
from torch_geometric.data.data import DataEdgeAttr, DataTensorAttr, GlobalStorage
from tqdm import tqdm
import numpy as np

from model import MSADF_DTA
from utils import AverageMeter
from metrics import get_cindex
from log.train_logger import TrainLogger

warnings.filterwarnings("ignore", category=UserWarning, module="torch.cuda.nccl")
warnings.filterwarnings(
    "ignore",
    ".*adaptive_max_pool2d_backward_cuda.*deterministic implementation.*",
    module="torch.autograd.graph"
)

# ============================================================
# 固定随机种子
# ============================================================
def set_seed(seed=42):
    os.environ["PYTHONHASHSEED"] = str(seed)
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True, warn_only=True)

def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)

# ============================================================
# Dataset
# ============================================================
class MultiGraphDataset(torch.utils.data.Dataset):
    def __init__(self, root, split='train'):
        print(f"[Dataset] 开始加载 {split} 集数据...")
        with safe_globals([DataEdgeAttr, DataTensorAttr, GlobalStorage]):
            self.atomic_list = torch.load(os.path.join(root, 'processed', f'atomic_{split}.pt'), weights_only=True)
            self.brics_list = torch.load(os.path.join(root, 'processed', f'brics_{split}.pt'), weights_only=True)
            self.protein_list = torch.load(os.path.join(root, 'processed', f'protein_{split}.pt'), weights_only=True)
        print(f"[Dataset] {split} 集加载完成：atomic {len(self.atomic_list)}, brics {len(self.brics_list)}, protein {len(self.protein_list)}")

    def __len__(self):
        return len(self.atomic_list)

    def __getitem__(self, idx):
        atomic_data = self.atomic_list[idx]
        brics_data = self.brics_list[idx]
        target, y = self.protein_list[idx]
        return atomic_data, brics_data, target, y

def collate_fn(batch):
    atomic_list, brics_list, target_list, y_list = zip(*batch)
    atomic_batch = Batch.from_data_list(atomic_list)
    brics_batch = Batch.from_data_list(brics_list)
    target_batch = torch.cat(target_list, dim=0)
    y_batch = torch.cat(y_list, dim=0)
    return atomic_batch, brics_batch, target_batch, y_batch

# ============================================================
# 验证函数
# ============================================================
def val(model, criterion, dataloader, device):
    model.eval()
    running_loss = AverageMeter()
    y_true, y_pred = [], []

    with torch.no_grad():
        for atomic_batch, brics_batch, target_batch, y_batch in dataloader:
            atomic_batch, brics_batch = atomic_batch.to(device), brics_batch.to(device)
            target_batch, y_batch = target_batch.to(device), y_batch.to(device)
            pred = model(atomic_batch, brics_batch, target_batch)
            loss = criterion(pred.view(-1), y_batch.view(-1))
            running_loss.update(loss.item(), y_batch.size(0))
            y_true.append(y_batch.detach().cpu().numpy().reshape(-1))
            y_pred.append(pred.detach().cpu().numpy().reshape(-1))

    y_true = np.concatenate(y_true)
    y_pred = np.concatenate(y_pred)
    epoch_loss = running_loss.get_average()
    test_cindex = get_cindex(y_true, y_pred)
    running_loss.reset()
    model.train()
    return epoch_loss, test_cindex

# ============================================================
# CSV 保存
# ============================================================
def save_log_csv(csv_path, epoch, train_loss, train_ci, val_loss=None, val_ci=None):
    import csv
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    new_file = not os.path.exists(csv_path)
    with open(csv_path, "a", newline="") as f:
        writer = csv.writer(f)
        if new_file:
            writer.writerow(["epoch","train_loss","train_ci","val_loss","val_ci"])
        writer.writerow([epoch, train_loss, train_ci, val_loss, val_ci])

# ============================================================
# 主函数
# ============================================================
def main():
    set_seed(42)

    datasets = ["davis", "kiba"]
    data_root = "./data"
    lr = 5e-4
    batch_size = 512
    epochs = 600
    embedding_size = 128
    filter_num = 32
    out_dim = 1

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # ============================================================
    # 循环两个数据集
    # ============================================================
    for dataset in datasets:
        print(f"\n🚀 Start training on dataset: {dataset}")

        logger = TrainLogger(dict(
            dataset=dataset,
            lr=lr,
            batch_size=batch_size,
            save_dir=os.path.join(data_root, "save")
        ))

        dataset_path = os.path.join(data_root, dataset)

        train_set = MultiGraphDataset(dataset_path, split="train")
        val_set = MultiGraphDataset(dataset_path, split="test")

        train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
        val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)

        csv_path = os.path.join(logger.get_model_dir(), "metrics.csv")

        model = MSADF_DTA(3, 26, embedding_size=embedding_size,
                          filter_num=filter_num, out_dim=out_dim).to(device)

        optimizer = optim.Adam(model.parameters(), lr=lr)
        criterion = nn.MSELoss()

        best_val_loss = float("inf")

        best_state = None


        for epoch in range(1, epochs + 1):
            model.train()
            running_loss = AverageMeter()
            running_ci = AverageMeter()

            for atomic, brics, target, y in train_loader:
                atomic, brics = atomic.to(device), brics.to(device)
                target, y = target.to(device), y.to(device)

                optimizer.zero_grad()
                pred = model(atomic, brics, target)
                loss = criterion(pred.view(-1), y.view(-1))
                loss.backward()
                optimizer.step()

                running_loss.update(loss.item(), y.size(0))
                running_ci.update(
                    get_cindex(
                        y.detach().cpu().numpy().reshape(-1),
                        pred.detach().cpu().numpy().reshape(-1)
                    ),
                    y.size(0)
                )

            val_loss, val_ci = val(model, criterion, val_loader, device)

            save_log_csv(csv_path, epoch,
                         running_loss.get_average(),
                         running_ci.get_average(),
                         val_loss, val_ci)

            if val_loss < best_val_loss:
                best_val_loss = val_loss

                best_state = model.state_dict()

        torch.save(best_state,
                   os.path.join(logger.get_model_dir(), "best_model.pt"))

        print(f"[Best] val_loss={best_val_loss:.4f}")


if __name__ == "__main__":
    main()
