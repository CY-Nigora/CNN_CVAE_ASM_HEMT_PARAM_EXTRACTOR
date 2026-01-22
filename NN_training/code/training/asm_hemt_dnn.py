#!/usr/bin/env python3


import os
import json
import math
import time
import argparse
from dataclasses import dataclass, asdict
from typing import List, Tuple, Dict, Optional

import numpy as np
import h5py

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter

# ----------------------------
# 1) 实用函数 & 固定随机种子
# ----------------------------

def set_seed(seed: int = 42):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = False  # 保持高性能
    torch.backends.cudnn.benchmark = True


# ----------------------------
# 2) 数据加载 & 预处理
# ----------------------------

PARAM_NAMES = [
    'VOFF', 'U0', 'NS0ACCS', 'NFACTOR', 'ETA0',
    'VSAT', 'VDSCALE', 'CDSCD', 'LAMBDA', 'MEXPACCD', 'DELTA'
]

# 供记录/参考（不会用于剪裁），仅帮助选择对数变换
PARAM_RANGE = {
    'VOFF': (-1.2, 2.6),
    'U0': (0, 2.2),
    'NS0ACCS': (1e15, 1e20),
    'NFACTOR': (0.1, 5),
    'ETA0': (0, 1),
    'VSAT': (5e4, 1e7),
    'VDSCALE': (0.5, 1e6),
    'CDSCD': (1e-5, 0.75),
    'LAMBDA': (0, 0.2),
    'MEXPACCD': (0.05, 12),
    'DELTA': (2, 100)
}


def choose_log_mask(param_range: Dict[str, Tuple[float, float]], names: List[str]) -> np.ndarray:
    """根据参数量纲跨度决定是否进行 log10 变换。
    规则：若下界 > 0 且 upper/lower >= 50（~1.7 个数量级），则对数化。
    """
    mask = []
    for n in names:
        lo, hi = param_range[n]
        if lo > 0 and (hi / lo) >= 50:
            mask.append(True)
        else:
            mask.append(False)
    return np.array(mask, dtype=bool)


class YTransform:
    """对 Y 做（可选）log10 + z-score 标准化，并负责反变换。"""
    def __init__(self, names: List[str], log_mask: np.ndarray):
        assert len(names) == len(log_mask)
        self.names = names
        self.log_mask = torch.tensor(log_mask, dtype=torch.bool)
        self.mean: Optional[torch.Tensor] = None
        self.std: Optional[torch.Tensor] = None

    def fit(self, y: torch.Tensor):
        # y: (N, D)
        y_t = y.clone()
        mask = self.log_mask.to(y_t.device)
        if mask.any():
            y_t[:, mask] = torch.log10(y_t[:, mask].clamp_min(1e-12))
        # 存 CPU，使用时再 .to(输入.device)
        self.mean = y_t.mean(dim=0).detach().cpu().to(torch.float32)
        self.std = y_t.std(dim=0).clamp_min(1e-8).detach().cpu().to(torch.float32)

    def transform(self, y: torch.Tensor) -> torch.Tensor:
        y_t = y
        mask = self.log_mask.to(y.device)
        if mask.any():
            y_t = y_t.clone()
            y_t[:, mask] = torch.log10(y_t[:, mask].clamp_min(1e-12))
        return (y_t - self.mean.to(y.device)) / self.std.to(y.device)

    def inverse(self, y_norm: torch.Tensor) -> torch.Tensor:
        mean = self.mean.to(y_norm.device)
        std = self.std.to(y_norm.device)
        y_t = y_norm * std + mean
        y = y_t
        mask = self.log_mask.to(y_norm.device)
        if mask.any():
            y = y.clone()
            y[:, mask] = torch.pow(10.0, y[:, mask])
        return y

    def state_dict(self) -> Dict:
    
        return {
            'names': self.names,
            'log_mask': self.log_mask.cpu().numpy().tolist(),
            'mean': self.mean.cpu().numpy().tolist(),
            'std': self.std.cpu().numpy().tolist()
        }

    @staticmethod
    def from_state_dict(state: Dict) -> 'YTransform':
        obj = YTransform(state['names'], np.array(state['log_mask'], dtype=bool))
        obj.mean = torch.tensor(state['mean'], dtype=torch.float32)
        obj.std = torch.tensor(state['std'], dtype=torch.float32).clamp_min(1e-8)
        return obj


class XStandardizer:
    """对展开后的 X (N, F) 做标准化：x' = (x - mean)/std（使用 train 统计量）。"""
    def __init__(self):
        self.mean: Optional[np.ndarray] = None
        self.std: Optional[np.ndarray] = None

    def fit(self, x_train: np.ndarray):
        self.mean = x_train.mean(axis=0)
        std = x_train.std(axis=0)
        std[std < 1e-12] = 1e-12
        self.std = std

    def transform(self, x: np.ndarray) -> np.ndarray:
        return (x - self.mean) / self.std

    def state_dict(self) -> Dict:
        return {
            'mean': self.mean.tolist(),
            'std': self.std.tolist()
        }

    @staticmethod
    def from_state_dict(state: Dict) -> 'XStandardizer':
        obj = XStandardizer()
        obj.mean = np.array(state['mean'], dtype=np.float32)
        obj.std = np.array(state['std'], dtype=np.float32)
        obj.std[obj.std < 1e-12] = 1e-12
        return obj


class ArrayDataset(Dataset):
    def __init__(self, x: np.ndarray, y: np.ndarray, augment_std: float = 0.0, augment_prob: float = 0.0):
        assert x.shape[0] == y.shape[0]
        self.x = x.astype(np.float32)
        self.y = y.astype(np.float32)
        self.augment_std = float(augment_std)
        self.augment_prob = float(augment_prob)

    def __len__(self):
        return self.x.shape[0]

    def __getitem__(self, idx):
        xi = self.x[idx].copy()
        yi = self.y[idx].copy()
        # 在标准化后的空间做轻量噪声增强（仅训练集会设置 std>0）
        if self.augment_std > 0.0:
            import random
            if random.random() < self.augment_prob:
                xi += np.random.randn(*xi.shape).astype(np.float32) * self.augment_std
        return (
            torch.from_numpy(xi),  # (F,)
            torch.from_numpy(yi)   # (D,)
        )


# ----------------------------
# 3) 模型定义（仅 DNN）
# ----------------------------

class MLP(nn.Module):
    def __init__(self, input_dim: int, hidden: List[int], output_dim: int, dropout: float = 0.1):
        super().__init__()
        layers: List[nn.Module] = []
        prev = input_dim
        for h in hidden:
            layers += [
                nn.Linear(prev, h),
                nn.BatchNorm1d(h),
                nn.GELU(),
                nn.Dropout(dropout)
            ]
            prev = h
        layers += [nn.Linear(prev, output_dim)]
        self.net = nn.Sequential(*layers)
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_uniform_(m.weight, a=math.sqrt(5))
                if m.bias is not None:
                    fan_in, _ = nn.init._calculate_fan_in_and_fan_out(m.weight)
                    bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
                    nn.init.uniform_(m.bias, -bound, bound)

    def forward(self, x):  # x: (B, F)
        return self.net(x)


class MultiHeadMLP(nn.Module):
    """共享干路 + 逐参数小头 + 可选不确定性加权（learnable log_sigma）。"""
    def __init__(self, input_dim: int, hidden: List[int], out_names: List[str], dropout: float = 0.1, use_uncertainty: bool = True):
        super().__init__()
        self.out_names = out_names
        layers: List[nn.Module] = []
        prev = input_dim
        for h in hidden:
            layers += [
                nn.Linear(prev, h),
                nn.BatchNorm1d(h),
                nn.GELU(),
                nn.Dropout(dropout)
            ]
            prev = h
        self.trunk = nn.Sequential(*layers)
        self.heads = nn.ModuleDict({name: nn.Linear(prev, 1) for name in out_names})
        # 任务不确定性（log_sigma），初始化为 0 -> sigma=1
        self.use_uncertainty = use_uncertainty
        if use_uncertainty:
            self.log_sigma = nn.Parameter(torch.zeros(len(out_names)))
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_uniform_(m.weight, a=math.sqrt(5))
                if m.bias is not None:
                    fan_in, _ = nn.init._calculate_fan_in_and_fan_out(m.weight)
                    bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
                    nn.init.uniform_(m.bias, -bound, bound)

    def forward(self, x):  # -> (B, D)
        feat = self.trunk(x)
        outs = [self.heads[n](feat) for n in self.out_names]
        return torch.cat(outs, dim=1)  # (B, D)


# ----------------------------
# 4) 训练/评估流程
# ----------------------------

@dataclass
class TrainConfig:
    data: str
    outdir: str = 'runs'
    seed: int = 42
    test_split: float = 0.15
    val_split: float = 0.15
    max_epochs: int = 200
    batch_size: int = 256
    lr: float = 1e-3
    weight_decay: float = 1e-4
    dropout: float = 0.1
    hidden: Tuple[int, ...] = (960, 512, 256, 128)
    trials: int = 0  # 随机搜索轮数；0 表示不做搜索
    patience: int = 20
    num_workers: int = 4
    compile: bool = False  # 需要 PyTorch 2.0+
    use_onecycle: bool = True
    # 轻量数据增强
    aug_noise_std: float = 0.015
    aug_prob: float = 0.5
    # 新增：多头与不确定性、输出约束
    multihead: bool = True
    uncertainty_weighting: bool = True
    enforce_bounds: bool = True


def split_indices(n: int, test_ratio: float, val_ratio: float, seed: int = 42):
    rng = np.random.default_rng(seed)
    idx = np.arange(n)
    rng.shuffle(idx)
    n_test = int(n * test_ratio)
    n_val = int(n * val_ratio)
    test_idx = idx[:n_test]
    val_idx = idx[n_test:n_test + n_val]
    train_idx = idx[n_test + n_val:]
    return train_idx, val_idx, test_idx


def load_and_prepare(data_path: str, cfg: TrainConfig):
    assert os.path.isfile(data_path), f"Data file not found: {data_path}"
    with h5py.File(data_path, 'r') as f:
        X = f['X'][...]  # (N, 7, 121)
        Y = f['Y'][...]  # (N, 11, 1)
    N = X.shape[0]

    # 展开输入到 1D
    X = X.reshape(N, -1).astype(np.float32)
    Y = Y.reshape(N, len(PARAM_NAMES)).astype(np.float32)

    # 划分索引
    tr_idx, va_idx, te_idx = split_indices(N, cfg.test_split, cfg.val_split, cfg.seed)

    # 计算 X 标准化（仅用训练集统计量）
    x_scaler = XStandardizer()
    x_scaler.fit(X[tr_idx])

    X_tr = x_scaler.transform(X[tr_idx])
    X_va = x_scaler.transform(X[va_idx])
    X_te = x_scaler.transform(X[te_idx])

    # 计算 Y 预处理（log + z-score），仅用训练集
    log_mask_np = choose_log_mask(PARAM_RANGE, PARAM_NAMES)
    y_tf = YTransform(PARAM_NAMES, log_mask_np)
    y_tf.fit(torch.from_numpy(Y[tr_idx]))

    Y_tr = y_tf.transform(torch.from_numpy(Y[tr_idx])).numpy()
    Y_va = y_tf.transform(torch.from_numpy(Y[va_idx])).numpy()
    Y_te = y_tf.transform(torch.from_numpy(Y[te_idx])).numpy()

    train_ds = ArrayDataset(X_tr, Y_tr, augment_std=cfg.aug_noise_std, augment_prob=cfg.aug_prob)
    val_ds   = ArrayDataset(X_va, Y_va)
    test_ds  = ArrayDataset(X_te, Y_te)

    return train_ds, val_ds, test_ds, x_scaler, y_tf, (tr_idx, va_idx, te_idx)


def make_loaders(train_ds, val_ds, test_ds, batch_size: int, num_workers: int):
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True, drop_last=False)
    val_loader   = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True, drop_last=False)
    test_loader  = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True, drop_last=False)
    return train_loader, val_loader, test_loader


def train_one_epoch(model, loader, optimizer, scaler, device, criterion, scheduler=None):
    model.train()
    total = 0.0
    n = 0
    printed_once = False
    for x, y in loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        if device.type == 'cuda' and not printed_once:
            assert x.is_cuda and y.is_cuda and next(model.parameters()).is_cuda, 'Tensors or model are not on CUDA as expected.'
            torch.cuda.synchronize()
            print(f"[Debug] Devices -> x:{x.device}, y:{y.device}, model:{next(model.parameters()).device}; mem_alloc={torch.cuda.memory_allocated()/1e6:.1f}MB")
            printed_once = True
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type='cuda' if device.type=='cuda' else 'cpu', dtype=torch.float16 if device.type=='cuda' else torch.bfloat16):
            pred = model(x)
            loss = criterion(pred, y)
        scaler.scale(loss).backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=2.0)
        scaler.step(optimizer)
        scaler.update()
        if scheduler is not None:
            scheduler.step()
        total += loss.item() * x.size(0)
        n += x.size(0)
    return total / max(1, n)


def evaluate(model, loader, device, criterion) -> Tuple[float, torch.Tensor, torch.Tensor]:
    model.eval()
    total = 0.0
    n = 0
    preds = []
    gts = []
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            pred = model(x)
            loss = criterion(pred, y)
            total += loss.item() * x.size(0)
            n += x.size(0)
            preds.append(pred.cpu())
            gts.append(y.cpu())
    preds = torch.cat(preds, dim=0)
    gts = torch.cat(gts, dim=0)
    return total / max(1, n), preds, gts


def mae_per_param(pred_norm: torch.Tensor, gt_norm: torch.Tensor, y_tf: YTransform, enforce_bounds: bool = False) -> Dict[str, float]:
    pred_raw = y_tf.inverse(pred_norm)
    gt_raw = y_tf.inverse(gt_norm)
    if enforce_bounds:
        # 物理边界裁剪（仅用于指标/推理，不反向传播）
        for i, name in enumerate(y_tf.names):
            lo, hi = PARAM_RANGE.get(name, (-float('inf'), float('inf')))
            pred_raw[:, i] = pred_raw[:, i].clamp(min=lo if not math.isinf(lo) else None,
                                                  max=hi if not math.isinf(hi) else None)
    mae = torch.mean(torch.abs(pred_raw - gt_raw), dim=0).numpy()
    return {name: float(mae[i]) for i, name in enumerate(y_tf.names)}


def save_state(outdir: str, x_scaler: XStandardizer, y_tf: YTransform, cfg: TrainConfig, best_path: str):
    os.makedirs(outdir, exist_ok=True)
    meta = {
        'x_scaler': x_scaler.state_dict(),
        'y_transform': y_tf.state_dict(),
        'config': asdict(cfg),
        'param_names': PARAM_NAMES,
                'input_dim': len(x_scaler.mean),
    }
    with open(os.path.join(outdir, 'transforms.json'), 'w') as f:
        json.dump(meta, f, indent=2)
    # best model 已由主流程保存


# ----------------------------
# 5) 随机搜索（简单高效）
# ----------------------------

def sample_hparams() -> Dict:
    # 简单随机空间（可按需扩展）
    hidden_choices = [
        (1024, 512, 256, 128),
        (960, 512, 256, 128),
        (896, 448, 256, 128),
        (768, 384, 192, 128),
        (1024, 512, 256)
    ]
    bs_choices = [128, 256, 512]
    lr = 10 ** np.random.uniform(-3.7, -2.5)  # ~2e-4 ~ 3e-3
    wd = 10 ** np.random.uniform(-6, -3)      # 1e-6 ~ 1e-3
    dropout = float(np.random.uniform(0.0, 0.3))
    hidden = hidden_choices[np.random.randint(0, len(hidden_choices))]
    bs = bs_choices[np.random.randint(0, len(bs_choices))]
    return {'lr': lr, 'weight_decay': wd, 'dropout': dropout, 'hidden': hidden, 'batch_size': bs}


# ----------------------------
# 6) 主训练入口
# ----------------------------

def run_once(cfg: TrainConfig, device):
    train_ds, val_ds, test_ds, x_scaler, y_tf, splits = load_and_prepare(cfg.data, cfg)
    train_loader, val_loader, test_loader = make_loaders(train_ds, val_ds, test_ds, cfg.batch_size, cfg.num_workers)

    input_dim = train_ds.x.shape[1]
    output_dim = train_ds.y.shape[1]

    if cfg.multihead:
        model = MultiHeadMLP(input_dim, list(cfg.hidden), PARAM_NAMES, dropout=cfg.dropout, use_uncertainty=cfg.uncertainty_weighting).to(device)
    else:
        model = MLP(input_dim, list(cfg.hidden), output_dim, dropout=cfg.dropout).to(device)
    if cfg.compile and hasattr(torch, 'compile'):
        try:
            model = torch.compile(model)  # type: ignore
        except Exception:
            pass

    base_criterion = nn.SmoothL1Loss(beta=0.02, reduction='none')  # per-dim loss

    def criterion(pred, target):
        # pred, target: (B, D)
        per_elem = base_criterion(pred, target)  # (B, D)
        per_dim = per_elem.mean(dim=0)           # (D,)
        if cfg.multihead and cfg.uncertainty_weighting and hasattr(model, 'log_sigma'):
            # Kendall & Gal (2018): sum(exp(-2s_i)*L_i + s_i)
            s = model.log_sigma
            loss = torch.sum(torch.exp(-2*s) * per_dim + s)
            return loss
        else:
            return per_dim.mean()

    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)

    steps_per_epoch = max(1, len(train_loader))
    scheduler = None
    if cfg.use_onecycle:
        scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=cfg.lr, epochs=cfg.max_epochs, steps_per_epoch=steps_per_epoch)

    scaler = torch.amp.GradScaler('cuda' if device.type=='cuda' else 'cpu', enabled=(device.type=='cuda'))

    # 日志目录
    stamp = time.strftime('%Y%m%d-%H%M%S')
    run_dir = os.path.join(cfg.outdir, f'asm_hemt_dnn_{stamp}')
    os.makedirs(run_dir, exist_ok=True)
    writer = SummaryWriter(log_dir=run_dir)

    best_val = float('inf')
    best_path = os.path.join(run_dir, 'best_model.pt')
    patience = cfg.patience
    no_improve = 0

    for epoch in range(1, cfg.max_epochs + 1):
        train_loss = train_one_epoch(model, train_loader, optimizer, scaler, device, criterion, scheduler)
        val_loss, val_pred, val_gt = evaluate(model, val_loader, device, criterion)

        # 记录到 TensorBoard
        writer.add_scalar('loss/train', train_loss, epoch)
        writer.add_scalar('loss/val', val_loss, epoch)

        # 逐参数 MAE（回到物理量空间）
        val_mae = mae_per_param(val_pred, val_gt, y_tf, enforce_bounds=cfg.enforce_bounds)
        for k, v in val_mae.items():
            writer.add_scalar(f'mae_val/{k}', v, epoch)

        print(f"Epoch {epoch:03d} | train {train_loss:.6f} | val {val_loss:.6f} | "
              f"MAE(val): " + ", ".join([f"{k}:{v:.3g}" for k, v in val_mae.items()]))

        # 早停与保存
        if val_loss < best_val - 1e-6:
            best_val = val_loss
            no_improve = 0
            torch.save({'model': model.state_dict(), 'config': asdict(cfg)}, best_path)
        else:
            no_improve += 1
            if no_improve >= patience:
                print(f"Early stopping at epoch {epoch} (no improve {patience})")
                break

    # 加载最好模型，测试集评估
    ckpt = torch.load(best_path, map_location=device)
    model.load_state_dict(ckpt['model'])
    test_loss, test_pred, test_gt = evaluate(model, test_loader, device, criterion)
    test_mae = mae_per_param(test_pred, test_gt, y_tf, enforce_bounds=cfg.enforce_bounds)
    print("\n==== Test Metrics ====")
    print(f"Test loss (SmoothL1 on normalized targets): {test_loss:.6f}")
    writer.add_scalar('loss/test', test_loss, cfg.max_epochs)
    for k, v in test_mae.items():
        print(f"MAE[{k}] = {v:.6g}")
        writer.add_scalar(f'mae_test/{k}', v, cfg.max_epochs)
    # 也将最终指标汇总进 HParams 面板
    metric_dict = {'final/test_loss': test_loss}
    for k, v in test_mae.items():
        metric_dict[f'final_mae/{k}'] = v
    writer.add_hparams({"tag": "final"}, metric_dict)

    # 保存变换与配置
    save_state(run_dir, x_scaler, y_tf, cfg, best_path)

    writer.flush()
    writer.close()

    return {
        'run_dir': run_dir,
        'best_model': best_path,
        'test_loss': test_loss,
        'test_mae': test_mae,
        'best_val': best_val,
    }


def random_search(cfg: TrainConfig, device):
    """顺序随机搜索若干 trial，按 best_val 选最优配置并打印结果。"""
    best_score = float('inf')
    best_cfg = None
    best_res = None

    for t in range(1, cfg.trials + 1):
        hp = sample_hparams()
        trial_cfg = TrainConfig(
            data=cfg.data,
            outdir=cfg.outdir,
            seed=cfg.seed + t,  # 变化 seed，避免划分/初始化完全相同
            test_split=cfg.test_split,
            val_split=cfg.val_split,
            max_epochs=min(cfg.max_epochs, 120),
            batch_size=hp['batch_size'],
            lr=hp['lr'],
            weight_decay=hp['weight_decay'],
            dropout=hp['dropout'],
            hidden=tuple(hp['hidden']),
            trials=0,
            patience=max(10, cfg.patience // 2),
            num_workers=cfg.num_workers,
            compile=cfg.compile,
            use_onecycle=cfg.use_onecycle,
            aug_noise_std=cfg.aug_noise_std,
            aug_prob=cfg.aug_prob,
            multihead=cfg.multihead,
            uncertainty_weighting=cfg.uncertainty_weighting,
            enforce_bounds=cfg.enforce_bounds,
        )
        print(f"[Trial {t}/{cfg.trials}] hp: hidden={trial_cfg.hidden}, bs={trial_cfg.batch_size}, lr={trial_cfg.lr:.3g}, wd={trial_cfg.weight_decay:.1e}, drop={trial_cfg.dropout:.2f}")
        res = run_once(trial_cfg, device)
        score = res.get('best_val', res['test_loss'])
        print(f"[Trial {t}] best_val={score:.6f}  test_loss={res['test_loss']:.6f}  run_dir={res['run_dir']}")
        if score < best_score:
            best_score = score
            best_cfg = trial_cfg
            best_res = res

    if best_cfg is not None:
        print("=== Random Search Done ===")
        print(f"Best config: hidden={best_cfg.hidden}, bs={best_cfg.batch_size}, lr={best_cfg.lr:.3g}, wd={best_cfg.weight_decay:.1e}, drop={best_cfg.dropout:.2f}")
        print(f"Best run: {best_res['run_dir']}  | best_val={best_score:.6f}  test_loss={best_res['test_loss']:.6f}")
    else:
        print("Random search received trials=0; nothing to do.")

    return best_cfg, best_res


# ----------------------------
# 7) 推理与 CLI 参数
# ----------------------------

def _flatten_X_any(X: np.ndarray) -> np.ndarray:
    """把输入展平为 (N, F)。支持：
      - (N,7,121)  -> (N,847)
      - (7,121)    -> (1,847)
      - (N,F) / (F,)  兼容
      - 其它维度（如 4D）也会被展平为 (N,-1)
    """
    X = np.asarray(X)
    if X.ndim == 3:
        # 认为是 (N,7,121)
        X = X.reshape(X.shape[0], -1)
    elif X.ndim == 2:
        # 认为是 (7,121) 或 (N,F)
        if X.shape[0] == 7 and X.shape[1] == 121:
            X = X.reshape(1, -1)
        else:
            # 已经是 (N,F)
            pass
    elif X.ndim == 1:
        X = X.reshape(1, -1)
    elif X.ndim >= 4:
        X = X.reshape(X.shape[0], -1)
    else:
        raise ValueError(f"unsupported X shape: {X.shape}")
    return X.astype(np.float32)


def load_artifacts(run_dir: str, device):
    """加载 best_model + transforms，用于推理。返回 (model, x_scaler, y_tf, param_names, meta)。"""
    tr_path = os.path.join(run_dir, 'transforms.json')
    md_path = os.path.join(run_dir, 'best_model.pt')
    assert os.path.isfile(tr_path), f"transforms.json not found in {run_dir}"
    assert os.path.isfile(md_path), f"best_model.pt not found in {run_dir}"

    with open(tr_path, 'r') as f:
        meta = json.load(f)
    x_scaler = XStandardizer.from_state_dict(meta['x_scaler'])
    y_tf = YTransform.from_state_dict(meta['y_transform'])
    param_names = meta.get('param_names', PARAM_NAMES)
    cfg = meta.get('config', {})

    input_dim = int(meta.get('input_dim', len(x_scaler.mean)))
    hidden = tuple(cfg.get('hidden', [1024, 512, 256, 128]))
    output_dim = len(param_names)
    use_mh = cfg.get('multihead', True)
    use_unc = cfg.get('uncertainty_weighting', True)

    if use_mh:
        model = MultiHeadMLP(input_dim, list(hidden), param_names, dropout=cfg.get('dropout', 0.1), use_uncertainty=use_unc).to(device)
    else:
        model = MLP(input_dim, list(hidden), output_dim, dropout=cfg.get('dropout', 0.1)).to(device)
    ckpt = torch.load(md_path, map_location=device)
    model.load_state_dict(ckpt['model'])
    model.eval()
    return model, x_scaler, y_tf, param_names, meta


def infer_cli(args, device):
    model, x_scaler, y_tf, names, meta = load_artifacts(args.infer_run, device)
    expected_F = int(meta.get('input_dim', len(x_scaler.mean)))
    
    # 读取输入
    if args.input_npy:
        X = np.load(args.input_npy)
    elif args.input_h5:
        with h5py.File(args.input_h5, 'r') as f:
            X = f['X'][...]
        if args.index is not None and X.ndim == 3:
            idx = int(np.clip(args.index, 0, X.shape[0]-1))
            X = X[idx]
    else:
        raise ValueError('Please provide --input-npy or --input-h5')

    
    x = _flatten_X_any(X)
    if x.shape[1] != expected_F:
        raise ValueError(f"Flattened feature dim {x.shape[1]} != expected {expected_F}. Check channels/grid.")

    x_std = x_scaler.transform(x)
    with torch.no_grad():
        xt = torch.from_numpy(x_std).to(device)
        pred_norm = model(xt)
        pred_raw = y_tf.inverse(pred_norm).cpu()
        if meta.get('config', {}).get('enforce_bounds', True):
            for i, n in enumerate(names):
                lo, hi = PARAM_RANGE.get(n, (-float('inf'), float('inf')))
                pred_raw[:, i] = pred_raw[:, i].clamp(min=lo if not math.isinf(lo) else None,
                                                       max=hi if not math.isinf(hi) else None)
        pred = pred_raw.numpy()

    # 输出
    N = pred.shape[0]
    names = list(names)
    if N == 1:
        print("Prediction (physical units):")
        w = max(len(n) for n in names)
        for i, n in enumerate(names):
            print(f"  {n:<{w}} : {pred[0, i]:.6g}")
    else:
        show = min(3, N)
        print(f"Predicted {N} samples. Showing first {show} (physical units):")
        w = max(len(n) for n in names)
        for s in range(show):
            print(f"[Sample {s}]")
            for i, n in enumerate(names):
                print(f"  {n:<{w}} : {pred[s, i]:.6g}")
        if args.save_csv is None:
            print("(Tip) Use --save-csv to save all predictions to a CSV file.")

    if args.save_csv:
        import csv
        with open(args.save_csv, 'w', newline='') as f:
            w = csv.writer(f)
            if N == 1:
                w.writerow(['param', 'value'])
                for i, n in enumerate(names):
                    w.writerow([n, pred[0, i]])
            else:
                w.writerow(['index'] + names)
                for idx in range(N):
                    w.writerow([idx] + [pred[idx, i] for i in range(len(names))])
        print(f"Saved CSV -> {args.save_csv}")


def parse_args():
    parser = argparse.ArgumentParser(description='ASM-HEMT DNN Regressor')
    # 训练/通用
    parser.add_argument('--data', type=str, default=None, help='Path to .h5 dataset with X,Y')
    parser.add_argument('--outdir', type=str, default='runs')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--test-split', type=float, default=0.15)
    parser.add_argument('--val-split', type=float, default=0.15)
    parser.add_argument('--max-epochs', type=int, default=200)
    parser.add_argument('--batch-size', type=int, default=256)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--weight-decay', type=float, default=1e-4)
    parser.add_argument('--dropout', type=float, default=0.1)
    parser.add_argument('--hidden', type=str, default='960,512,256,128', help='Comma-separated hidden sizes')
    parser.add_argument('--trials', type=int, default=0)
    parser.add_argument('--patience', type=int, default=20)
    parser.add_argument('--num-workers', type=int, default=4)
    parser.add_argument('--compile', action='store_true')
    parser.add_argument('--no-onecycle', action='store_true')
    # 新增特性开关
    parser.add_argument('--aug-noise-std', type=float, default=0.015)
    parser.add_argument('--aug-prob', type=float, default=0.5)
    parser.add_argument('--no-multihead', action='store_true', help='Disable multi-head; default is ON')
    parser.add_argument('--no-uncertainty', action='store_true', help='Disable uncertainty weighting')
    parser.add_argument('--no-bounds', action='store_true', help='Disable output range enforcement in metrics/inference')
    # 推理
    parser.add_argument('--infer-run', type=str, default=None, help='Run dir containing best_model.pt & transforms.json')
    parser.add_argument('--input-npy', type=str, default=None)
    parser.add_argument('--input-h5', type=str, default=None)
    parser.add_argument('--index', type=int, default=None, help='Index for --input-h5; if None and X is 4D, run full batch')
    parser.add_argument('--save-csv', type=str, default=None)

    args = parser.parse_args()
    hidden = tuple(int(x) for x in args.hidden.split(','))

    cfg = TrainConfig(
        data=args.data,
        outdir=args.outdir,
        seed=args.seed,
        test_split=args.test_split,
        val_split=args.val_split,
        max_epochs=args.max_epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        weight_decay=args.weight_decay,
        dropout=args.dropout,
        hidden=hidden,
        trials=args.trials,
        patience=args.patience,
        num_workers=args.num_workers,
        compile=args.compile,
        use_onecycle=not args.no_onecycle,
                aug_noise_std=args.aug_noise_std,
        aug_prob=args.aug_prob,
        multihead=(not args.no_multihead),
        uncertainty_weighting=(not args.no_uncertainty),
        enforce_bounds=(not args.no_bounds),
    )
    return cfg, args

def main():
    cfg, args = parse_args()
    set_seed(cfg.seed)

    # 仅使用第一个 GPU
    if torch.cuda.is_available():
        try:
            torch.cuda.set_device(0)
        except Exception:
            pass
        device = torch.device('cuda:0')
        name = torch.cuda.get_device_name(0)
        cap = torch.cuda.get_device_capability(0)
        print(f'Using CUDA device 0: {name} (cc {cap[0]}.{cap[1]})')
    else:
        device = torch.device('cpu')
        print('CUDA not available, using CPU')

    # 推理模式优先
    if getattr(args, 'infer_run', None):
        infer_cli(args, device)
        return

    # 训练模式
    if cfg.data is None:
        raise SystemExit('--data is required for training')
    if cfg.trials and cfg.trials > 0:
        random_search(cfg, device)
    else:
        run_once(cfg, device)


if __name__ == '__main__':
    main()
