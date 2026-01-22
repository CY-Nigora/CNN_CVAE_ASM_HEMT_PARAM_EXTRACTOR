#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ASM-HEMT two-stage DNN training with optional proxy g(Y->X) consistency.
• YTransform uses dataset-driven z-score (per train split).
• Proxy scaling is INDEPENDENT from the main model when auto-training proxy.
• When auto-training proxy, switch to TorchScript immediately for parity with --proxy-run.
• Removes unused features (random search, extra inference CLIs, etc.) and tidies code.
"""
import os
import csv
import json
import math
import time
import argparse
from dataclasses import dataclass, asdict
from typing import List, Tuple, Dict, Optional

import numpy as np
import h5py

import hashlib
from contextlib import contextmanager
import random

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter
import torch.nn.functional as F

# ============================
# 0) for safe running
# ============================

def add_hparams_safe(base_writer: SummaryWriter, run_dir: str, hparams: dict, metrics: dict):
    try:
        # 大多数环境下这一步就能成功
        base_writer.add_hparams(hparams, metrics)
        return
    except Exception as e:
        print(f"[warn] add_hparams failed on base writer: {e}. Falling back to dedicated hparams dir.")

    hp_dir = os.path.join(run_dir, "hparams")
    os.makedirs(hp_dir, exist_ok=True)
    try:
        with SummaryWriter(log_dir=hp_dir) as w_hp:
            # 这里的 SummaryWriter 会确保 hp_dir 存在，不再依赖内部再创建
            w_hp.add_hparams(hparams, metrics)
        return
    except Exception as e2:
        print(f"[warn] add_hparams failed again: {e2}. Dumping hparams to JSON.")
        try:
            with open(os.path.join(hp_dir, "hparams.json"), "w", encoding="utf-8") as f:
                json.dump({"hparams": hparams, "metrics": metrics}, f, indent=2)
        except Exception as e3:
            print(f"[warn] writing hparams.json failed: {e3}")

# ============================
# 1) Utils & RNG isolation
# ============================


def set_seed(seed: int = 42):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True


def _seed_from_proxy_cfg(hidden, activation, norm, max_epochs, lr, weight_decay, beta, extra=None) -> int:
    payload = {
        "hidden": tuple(hidden),
        "activation": str(activation),
        "norm": str(norm),
        "max_epochs": int(max_epochs),
        "lr": float(lr),
        "weight_decay": float(weight_decay),
        "beta": float(beta),
        "extra": extra,
    }
    s = json.dumps(payload, sort_keys=True)
    h = hashlib.sha256(s.encode()).hexdigest()[:16]
    cur_seed = int(h, 16) % (2**32 - 1)
    print('current generated proxy seed: ',cur_seed)

    return cur_seed


@contextmanager
def scoped_rng(seed: int):
    """Local RNG scope without polluting global RNG states."""
    np_state = np.random.get_state()
    py_state = random.getstate()
    torch_state = torch.random.get_rng_state()
    cuda_states = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
    try:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        yield
    finally:
        np.random.set_state(np_state)
        random.setstate(py_state)
        torch.random.set_rng_state(torch_state)
        if cuda_states is not None:
            torch.cuda.set_rng_state_all(cuda_states)


# ============================
# 2) Data & transforms
# ============================
PARAM_NAMES = [
    'VOFF', 'U0', 'NS0ACCS', 'NFACTOR', 'ETA0',
    'VSAT', 'VDSCALE', 'CDSCD', 'LAMBDA', 'MEXPACCD', 'DELTA'
]

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
    mask = []
    for n in names:
        lo, hi = param_range[n]
        mask.append(bool(lo > 0 and (hi / max(lo, 1e-30)) >= 50))
    return np.array(mask, dtype=bool)

# --- util: zero-based softplus so margin==0 gives zero penalty at boundary ---
def softplus0(t: torch.Tensor, beta: float = 2.0) -> torch.Tensor:
    # PyTorch: softplus(t,beta) = 1/beta * log(1 + exp(beta*t))
    # softplus(0,beta) = log(2)/beta  -> subtract it to make zero at t=0
    return torch.clamp_min(F.softplus(t, beta=beta) - (math.log(2.0)/beta), 0.0)


class YTransform:
    """log10 (selected dims) + z-score using TRAIN split statistics."""
    def __init__(self, names: List[str], log_mask: np.ndarray):
        assert len(names) == len(log_mask)
        self.names = list(names)
        self.log_mask = torch.tensor(log_mask, dtype=torch.bool)
        self.mean: Optional[torch.Tensor] = None
        self.std: Optional[torch.Tensor] = None

    def fit(self, y: torch.Tensor):
        y_t = y.clone()
        mask = self.log_mask.to(y_t.device)
        if mask.any():
            y_t[:, mask] = torch.log10(y_t[:, mask].clamp_min(1e-12))
        self.mean = y_t.mean(dim=0).detach().cpu().to(torch.float32)
        self.std  = y_t.std(dim=0).clamp_min(1e-8).detach().cpu().to(torch.float32)

    def transform(self, y: torch.Tensor) -> torch.Tensor:
        y_t = y
        mask = self.log_mask.to(y.device)
        if mask.any():
            y_t = y_t.clone()
            y_t[:, mask] = torch.log10(y_t[:, mask].clamp_min(1e-12))
        return (y_t - self.mean.to(y.device)) / self.std.to(y.device)

    def inverse(self, y_norm: torch.Tensor) -> torch.Tensor:
        # 在 FP32 下做反变换，避免半精度 pow/exp 溢出
        y_norm = y_norm.to(torch.float32)
        mean = self.mean.to(y_norm.device, dtype=torch.float32)
        std  = self.std.to(y_norm.device,  dtype=torch.float32)

        y_t = y_norm * std + mean   # 回到 log10 或线性空间
        mask = self.log_mask.to(y_norm.device)

        if mask.any():
            # 在 log10 空间裁到 [-38, 38]，确保 10**x ≤ 1e38
            y_t[:, mask] = y_t[:, mask].clamp_(min=-38.0, max=21.0)
            y_t[:, mask] = torch.pow(10.0, y_t[:, mask])

        return y_t

    def state_dict(self) -> Dict:
        return {
            'names': self.names,
            'log_mask': self.log_mask.cpu().numpy().tolist(),
            'mean': self.mean.cpu().numpy().tolist(),
            'std': self.std.cpu().numpy().tolist(),
            'norm_type': 'zscore'
        }

    @staticmethod
    def from_state_dict(state: Dict) -> 'YTransform':
        obj = YTransform(state['names'], np.array(state['log_mask'], dtype=bool))
        obj.mean = torch.tensor(state['mean'], dtype=torch.float32)
        obj.std  = torch.tensor(state['std'],  dtype=torch.float32).clamp_min(1e-8)
        return obj


class XStandardizer:
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
    def inverse(self, x_std: np.ndarray) -> np.ndarray:
        return x_std * self.std + self.mean
    def state_dict(self) -> Dict:
        return {'mean': self.mean.tolist(), 'std': self.std.tolist()}
    @staticmethod
    def from_state_dict(state: Dict) -> 'XStandardizer':
        obj = XStandardizer()
        obj.mean = np.array(state['mean'], dtype=np.float32)
        obj.std  = np.array(state['std'],  dtype=np.float32)
        obj.std[obj.std < 1e-12] = 1e-12
        return obj


class ArrayDataset(Dataset):
    def __init__(self, x: np.ndarray, y: np.ndarray, augment_std: float = 0.0, augment_prob: float = 0.0, weights: Optional[np.ndarray] = None):
        assert x.shape[0] == y.shape[0]
        self.x = x.astype(np.float32)
        self.y = y.astype(np.float32)
        self.w = None if weights is None else weights.astype(np.float32).reshape(-1)
        self.augment_std = float(augment_std)
        self.augment_prob = float(augment_prob)
    def __len__(self):
        return self.x.shape[0]
    def __getitem__(self, idx):
        xi = self.x[idx].copy()
        yi = self.y[idx].copy()
        if self.augment_std > 0.0:
            import random
            if random.random() < self.augment_prob:
                xi += np.random.randn(*xi.shape).astype(np.float32) * self.augment_std
        if self.w is None:
            return torch.from_numpy(xi), torch.from_numpy(yi)
        else:
            return torch.from_numpy(xi), torch.from_numpy(yi), torch.tensor(self.w[idx])


class MeasDataset(Dataset):
    def __init__(self, x_std: np.ndarray):
        self.x = x_std.astype(np.float32)
    def __len__(self):
        return self.x.shape[0]
    def __getitem__(self, idx):
        return torch.from_numpy(self.x[idx])


# ============================
# 3) Models
# ============================
class MLP(nn.Module):
    def __init__(self, input_dim: int, hidden: List[int], output_dim: int, dropout: float = 0.1):
        super().__init__()
        layers: List[nn.Module] = []
        prev = input_dim
        for h in hidden:
            layers += [nn.Linear(prev, h), nn.BatchNorm1d(h), nn.GELU(), nn.Dropout(dropout)]
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
    def forward(self, x):
        return self.net(x)


class MultiHeadMLP(nn.Module):
    def __init__(self, input_dim: int, hidden: List[int], out_names: List[str], dropout: float = 0.1, use_uncertainty: bool = True):
        super().__init__()
        self.out_names = out_names
        layers: List[nn.Module] = []
        prev = input_dim
        for h in hidden:
            layers += [nn.Linear(prev, h), nn.BatchNorm1d(h), nn.GELU(), nn.Dropout(dropout)]
            prev = h
        self.trunk = nn.Sequential(*layers)
        self.heads = nn.ModuleDict({name: nn.Linear(prev, 1) for name in out_names})
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
    def forward(self, x):
        feat = self.trunk(x)
        outs = [self.heads[n](feat) for n in self.out_names]
        return torch.cat(outs, dim=1)


def _make_act(name: str) -> nn.Module:
    name = (name or 'gelu').lower()
    if name == 'relu':
        return nn.ReLU()
    if name in ('silu', 'swish'):
        return nn.SiLU()
    return nn.GELU()


def _make_norm(name: str, dim: int) -> nn.Module:
    name = (name or 'layernorm').lower()
    if name in ('batchnorm', 'bn'):
        return nn.BatchNorm1d(dim)
    if name in ('none', 'identity'):
        return nn.Identity()
    return nn.LayerNorm(dim)


class ProxyMLP(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, hidden: List[int], activation: str = 'gelu', norm: str = 'layernorm'):
        super().__init__()
        layers: List[nn.Module] = []
        prev = in_dim
        act = _make_act(activation)
        for h in hidden:
            layers += [nn.Linear(prev, h), act, _make_norm(norm, h)]
            prev = h
        layers += [nn.Linear(prev, out_dim)]
        self.net = nn.Sequential(*layers)
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_uniform_(m.weight, a=math.sqrt(5))
                if m.bias is not None:
                    fan_in, _ = nn.init._calculate_fan_in_and_fan_out(m.weight)
                    bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
                    nn.init.uniform_(m.bias, -bound, bound)
    def forward(self, y_norm):
        return self.net(y_norm)


# ============================
# 4) Train / Eval helpers
# ============================
@dataclass
class TrainConfig:
    data: str
    outdir: str = 'runs'
    seed: int = 42
    test_split: float = 0.15
    val_split: float = 0.15
    max_epochs: int = 200
    onecycle_epochs: int = 200
    batch_size: int = 256
    lr: float = 1e-3
    weight_decay: float = 1e-4
    dropout: float = 0.1
    hidden: Tuple[int, ...] = (960, 512, 256, 128)
    patience: int = 20
    num_workers: int = 0
    compile: bool = False
    use_onecycle: bool = True
    aug_noise_std: float = 0.015
    aug_prob: float = 0.5
    multihead: bool = True
    uncertainty_weighting: bool = True
    enforce_bounds: bool = True
    # consistency & proxy
    meas_h5: Optional[str] = None
    lambda_cyc_sim: float = 0.0
    lambda_cyc_meas: float = 0.0
    cyc_warmup_epochs: int = 15
    proxy_run: Optional[str] = None
    auto_train_proxy: bool = True
    # proxy hyperparams
    proxy_hidden: Tuple[int, ...] = (512, 512)
    proxy_activation: str = 'gelu'
    proxy_norm: str = 'layernorm'
    proxy_epochs: int = 100
    proxy_lr: float = 1e-3
    proxy_wd: float = 1e-4
    proxy_beta: float = 0.02
    proxy_seed: Optional[int] = None
    proxy_patience: int = 15
    proxy_min_delta: float = 1e-6
    proxy_batch_size: int = 1024
    # proxy utility
    train_proxy_only: bool = False
    # finetune
    finetune_from: Optional[str] = None
    ft_use_prev_transforms: bool = True
    ft_freeze_trunk: bool = False
    ft_reset_head: bool = False
    ft_allow_arch_change: bool = False
    ft_strict: bool = True
    # weak L_sup & strong L_cyc
    sup_weight: float = 0.05          # The weight of the supervision item L_sup is set between 0 and 0.1, and 0.05 is recommended as a "safety belt"
    prior_l2: float = 1e-3            # Weak prior: L2 shrinkage coefficient for y_norm (to avoid divergence/extreme solutions)
    prior_bound: float = 1e-3         # Weak prior: soft penalty coefficient for physical domain violations (softplus)
    prior_bound_margin: float = 0.0   # The margin of the soft penalty for the border (can be left as 0.0)
    es_metric: str = 'val_cyc_meas'   # option: 'val_total' | 'val_sup' | 'val_cyc_sim' | 'val_cyc_meas'
    es_min_delta: float = 1e-6
    # switch for L_trust
    trust_alpha : float = 0.0
    trust_tau : float = 1.6
    trust_ref_max : int = 20000
    trust_ref_batch : int = 4096
    # switch for L_trust_meas
    trust_alpha_meas : float = 0.05
    cyc_meas_knn_weight : bool = False
    cyc_meas_knn_gamma : float = 0.5



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


def load_and_prepare(data_path: str, cfg: TrainConfig,
                     override_scalers: Optional[Tuple['XStandardizer','YTransform']] = None):
    assert os.path.isfile(data_path), f"Data file not found: {data_path}"
    with h5py.File(data_path, 'r') as f:
        X = f['X'][...]
        Y = f['Y'][...]
    N = X.shape[0]
    X = X.reshape(N, -1).astype(np.float32)
    Y = Y.reshape(N, len(PARAM_NAMES)).astype(np.float32)

    tr_idx, va_idx, te_idx = split_indices(N, cfg.test_split, cfg.val_split, cfg.seed)

    if override_scalers is None:
        x_scaler = XStandardizer(); x_scaler.fit(X[tr_idx])
        X_tr = x_scaler.transform(X[tr_idx]); X_va = x_scaler.transform(X[va_idx]); X_te = x_scaler.transform(X[te_idx])
        log_mask_np = choose_log_mask(PARAM_RANGE, PARAM_NAMES)
        y_tf = YTransform(PARAM_NAMES, log_mask_np)
        y_tf.fit(torch.from_numpy(Y[tr_idx]))
    else:
        x_scaler, y_tf = override_scalers
        X_tr = x_scaler.transform(X[tr_idx]); X_va = x_scaler.transform(X[va_idx]); X_te = x_scaler.transform(X[te_idx])

    Y_tr = y_tf.transform(torch.from_numpy(Y[tr_idx])).numpy()
    Y_va = y_tf.transform(torch.from_numpy(Y[va_idx])).numpy()
    Y_te = y_tf.transform(torch.from_numpy(Y[te_idx])).numpy()

    train_ds = ArrayDataset(X_tr, Y_tr, augment_std=cfg.aug_noise_std, augment_prob=cfg.aug_prob)
    val_ds   = ArrayDataset(X_va, Y_va)
    test_ds  = ArrayDataset(X_te, Y_te)

    return train_ds, val_ds, test_ds, x_scaler, y_tf, (tr_idx, va_idx, te_idx), X, Y


def make_loaders(train_ds, val_ds, test_ds, batch_size: int, num_workers: int):
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,  num_workers=num_workers, pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)
    test_loader  = DataLoader(test_ds,  batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)
    return train_loader, val_loader, test_loader


def make_meas_loader(meas_h5: str, x_scaler: XStandardizer, batch_size: int, num_workers: int = 0):
    assert os.path.isfile(meas_h5), f"Meas file not found: {meas_h5}"
    with h5py.File(meas_h5, 'r') as f:
        Xm = f['X'][...]
    Nm = Xm.shape[0]
    Xm = Xm.reshape(Nm, -1).astype(np.float32)
    Xm_std = x_scaler.transform(Xm)
    ds = MeasDataset(Xm_std)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)
    return loader, Nm


# ============================
# 5) Proxy train/export/load
# ============================

def _export_proxy_torchscript(model: nn.Module, in_dim: int, device, outdir: str) -> str:
    model = model.eval().to(device)
    try:
        scripted = torch.jit.script(model)
    except Exception:
        example = torch.randn(1, in_dim, device=device)
        scripted = torch.jit.trace(model, example)
    ts_path = os.path.join(outdir, 'proxy_g.ts')
    scripted.save(ts_path)
    return ts_path


def _update_transforms_meta(outdir: str, updates: Dict):
    tf_path = os.path.join(outdir, 'transforms.json')
    if not os.path.isfile(tf_path):
        return
    with open(tf_path, 'r') as f:
        meta = json.load(f)
    meta.update(updates)
    with open(tf_path, 'w') as f:
        json.dump(meta, f, indent=2)


def train_proxy_g(X_tr_std: np.ndarray, Y_tr_norm: np.ndarray,
                  X_va_std: np.ndarray, Y_va_norm: np.ndarray,
                  device, outdir: str,
                  hidden: Tuple[int, ...] = (512, 512), activation: str = 'gelu', norm: str = 'layernorm',
                  max_epochs: int = 100, lr: float = 1e-3, weight_decay: float = 1e-4, beta: float = 0.02,
                  seed: Optional[int] = None,
                  patience: int = 15, min_delta: float = 1e-6,
                  batch_size: int = 1024):

    # 1) 本地种子
    local_seed = seed if seed is not None else _seed_from_proxy_cfg(
        hidden, activation, norm, max_epochs, lr, weight_decay, beta
    )

    with scoped_rng(local_seed):
        in_dim = Y_tr_norm.shape[1]
        out_dim = X_tr_std.shape[1]
        model = ProxyMLP(in_dim, out_dim, list(hidden), activation=activation, norm=norm).to(device)
        opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

        # 2) 正确使用 GradScaler（仅 CUDA 启用）
        scaler = torch.amp.GradScaler('cuda', enabled=True) if device.type == 'cuda' \
         else torch.amp.GradScaler('cpu',  enabled=False)

        crit = nn.SmoothL1Loss(beta=beta)

        def _eval(Xs, Ys):
            model.eval()
            with torch.no_grad():
                xt = torch.from_numpy(Xs).to(device)
                yt = torch.from_numpy(Ys).to(device)
                xhat = model(yt)
                loss = crit(xhat, xt).item()
            model.train()
            return loss

        best = float('inf')
        no_improve = 0
        best_path = os.path.join(outdir, 'proxy_g.pt')
        os.makedirs(outdir, exist_ok=True)

        B = int(batch_size)
        n = X_tr_std.shape[0]
        rng = np.random.default_rng(local_seed)

        # 3) 仅在 CUDA 上开启 autocast；CPU 关闭
        ac_kwargs = (
            dict(device_type='cuda', dtype=torch.float16, enabled=True)
            if device.type == 'cuda' else
            dict(device_type='cpu', dtype=torch.bfloat16, enabled=False)
        )

        for ep in range(1, max_epochs + 1):
            model.train()  # 明确训练态
            order = rng.permutation(n)
            total = 0.0
            cnt = 0

            for i in range(0, n, B):
                idx = order[i:i + B]
                xt = torch.from_numpy(X_tr_std[idx]).to(device)
                yt = torch.from_numpy(Y_tr_norm[idx]).to(device)

                opt.zero_grad(set_to_none=True)
                with torch.autocast(**ac_kwargs):
                    xhat = model(yt)
                    loss = crit(xhat, xt)

                scaler.scale(loss).backward()
                scaler.step(opt)
                scaler.update()

                total += loss.item() * xt.size(0)
                cnt += xt.size(0)

            train_avg = total / max(1, cnt)
            val = _eval(X_va_std, Y_va_norm)
            print(f"[proxy] epoch {ep:03d}  train={train_avg:.6f}  val={val:.6f}  best={best:.6f}  used patience={no_improve + 1}/{patience}")

            # patience 早停
            if val < best - min_delta:
                best = val
                no_improve = 0
                torch.save({
                    'model': model.state_dict(), 'in_dim': in_dim, 'out_dim': out_dim,
                    'hidden': list(hidden), 'activation': activation, 'norm': norm
                }, best_path)
            else:
                no_improve += 1
                if no_improve >= patience:
                    print(f"[proxy] early stopping at epoch {ep} (no improve {patience})")
                    break

        # 载入 best 并导出 TorchScript
        ck = torch.load(best_path, map_location=device)
        model.load_state_dict(ck['model'])
        model.eval()
        ts_path = _export_proxy_torchscript(model, in_dim, device, outdir)

        proxy_meta = {
            'proxy': {
                'arch': 'mlp', 'in_dim': in_dim, 'out_dim': out_dim,
                'hidden': ck.get('hidden', list(hidden)),
                'activation': ck.get('activation', activation),
                'norm': ck.get('norm', norm),
                'format': 'torchscript',
                'files': {'state_dict': os.path.basename(best_path), 'torchscript': os.path.basename(ts_path)}
            }
        }
        _update_transforms_meta(outdir, proxy_meta)

        return model, best_path, ts_path, proxy_meta['proxy']



def load_proxy_artifacts(run_dir: str, device):
    tr_path = os.path.join(run_dir, 'transforms.json')
    ts_path = os.path.join(run_dir, 'proxy_g.ts')
    pt_path = os.path.join(run_dir, 'proxy_g.pt')
    assert os.path.isfile(tr_path), f"transforms.json not found in {run_dir}"

    with open(tr_path, 'r') as f:
        meta = json.load(f)

    # Prefer proxy-specific scalers if present
    if 'proxy_x_scaler' in meta and 'proxy_y_transform' in meta:
        x_scaler = XStandardizer.from_state_dict(meta['proxy_x_scaler'])
        y_tf = YTransform.from_state_dict(meta['proxy_y_transform'])
    else:
        x_scaler = XStandardizer.from_state_dict(meta['x_scaler'])
        y_tf = YTransform.from_state_dict(meta['y_transform'])

    proxy_cfg = meta.get('proxy', None)

    if os.path.isfile(ts_path):
        proxy = torch.jit.load(ts_path, map_location=device)
        try:
            proxy.to(device)
        except Exception:
            pass
        proxy.eval()
        return proxy, x_scaler, y_tf, meta

    if os.path.isfile(pt_path):
        ck = torch.load(pt_path, map_location=device)
        in_dim = int(ck.get('in_dim', len(y_tf.names)))
        out_dim = int(ck.get('out_dim', len(x_scaler.mean)))
        hidden = ck.get('hidden', (proxy_cfg or {}).get('hidden', [512, 512]))
        activation = ck.get('activation', (proxy_cfg or {}).get('activation', 'gelu'))
        norm = ck.get('norm', (proxy_cfg or {}).get('norm', 'layernorm'))
        model = ProxyMLP(in_dim, out_dim, list(hidden), activation=activation, norm=norm).to(device)
        model.load_state_dict(ck['model'])
        model.eval()
        return model, x_scaler, y_tf, meta

    raise FileNotFoundError('proxy_g.ts/pt not found; please train proxy or supply --proxy-run')


# ============================
# 6) Train/eval main
# ============================
class CriterionWrapper:
    def __init__(self, model, use_uncertainty: bool):
        self.model = model
        self.base = nn.SmoothL1Loss(beta=0.02, reduction='none')
        self.use_uncertainty = use_uncertainty and hasattr(model, 'log_sigma')
    def __call__(self, pred, target, return_per_elem=False):
        per_elem = self.base(pred, target)
        if return_per_elem:
            return per_elem
        per_dim = per_elem.mean(dim=0)
        if self.use_uncertainty:
            s = torch.clamp(self.model.log_sigma, -6.0, 6.0)
            return torch.sum(torch.exp(-2*s) * per_dim + s)
        else:
            return per_dim.mean()
    def aggregate(self, per_elem):
        per_dim = per_elem.mean(dim=0)
        if self.use_uncertainty:
            s = torch.clamp(self.model.log_sigma, -6.0, 6.0)
            return torch.sum(torch.exp(-2*s) * per_dim + s)
        else:
            return per_dim.mean()


def train_one_epoch(model, loader, optimizer, scaler, device, criterion,
                    scheduler=None,
                    current_epoch: int = 1,
                    onecycle_epochs: int = 120,
                    y_tf: Optional[YTransform]=None,
                    proxy_g: Optional[nn.Module]=None,
                    lambda_cyc_sim: float=0.0,
                    meas_loader: Optional[DataLoader]=None,
                    lambda_cyc_meas: float=0.0,
                    y_tf_proxy: Optional[YTransform]=None,
                    x_mu_c: Optional[torch.Tensor]=None, x_std_c: Optional[torch.Tensor]=None,
                    x_mu_p: Optional[torch.Tensor]=None, x_std_p: Optional[torch.Tensor]=None,
                    y_idx_c_from_p: Optional[torch.Tensor]=None,
                    sup_weight: float = 0.05,
                    prior_l2: float = 1e-3,
                    prior_bound: float = 1e-3,
                    prior_bound_margin: float = 0.0,
                    trust_alpha: float = 0.0, trust_tau: float = 1.6,
                    yref_proxy_norm: Optional[torch.Tensor] = None,
                    trust_ref_batch: int = 4096,
                    trust_alpha_meas: float = 0.0,
                    cyc_meas_knn_weight: bool = False,
                    cyc_meas_knn_gamma: float = 0.5):
    model.train()
    total = 0.0; n = 0
    total_sup = 0.0                    
    total_prior_l2 = 0.0
    total_prior_bnd = 0.0
    total_cyc_sim = 0.0; total_cyc_meas = 0.0


    meas_iter = iter(meas_loader) if meas_loader is not None else None
    cycle_beta = 0.02
    cyc_crit = nn.SmoothL1Loss(beta=cycle_beta, reduction='mean')  # beta≈Huber阈值


    for batch in loader:
        if len(batch) == 3:
            x, y, w = batch; w = w.to(device, non_blocking=True)
        else:
            x, y = batch; w = None
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        # 1) 主干前向 + L_sup：用 autocast 提速
        with torch.autocast(device_type='cuda' if device.type=='cuda' else 'cpu',
                            dtype=torch.float16 if device.type=='cuda' else torch.bfloat16):
            pred = model(x)
            per_elem = criterion(pred, y, return_per_elem=True)
            if w is not None:
                per_elem = per_elem * w[:, None]
            sup_loss = criterion.aggregate(per_elem)  # 未加权 L_sup
            loss = sup_weight * sup_loss             # 先只加弱监督项

        # 2) 一致性 + 先验：禁用 autocast，强制用 FP32，稳定 pow/softplus
        with torch.autocast(device_type='cuda' if device.type=='cuda' else 'cpu', enabled=False):
            pred32 = pred.to(torch.float32)

            # —— L2 先验（对 y_norm）——
            prior_term_l2 = pred32.pow(2).mean() if prior_l2 > 0.0 else pred32.new_tensor(0.0)

            # 软边界：超出用 softplus 惩罚，避免硬裁剪带来的梯度截断
            # ==== 物理域软边界先验（归一化 + log 量纲）====
            prior_term_bound = pred32.new_tensor(0.0)
            if (prior_bound > 0.0) and (y_tf is not None):
                y_phys = y_tf.inverse(pred32)  # FP32 + log域clamp已在 inverse 中处理
                dev = pred32.device; names = y_tf.names
                lo = torch.tensor([PARAM_RANGE[n][0] for n in names], device=dev, dtype=torch.float32)
                hi = torch.tensor([PARAM_RANGE[n][1] for n in names], device=dev, dtype=torch.float32)
                width = (hi - lo).clamp_min(1e-12)
                log_mask = y_tf.log_mask.to(dev)

                # 线性量纲：按区间宽度归一化 + softplus
                bound_lin = pred32.new_tensor(0.0)
                if (~log_mask).any():
                    y_lin = y_phys[:, ~log_mask]
                    lo_lin = lo[~log_mask]; hi_lin = hi[~log_mask]; w_lin = width[~log_mask]
                    over_hi_lin = softplus0((y_lin - (hi_lin + prior_bound_margin * w_lin)) / w_lin, beta=2.0)
                    over_lo_lin = softplus0(((lo_lin - prior_bound_margin * w_lin) - y_lin) / w_lin, beta=2.0)
                    bound_lin = (over_hi_lin + over_lo_lin).mean()

                # log10 量纲：在 log10 域中施加同样的软边界
                bound_log = pred32.new_tensor(0.0)
                if log_mask.any():
                    y_log = torch.log10(y_phys[:, log_mask].clamp_min(1e-12))
                    lo_log = torch.log10(lo[log_mask].clamp_min(1e-12))
                    hi_log = torch.log10(hi[log_mask].clamp_min(1e-12))
                    w_log  = (hi_log - lo_log).clamp_min(1e-6)
                    over_hi_log = softplus0((y_log - (hi_log + prior_bound_margin * w_log)) / w_log, beta=2.0)
                    over_lo_log = softplus0(((lo_log - prior_bound_margin * w_log) - y_log) / w_log, beta=2.0)
                    bound_log = (over_hi_log + over_lo_log).mean()

                prior_term_bound = bound_lin + bound_log


            loss = loss + prior_l2 * prior_term_l2 + prior_bound * prior_term_bound

            # —— 合成一致（sim）——
            cyc_sim = pred32.new_tensor(0.0)
            if (proxy_g is not None and lambda_cyc_sim > 0.0 and
                y_tf is not None and y_tf_proxy is not None and
                x_mu_c is not None and x_std_c is not None and x_mu_p is not None and x_std_p is not None):
                y_phys = y_tf.inverse(pred32)  # FP32 + 已钳制
                if y_idx_c_from_p is not None:
                    y_phys = y_phys.index_select(1, y_idx_c_from_p)
                y_proxy_norm   = y_tf_proxy.transform(y_phys)
                xhat_proxy_std = proxy_g(y_proxy_norm)
                xhat_phys      = xhat_proxy_std * x_std_p + x_mu_p
                xhat_curr_std  = (xhat_phys - x_mu_c) / x_std_c
                cyc_sim = cyc_crit(xhat_curr_std, x)
                loss = loss + lambda_cyc_sim * cyc_sim

            # —— 测量一致（meas）——
            cyc_meas = pred32.new_tensor(0.0)
            if (proxy_g is not None and meas_iter is not None and lambda_cyc_meas > 0.0 and
                y_tf is not None and y_tf_proxy is not None and
                x_mu_c is not None and x_std_c is not None and x_mu_p is not None and x_std_p is not None):
                try:
                    xm = next(meas_iter)
                except StopIteration:
                    meas_iter = iter(meas_loader); xm = next(meas_iter)
                xm = xm.to(pred32.device, non_blocking=True)
                was_training = model.training
                if was_training: model.eval()
                ym_hat = model(xm)                           # [Bm, Dy_norm]
                if was_training: model.train()

                # 物理域 / 代理域
                ym_phys       = y_tf.inverse(ym_hat.to(torch.float32))
                ym_phys_proj  = (ym_phys.index_select(1, y_idx_c_from_p)
                                if y_idx_c_from_p is not None else ym_phys)
                ym_proxy_norm = y_tf_proxy.transform(ym_phys_proj)       # [Bm, Dy_proxy]

                # 走 proxy 闭环
                xmh_proxy_std = proxy_g(ym_proxy_norm)
                xmh_phys      = xmh_proxy_std * x_std_p + x_mu_p
                xmh_curr_std  = (xmh_phys - x_mu_c) / x_std_c

                # per-sample SmoothL1（与训练同 beta=0.02）
                diff  = (xmh_curr_std - xm).reshape(xm.size(0), -1)
                beta  = 0.02
                absd  = diff.abs()
                cyc_ps_m = torch.where(absd < beta, 0.5 * (diff**2) / beta, absd - 0.5 * beta).mean(dim=1)  # [Bm]

                # （可选）kNN 距离：用于权重和 L_trust_meas，复用一次距离计算
                dmin_m = None
                if ( (cyc_meas_knn_weight or (trust_alpha_meas > 0.0))
                    and (yref_proxy_norm is not None) and (yref_proxy_norm.numel() > 0) ):
                    Nref = yref_proxy_norm.shape[0]
                    if Nref > trust_ref_batch:
                        idx = torch.randint(0, Nref, (trust_ref_batch,), device=yref_proxy_norm.device)
                        yref_sub = yref_proxy_norm.index_select(0, idx)
                    else:
                        yref_sub = yref_proxy_norm
                    dists_m = torch.cdist(ym_proxy_norm, yref_sub, p=2)      # [Bm, Nref_sub]
                    dmin_m  = dists_m.min(dim=1).values                      # [Bm]

                # （可选）kNN 加权 meas 一致损失：越远样本权重大
                if cyc_meas_knn_weight and (dmin_m is not None):
                    # 经验公式：w = (dmin / tau)^gamma ，上界裁剪，避免过大
                    w_knn = torch.clamp(dmin_m / max(1e-6, trust_tau), min=0.0, max=4.0).pow(cyc_meas_knn_gamma).detach()
                    cyc_meas = (w_knn * cyc_ps_m).mean()
                else:
                    cyc_meas = cyc_ps_m.mean()

                loss = loss + lambda_cyc_meas * cyc_meas

                # （可选）L_trust 在 meas 分支：推回代理域后，不要离训练支持域太远
                if (trust_alpha_meas > 0.0) and (dmin_m is not None):
                    trust_loss_m = torch.clamp_min(dmin_m - trust_tau, 0.0).pow(2).mean()
                    loss = loss + trust_alpha_meas * trust_loss_m

            # === PATCH E: L_trust ===
            trust_loss = pred.new_zeros(())
            if (trust_alpha > 0.0 and y_tf is not None and y_tf_proxy is not None
                    and yref_proxy_norm is not None and yref_proxy_norm.numel() > 0):
                # 把 y_norm -> 物理域 -> 代理域的 z 空间（与 proxy 训练归一化口径一致）
                pred32 = pred.to(torch.float32)                      # 保持梯度
                y_phys = y_tf.inverse(pred32)                        # [B, D_phys]
                y_phys_p = (y_phys.index_select(1, y_idx_c_from_p)
                            if y_idx_c_from_p is not None else y_phys)
                y_proxy_norm = y_tf_proxy.transform(y_phys_p)        # [B, D_proxy] (z-score)

                # 参考集按 batch 随机下采样，加速 cdist
                Nref = yref_proxy_norm.shape[0]
                if Nref > trust_ref_batch:
                    idx = torch.randint(0, Nref, (trust_ref_batch,), device=yref_proxy_norm.device)
                    yref_sub = yref_proxy_norm.index_select(0, idx)
                else:
                    yref_sub = yref_proxy_norm

                # kNN(min) 距离（z 空间）
                dists = torch.cdist(y_proxy_norm, yref_sub, p=2)     # [B, Nref_sub]
                dmin = dists.min(dim=1).values                       # [B]
                # hinge 二次：max(0, d - tau)^2
                trust_loss = torch.clamp_min(dmin - trust_tau, 0.0).pow(2).mean()

                loss = loss + trust_alpha * trust_loss


        # —— NaN/Inf 守卫 —— before backwards
        if not torch.isfinite(loss):
            print(f"[nan-guard] non-finite loss at epoch {current_epoch}: "
                f"sup={float(sup_loss):.6g}, cyc_sim={float(cyc_sim):.6g}, "
                f"cyc_meas={float(cyc_meas):.6g}, prior_l2={float(prior_term_l2):.6g}, "
                f"prior_bnd={float(prior_term_bound):.6g} -> skip step")
            optimizer.zero_grad(set_to_none=True)
            continue

        scaler.scale(loss).backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=2.0)
        scaler.step(optimizer)
        scaler.update()
        if (scheduler is not None) and (current_epoch <= onecycle_epochs):
            scheduler.step()

        bs = x.size(0)
        total += loss.item() * bs; n += bs
        total_sup += sup_loss.item() * bs
        total_prior_l2 += float(prior_term_l2) * bs
        total_prior_bnd += float(prior_term_bound) * bs
        total_cyc_sim += (float(cyc_sim) if isinstance(cyc_sim, float) else float(cyc_sim)) * bs
        total_cyc_meas += (float(cyc_meas) if isinstance(cyc_meas, float) else float(cyc_meas)) * bs


    avg = total / max(1, n)
    avg_sup = total_sup / max(1, n)
    avg_prior_l2 = total_prior_l2 / max(1, n)
    avg_prior_bnd = total_prior_bnd / max(1, n)
    avg_cyc_sim = total_cyc_sim / max(1, n)
    avg_cyc_meas = total_cyc_meas / max(1, n)


    return avg, avg_cyc_sim, avg_cyc_meas



@torch.no_grad()
def evaluate_full(model, loader, device, criterion,
                  *,
                  # 与训练保持一致所需上下文
                  y_tf: Optional[YTransform],
                  proxy_g: Optional[nn.Module],
                  lambda_cyc_sim: float,
                  meas_loader: Optional[DataLoader],
                  lambda_cyc_meas: float,
                  y_tf_proxy: Optional[YTransform],
                  x_mu_c: Optional[torch.Tensor], x_std_c: Optional[torch.Tensor],
                  x_mu_p: Optional[torch.Tensor], x_std_p: Optional[torch.Tensor],
                  y_idx_c_from_p: Optional[torch.Tensor],
                  # Step-1/3 的系数
                  sup_weight: float, prior_l2: float, prior_bound: float, prior_bound_margin: float,
                  enforce_bounds: bool = False,
                  diag_cfg: Optional[Dict] = None,
                  yref_proxy_norm: Optional[torch.Tensor] = None,
                  diag_outdir: Optional[str] = None,
                  diag_tag: Optional[str] = None
                  ) -> Dict[str, float]:
    """
    计算与训练期一致的验证/测试指标：
      total = sup_weight*L_sup + lambda_cyc_sim*L_cyc_sim + lambda_cyc_meas*L_cyc_meas + prior_terms
    同时返回未加权的分量，便于日志与早停。
    """

    # diagnostics collector
    diag_rows = []
    diag_enabled = bool(diag_cfg and diag_cfg.get('enable', False))
    diag_max = int(diag_cfg.get('max_samples', 64)) if diag_cfg else 0
    diag_k = int(diag_cfg.get('knn_k', 8)) if diag_cfg else 0
    diag_count = 0

    model.eval()
    cycle_beta = 0.02
    cyc_crit = nn.SmoothL1Loss(beta=cycle_beta, reduction='mean')

    total_sup = 0.0
    total_prior_l2 = 0.0
    total_prior_bnd = 0.0
    total_cyc_sim = 0.0
    n_sup = 0
    n_sim = 0

    # --- 在验证集上：监督项 + 先验 + 合成一致 ---
    for batch in loader:
        if len(batch) == 3:
            x, y, _ = batch
        else:
            x, y = batch
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)

        pred = model(x)

        # L_sup（未加权）
        per_elem = criterion(pred, y, return_per_elem=True)
        sup_loss = criterion.aggregate(per_elem)

        # === Probe-C: per-sample priors/sup ===
        with torch.no_grad():
            # per-sample supervised (简化版：对每样本维度平均；忽略uncertainty权重，仅用于诊断)
            per_elem = criterion(pred, y, return_per_elem=True)    # [B, D]
            sup_ps = per_elem.mean(dim=1)                           # [B]

            # per-sample prior_l2 on y_norm
            prior_l2_ps = pred.pow(2).mean(dim=1)                   # [B]

            # per-sample soft bound prior，与训练一致（线性/对数分别softplus + 区间宽度归一）
            if (prior_bound > 0.0) and (y_tf is not None):
                pred32 = pred.to(torch.float32)
                y_phys = y_tf.inverse(pred32)
                names = y_tf.names; dev = pred32.device
                lo = torch.tensor([PARAM_RANGE[n][0] for n in names], device=dev, dtype=torch.float32)
                hi = torch.tensor([PARAM_RANGE[n][1] for n in names], device=dev, dtype=torch.float32)
                width = (hi - lo).clamp_min(1e-12)
                log_mask = y_tf.log_mask.to(dev)

                # linear dims
                bound_lin = pred32.new_zeros(pred32.size(0))
                if (~log_mask).any():
                    y_lin = y_phys[:, ~log_mask]
                    lo_lin = lo[~log_mask]; hi_lin = hi[~log_mask]; w_lin = width[~log_mask]
                    over_hi_lin = softplus0((y_lin - (hi_lin + prior_bound_margin * w_lin)) / w_lin, beta=2.0)
                    over_lo_lin = softplus0(((lo_lin - prior_bound_margin * w_lin) - y_lin) / w_lin, beta=2.0)
                    bound_lin = (over_hi_lin + over_lo_lin).mean(dim=1)

                # log10 dims
                bound_log = pred32.new_zeros(pred32.size(0))
                if log_mask.any():
                    y_log = torch.log10(y_phys[:, log_mask].clamp_min(1e-12))
                    lo_log = torch.log10(lo[log_mask].clamp_min(1e-12))
                    hi_log = torch.log10(hi[log_mask].clamp_min(1e-12))
                    w_log  = (hi_log - lo_log).clamp_min(1e-6)
                    over_hi_log = softplus0((y_log - (hi_log + prior_bound_margin * w_log)) / w_log, beta=2.0)
                    over_lo_log = softplus0(((lo_log - prior_bound_margin * w_log) - y_log) / w_log, beta=2.0)
                    bound_log = (over_hi_log + over_lo_log).mean(dim=1)

                prior_bnd_ps = bound_lin + bound_log
            else:
                prior_bnd_ps = pred.new_zeros(pred.size(0))


        # prior: L2 on y_norm
        prior_term_l2 = pred.pow(2).mean() if prior_l2 > 0.0 else torch.tensor(0.0, device=pred.device)

        # 软边界：超出用 softplus 惩罚，避免硬裁剪带来的梯度截断
        # ==== 物理域软边界先验（归一化 + log 量纲）====
        pred32 = pred32 = pred.to(torch.float32)
        if (prior_bound > 0.0) and (y_tf is not None):
            y_phys = y_tf.inverse(pred32)  # FP32 + log域clamp已在 inverse 中处理
            dev = pred32.device; names = y_tf.names
            lo = torch.tensor([PARAM_RANGE[n][0] for n in names], device=dev, dtype=torch.float32)
            hi = torch.tensor([PARAM_RANGE[n][1] for n in names], device=dev, dtype=torch.float32)
            width = (hi - lo).clamp_min(1e-12)
            log_mask = y_tf.log_mask.to(dev)

            # 线性量纲：按区间宽度归一化 + softplus
            bound_lin = pred32.new_tensor(0.0)
            if (~log_mask).any():
                y_lin = y_phys[:, ~log_mask]
                lo_lin = lo[~log_mask]; hi_lin = hi[~log_mask]; w_lin = width[~log_mask]
                over_hi_lin = softplus0((y_lin - (hi_lin + prior_bound_margin * w_lin)) / w_lin, beta=2.0)
                over_lo_lin = softplus0(((lo_lin - prior_bound_margin * w_lin) - y_lin) / w_lin, beta=2.0)
                bound_lin = (over_hi_lin + over_lo_lin).mean()

            # log10 量纲：在 log10 域中施加同样的软边界
            bound_log = pred32.new_tensor(0.0)
            if log_mask.any():
                y_log = torch.log10(y_phys[:, log_mask].clamp_min(1e-12))
                lo_log = torch.log10(lo[log_mask].clamp_min(1e-12))
                hi_log = torch.log10(hi[log_mask].clamp_min(1e-12))
                w_log  = (hi_log - lo_log).clamp_min(1e-6)
                over_hi_log = softplus0((y_log - (hi_log + prior_bound_margin * w_log)) / w_log, beta=2.0)
                over_lo_log = softplus0(((lo_log - prior_bound_margin * w_log) - y_log) / w_log, beta=2.0)
                bound_log = (over_hi_log + over_lo_log).mean()

            prior_term_bnd = bound_lin + bound_log
        else:
            prior_term_bnd = torch.tensor(0.0, device=pred.device)

        # 合成一致 (val 上用 x / pred 做闭环）
        cyc_sim = torch.tensor(0.0, device=pred.device)
        proxy_floor_ps = None

        if (proxy_g is not None and lambda_cyc_sim > 0.0 and
            y_tf is not None and y_tf_proxy is not None and
            x_mu_c is not None and x_std_c is not None and x_mu_p is not None and x_std_p is not None):
            y_phys = y_tf.inverse(pred)
            if y_idx_c_from_p is not None:
                y_phys = y_phys.index_select(1, y_idx_c_from_p)
            y_proxy_norm = y_tf_proxy.transform(y_phys)
            xhat_proxy_std = proxy_g(y_proxy_norm)
            xhat_phys = xhat_proxy_std * x_std_p + x_mu_p
            xhat_curr_std = (xhat_phys - x_mu_c) / x_std_c
            cyc_sim = cyc_crit(xhat_curr_std, x)

            # === Probe-A/B/C additions (only when diag is enabled) ===
            if diag_enabled and (y_tf is not None) and (y_tf_proxy is not None) and (proxy_g is not None) and (diag_tag == "test"):
                B = x.size(0)
                print('[Diag] Start to calcaulte diag in test preocess... ')

                # (C-1) per-sample cyc_sim using SmoothL1 (和训练同beta=0.02)
                beta = 0.02
                diff = (xhat_curr_std - x).reshape(B, -1)
                absd = diff.abs()
                cyc_ps = torch.where(absd < beta, 0.5 * (diff**2) / beta, absd - 0.5*beta).mean(dim=1)

                # calc proxy floor per-sample (only once)
                y_phys_here = y_tf.inverse(y.to(torch.float32))  # 物理域
                if y_idx_c_from_p is not None:
                    y_phys_here = y_phys_here.index_select(1, y_idx_c_from_p)  # 对齐到 proxy 的参数列
                y_proxy_norm_here = y_tf_proxy.transform(y_phys_here)
                xhat_proxy_ref_std = proxy_g(y_proxy_norm_here)  # z_x (proxy 标准化)
                xhat_proxy_ref_phys = xhat_proxy_ref_std * x_std_p + x_mu_p  # 物理域
                xhat_proxy_ref_curr_std  = (xhat_proxy_ref_phys - x_mu_c) / x_std_c 
                beta = 0.02
                diff_proxy_ref = (xhat_proxy_ref_curr_std - x).reshape(B, -1)
                absd_proxy_ref = diff_proxy_ref.abs()
                proxy_floor_ps = torch.where(absd_proxy_ref < beta, 0.5 * (diff_proxy_ref**2) / beta, absd_proxy_ref - 0.5*beta).mean(dim=1)
                # 记录到全局列表，便于最后打印分位数
                if 'proxy_floor_all' not in locals():
                    proxy_floor_all = []
                proxy_floor_all.extend(proxy_floor_ps.detach().cpu().tolist())


                # (A) KNN distance to proxy's Y_train_norm (if provided)
                if (yref_proxy_norm is not None):
                    dists = torch.cdist(y_proxy_norm, yref_proxy_norm.to(y_proxy_norm.device), p=2)  # [B, Nref]
                    knn_min = dists.min(dim=1).values
                    if diag_k > 1:
                        knn_vals, _ = dists.topk(k=min(diag_k, dists.size(1)), largest=False, dim=1)
                        knn_mean_k = knn_vals.mean(dim=1)
                    else:
                        knn_mean_k = knn_min
                else:
                    knn_min = torch.full((B,), float('nan'), device=x.device)
                    knn_mean_k = torch.full((B,), float('nan'), device=x.device)

                # (B) Jacobian spectral norm of proxy g at y_proxy_norm (subsampled)
                jac_sig = torch.full((B,), float('nan'), device=x.device)
                if diag_count < diag_max:
                    take = min(B, diag_max - diag_count)
                    for i in range(take):
                        y0 = y_proxy_norm[i].detach().to(torch.float32).requires_grad_(True)
                        def f(inp):
                            return proxy_g(inp.unsqueeze(0)).squeeze(0)   # out_dim vector
                        # J: [out_dim, in_dim]
                        J = torch.autograd.functional.jacobian(f, y0, create_graph=False, strict=True)
                        # SVD on CPU for stability
                        s = torch.linalg.svdvals(J.cpu())
                        jac_sig[i] = s.max().to(x.device)
                    diag_count += take

                # 收集到 diag_rows
                with torch.no_grad():
                    for i in range(B):
                        diag_rows.append({
                            'domain': 'sim',
                            'sup_ps': float(sup_ps[i]),
                            'prior_l2_ps': float(prior_l2_ps[i]),
                            'prior_bnd_ps': float(prior_bnd_ps[i]),
                            'cyc_sim_ps': float(cyc_ps[i]),
                            'knn_min': float(knn_min[i]),
                            'knn_mean_k': float(knn_mean_k[i]),
                            'jac_sigma_max': float(jac_sig[i]),
                            'proxy_floor_ps': float(proxy_floor_ps[i]) if 'proxy_floor_ps' in locals() else float('nan'),
                        })


        bs = x.size(0)
        total_sup += float(sup_loss) * bs
        total_prior_l2 += float(prior_term_l2) * bs
        total_prior_bnd += float(prior_term_bnd) * bs
        total_cyc_sim += float(cyc_sim) * bs
        n_sup += bs
        n_sim += bs

    # --- 在测量集上：测量一致（与训练一致）
    total_cyc_meas = 0.0
    n_meas = 0
    if (proxy_g is not None and meas_loader is not None and lambda_cyc_meas > 0.0 and
        y_tf is not None and y_tf_proxy is not None and
        x_mu_c is not None and x_std_c is not None and x_mu_p is not None and x_std_p is not None):
        for xm in meas_loader:
            xm = xm.to(device, non_blocking=True)
            # eval 模式避免 BN 偏差
            was_training = model.training
            if was_training: model.eval()
            ym_hat = model(xm)
            if was_training: model.train()
            ym_phys = y_tf.inverse(ym_hat)
            if y_idx_c_from_p is not None:
                ym_phys = ym_phys.index_select(1, y_idx_c_from_p)
            ym_proxy_norm = y_tf_proxy.transform(ym_phys)
            xmh_proxy_std = proxy_g(ym_proxy_norm)
            xmh_phys = xmh_proxy_std * x_std_p + x_mu_p
            xmh_curr_std = (xmh_phys - x_mu_c) / x_std_c
            cyc_meas = cyc_crit(xmh_curr_std, xm)

            # === PATCH F (fixed): append MEAS rows into diag using current xm batch ===
            if diag_enabled and (y_tf is not None) and (y_tf_proxy is not None) and (proxy_g is not None) and (diag_tag == "test"):
                model.eval()
                with torch.no_grad():
                    y_norm_m = ym_hat                       # [Bm, Dy_norm]
                    prior_l2_ps_m = y_norm_m.pow(2).mean(dim=1)

                    # per-sample prior_bnd（与训练同：线性/对数 + margin + zero-based softplus）
                    y_phys_m = y_tf.inverse(y_norm_m.to(torch.float32))
                    names = y_tf.names; dev = y_norm_m.device
                    lo = torch.tensor([PARAM_RANGE[n][0] for n in names], device=dev, dtype=torch.float32)
                    hi = torch.tensor([PARAM_RANGE[n][1] for n in names], device=dev, dtype=torch.float32)
                    width = (hi - lo).clamp_min(1e-12)
                    log_mask = y_tf.log_mask.to(dev)

                    bound_lin = y_norm_m.new_zeros(y_norm_m.size(0))
                    if (~log_mask).any():
                        y_lin = y_phys_m[:, ~log_mask]
                        lo_lin = lo[~log_mask]; hi_lin = hi[~log_mask]; w_lin = width[~log_mask]
                        over_hi_lin = softplus0((y_lin - (hi_lin + prior_bound_margin * w_lin)) / w_lin, beta=2.0)
                        over_lo_lin = softplus0(((lo_lin - prior_bound_margin * w_lin) - y_lin) / w_lin, beta=2.0)
                        bound_lin = (over_hi_lin + over_lo_lin).mean(dim=1)

                    bound_log = y_norm_m.new_zeros(y_norm_m.size(0))
                    if log_mask.any():
                        y_log = torch.log10(y_phys_m[:, log_mask].clamp_min(1e-12))
                        lo_log = torch.log10(lo[log_mask].clamp_min(1e-12))
                        hi_log = torch.log10(hi[log_mask].clamp_min(1e-12))
                        w_log  = (hi_log - lo_log).clamp_min(1e-6)
                        over_hi_log = softplus0((y_log - (hi_log + prior_bound_margin * w_log)) / w_log, beta=2.0)
                        over_lo_log = softplus0(((lo_log - prior_bound_margin * w_log) - y_log) / w_log, beta=2.0)
                        bound_log = (over_hi_log + over_lo_log).mean(dim=1)

                    prior_bnd_ps_m = bound_lin + bound_log

                    # per-sample cyc_meas（和训练同口径）
                    xhat_m_std = proxy_g( y_tf_proxy.transform(
                        (y_phys_m.index_select(1, y_idx_c_from_p) if y_idx_c_from_p is not None else y_phys_m)
                    ))
                    diff = (xhat_m_std - xm).reshape(xm.size(0), -1)
                    beta = 0.02
                    absd = diff.abs()
                    cyc_ps_m = torch.where(absd < beta, 0.5*(diff**2)/beta, absd - 0.5*beta).mean(dim=1)

                    # kNN(min) / mean_k
                    knn_min_m = knn_mean_k_m = torch.full_like(cyc_ps_m, float('nan'))
                    if yref_proxy_norm is not None:
                        y_proxy_norm_m = y_tf_proxy.transform(
                            (y_phys_m.index_select(1, y_idx_c_from_p) if y_idx_c_from_p is not None else y_phys_m)
                        )
                        dists = torch.cdist(y_proxy_norm_m, yref_proxy_norm.to(y_proxy_norm_m.device), p=2)
                        knn_min_m = dists.min(dim=1).values
                        k = max(1, diag_k)
                        vals, _ = dists.topk(k=min(k, dists.size(1)), largest=False, dim=1)
                        knn_mean_k_m = vals.mean(dim=1)

                    # 写 rows
                    for i in range(xm.size(0)):
                        diag_rows.append({
                            'domain': 'meas',
                            'sup_ps': float('nan'),
                            'prior_l2_ps': float(prior_l2_ps_m[i]),
                            'prior_bnd_ps': float(prior_bnd_ps_m[i]),
                            'cyc_sim_ps': float(cyc_ps_m[i]),   # 同一列名，图里叠加显示
                            'knn_min': float(knn_min_m[i]),
                            'knn_mean_k': float(knn_mean_k_m[i]),
                            'jac_sigma_max': float('nan'),
                            'proxy_floor_ps': float(proxy_floor_ps[i]) if 'proxy_floor_ps' in locals() else float('nan'),
                        })



            bs = xm.size(0)
            total_cyc_meas += float(cyc_meas) * bs
            n_meas += bs

    # --- 汇总平均 ---
    sup = total_sup / max(1, n_sup)
    prior_l2_avg = total_prior_l2 / max(1, n_sup)
    prior_bnd_avg = total_prior_bnd / max(1, n_sup)
    cyc_sim = total_cyc_sim / max(1, n_sim)
    cyc_meas = total_cyc_meas / max(1, n_meas) if n_meas > 0 else 0.0

    total = (sup_weight * sup
             + lambda_cyc_sim * cyc_sim
             + lambda_cyc_meas * cyc_meas
             + prior_l2 * prior_l2_avg
             + prior_bound * prior_bnd_avg)
    
    # --- write diagnostics CSV ---
    if diag_enabled and diag_outdir and len(diag_rows) > 0 and (diag_tag == "test"):
        print('[Diag] Start to write diag result... ')
        
        tag = diag_tag or 'eval'
        path = os.path.join(diag_outdir, f'diag_{tag}.csv')
        keys = list(diag_rows[0].keys())
        with open(path, 'w', newline='') as f:
            w = csv.DictWriter(f, fieldnames=keys)
            w.writeheader()
            w.writerows(diag_rows)
        print(f"[Diag] Wrote {len(diag_rows)} rows -> {path}")

    # === proxy floor stats (optional print) ===
    if diag_enabled and diag_tag == "test" and ('proxy_floor_all' in locals()) and len(proxy_floor_all) > 0:
        p50 = float(np.percentile(proxy_floor_all, 50))
        p90 = float(np.percentile(proxy_floor_all, 90))
        p95 = float(np.percentile(proxy_floor_all, 95))
        p99 = float(np.percentile(proxy_floor_all, 99))
        print(f"[diag] proxy_floor_ps  P50={p50:.4f}  P90={p90:.4f}  P95={p95:.4f}  P99={p99:.4f}  (beta=0.02)")


    return {
        'total': float(total),
        'sup': float(sup),
        'prior_l2': float(prior_l2_avg),
        'prior_bnd': float(prior_bnd_avg),
        'cyc_sim': float(cyc_sim),
        'cyc_meas': float(cyc_meas),
    }



def mae_per_param(pred_norm: torch.Tensor, gt_norm: torch.Tensor, y_tf: YTransform, enforce_bounds: bool = False) -> Dict[str, float]:
    pred_raw = y_tf.inverse(pred_norm)
    gt_raw = y_tf.inverse(gt_norm)
    if enforce_bounds:
        for i, name in enumerate(y_tf.names):
            lo, hi = PARAM_RANGE.get(name, (-float('inf'), float('inf')))
            pred_raw[:, i] = pred_raw[:, i].clamp(min=lo if not math.isinf(lo) else None,
                                                  max=hi if not math.isinf(hi) else None)
    mae = torch.mean(torch.abs(pred_raw - gt_raw), dim=0).numpy()
    return {name: float(mae[i]) for i, name in enumerate(y_tf.names)}


def save_state(outdir: str, x_scaler: XStandardizer, y_tf: YTransform, cfg: TrainConfig, parent_run: Optional[str] = None):
    os.makedirs(outdir, exist_ok=True)
    meta = {
        'x_scaler': x_scaler.state_dict(),
        'y_transform': y_tf.state_dict(),
        'config': asdict(cfg),
        'param_names': PARAM_NAMES,
        'input_dim': len(x_scaler.mean),
    }
    if parent_run:
        meta['parent_run'] = os.path.abspath(parent_run)
    with open(os.path.join(outdir, 'transforms.json'), 'w') as f:
        json.dump(meta, f, indent=2)


# ============================
# 7) Run once / main loop
# ============================

def _build_model_for_cfg(input_dim: int, output_dim: int, param_names: List[str], cfg_dict: Dict, device) -> nn.Module:
    use_mh = cfg_dict.get('multihead', True)
    use_unc = cfg_dict.get('uncertainty_weighting', True)
    hidden = tuple(cfg_dict.get('hidden', [1024, 512, 256, 128]))
    dropout = cfg_dict.get('dropout', 0.1)
    if use_mh:
        model = MultiHeadMLP(input_dim, list(hidden), param_names, dropout=dropout, use_uncertainty=use_unc).to(device)
    else:
        model = MLP(input_dim, list(hidden), output_dim, dropout=dropout).to(device)
    return model


def _reset_head_layers(model: nn.Module):
    if isinstance(model, MultiHeadMLP):
        for _, head in model.heads.items():
            if isinstance(head, nn.Linear):
                nn.init.kaiming_uniform_(head.weight, a=math.sqrt(5))
                if head.bias is not None:
                    fan_in, _ = nn.init._calculate_fan_in_and_fan_out(head.weight)
                    bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
                    nn.init.uniform_(head.bias, -bound, bound)
        if hasattr(model, 'log_sigma'):
            with torch.no_grad():
                model.log_sigma.zero_()
    elif isinstance(model, MLP):
        if isinstance(model.net[-1], nn.Linear):
            last = model.net[-1]
            nn.init.kaiming_uniform_(last.weight, a=math.sqrt(5))
            if last.bias is not None:
                fan_in, _ = nn.init._calculate_fan_in_and_fan_out(last.weight)
                bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
                nn.init.uniform_(last.bias, -bound, bound)


def _freeze_trunk(model: nn.Module):
    if isinstance(model, MultiHeadMLP):
        for p in model.trunk.parameters():
            p.requires_grad = False
    elif isinstance(model, MLP):
        for m in model.net[:-1]:
            if isinstance(m, (nn.Linear, nn.BatchNorm1d, nn.Dropout, nn.GELU)):
                for p in getattr(m, 'parameters', lambda: [])():
                    p.requires_grad = False


def _load_parent_artifacts(ft_path: str, device) -> Tuple[Optional[Dict], Optional[str]]:
    if os.path.isdir(ft_path):
        tr = os.path.join(ft_path, 'transforms.json')
        ck = os.path.join(ft_path, 'best_model.pt')
        if not (os.path.isfile(tr) and os.path.isfile(ck)):
            raise FileNotFoundError(f"finetune_from dir missing files: {tr} / {ck}")
        with open(tr, 'r') as f:
            meta = json.load(f)
        return meta, ck
    else:
        ck = ft_path
        if not os.path.isfile(ck):
            raise FileNotFoundError(f"Checkpoint not found: {ck}")
        meta = None
        parent = os.path.dirname(os.path.abspath(ck))
        tr = os.path.join(parent, 'transforms.json')
        if os.path.isfile(tr):
            with open(tr, 'r') as f:
                meta = json.load(f)
        return meta, ck

def run_proxy_only(cfg: TrainConfig, device):
    """仅训练代理 g(Y->X)，并将其与独立的缩放/变换保存到一个新的 run 目录。"""
    if cfg.data is None:
        raise SystemExit('--data is required for --train-proxy-only')

    # 读取完整数据以单独拟合 proxy 的两套缩放（与主模型独立）
    train_ds, val_ds, _, x_scaler_tmp, y_tf_tmp, splits, X_all, Y_all = load_and_prepare(cfg.data, cfg)
    tr_idx, va_idx, _ = splits

    N = X_all.shape[0]
    X_flat = X_all.reshape(N, -1).astype(np.float32)
    Y_flat = Y_all.reshape(N, len(PARAM_NAMES)).astype(np.float32)

    # 独立的 proxy 输入标准化（基于 train split）
    x_scaler_p = XStandardizer(); x_scaler_p.fit(X_flat[tr_idx])
    X_tr_std_p = x_scaler_p.transform(X_flat[tr_idx])
    X_va_std_p = x_scaler_p.transform(X_flat[va_idx])

    # 独立的 proxy 输出变换（log10+zscore，基于 train split）
    log_mask_np = choose_log_mask(PARAM_RANGE, PARAM_NAMES)
    y_tf_p = YTransform(PARAM_NAMES, log_mask_np)
    y_tf_p.fit(torch.from_numpy(Y_flat[tr_idx]))
    Y_tr_norm_p = y_tf_p.transform(torch.from_numpy(Y_flat[tr_idx])).numpy()
    Y_va_norm_p = y_tf_p.transform(torch.from_numpy(Y_flat[va_idx])).numpy()

    # 运行目录
    stamp = time.strftime('%Y%m%d-%H%M%S')
    run_dir = os.path.join(cfg.outdir, f'proxy_run_{stamp}')
    os.makedirs(run_dir, exist_ok=True)

    # 先保存（写 transforms.json，便于下游 --proxy-run 直接读取）
    save_state(run_dir, x_scaler_p, y_tf_p, cfg)

    # 保存 proxy 训练集的 Y_train_norm（用于 Probe-A 的 KNN 距离）
    np.save(os.path.join(run_dir, 'proxy_Ytr_norm.npy'), Y_tr_norm_p)

    # 训练并导出 TorchScript
    proxy_g, pt_path, ts_path, _ = train_proxy_g(
        X_tr_std_p, Y_tr_norm_p, X_va_std_p, Y_va_norm_p, device, run_dir,
        hidden=cfg.proxy_hidden, activation=cfg.proxy_activation, norm=cfg.proxy_norm,
        max_epochs=cfg.proxy_epochs, lr=cfg.proxy_lr, weight_decay=cfg.proxy_wd, beta=cfg.proxy_beta,
        seed=cfg.proxy_seed,
        patience=cfg.proxy_patience, min_delta=cfg.proxy_min_delta,
        batch_size=cfg.proxy_batch_size
    )

    # 把代理文件记录进 transforms.json（供主模型读取元数据）
    files = {'proxy_g.pt': os.path.basename(pt_path), 'proxy_g.ts': os.path.basename(ts_path)}
    _update_transforms_meta(run_dir, {'proxy': {
        'arch': 'mlp',
        'in_dim': len(PARAM_NAMES),
        'out_dim': len(x_scaler_p.mean),
        'hidden': list(cfg.proxy_hidden),
        'activation': cfg.proxy_activation,
        'norm': cfg.proxy_norm,
        'format': 'torchscript',
        'files': files
    }})

    print(f"[ProxyOnly] Saved to: {run_dir}\n  - TorchScript: {ts_path}\n  - Checkpoint:  {pt_path}")
    return {'run_dir': run_dir, 'proxy_ts': ts_path, 'proxy_pt': pt_path}

def run_once(cfg: TrainConfig, diag_cfg: dict, device):
    # finetune artifacts
    parent_meta: Optional[Dict] = None
    parent_ckpt: Optional[str] = None
    parent_run_dir: Optional[str] = None
    override_scalers: Optional[Tuple[XStandardizer, YTransform]] = None

    if cfg.finetune_from:
        parent_meta, parent_ckpt = _load_parent_artifacts(cfg.finetune_from, device)
        if parent_meta is not None and cfg.ft_use_prev_transforms:
            x_scaler_parent = XStandardizer.from_state_dict(parent_meta['x_scaler'])
            y_tf_parent = YTransform.from_state_dict(parent_meta['y_transform'])
            override_scalers = (x_scaler_parent, y_tf_parent)
            parent_run_dir = os.path.dirname(os.path.abspath(parent_ckpt)) if parent_ckpt else cfg.finetune_from
        elif parent_meta is not None:
            parent_run_dir = os.path.dirname(os.path.abspath(parent_ckpt)) if parent_ckpt else cfg.finetune_from

    # data
    train_ds, val_ds, test_ds, x_scaler, y_tf, splits, X_all, Y_all = load_and_prepare(cfg.data, cfg, override_scalers)
    train_loader, val_loader, test_loader = make_loaders(train_ds, val_ds, test_ds, cfg.batch_size, cfg.num_workers)

    input_dim = train_ds.x.shape[1]
    output_dim = train_ds.y.shape[1]

    # model
    if cfg.finetune_from and (parent_meta is not None) and (not cfg.ft_allow_arch_change):
        prev_cfg = parent_meta.get('config', {})
        param_names = parent_meta.get('param_names', PARAM_NAMES)
        model = _build_model_for_cfg(input_dim, output_dim, param_names, prev_cfg, device)
    else:
        if cfg.multihead:
            model = MultiHeadMLP(input_dim, list(cfg.hidden), PARAM_NAMES, dropout=cfg.dropout, use_uncertainty=cfg.uncertainty_weighting).to(device)
        else:
            model = MLP(input_dim, list(cfg.hidden), output_dim, dropout=cfg.dropout).to(device)

    if cfg.compile and hasattr(torch, 'compile'):
        try:
            model = torch.compile(model)  # type: ignore
        except Exception:
            pass

    # load finetune weights
    if cfg.finetune_from and parent_ckpt is not None:
        ckpt = torch.load(parent_ckpt, map_location=device)
        sd = ckpt.get('model', ckpt)
        try:
            model.load_state_dict(sd, strict=cfg.ft_strict)
            print(f"[Finetune] Loaded weights from {parent_ckpt} (strict={cfg.ft_strict})")
        except Exception:
            missing, unexpected = model.load_state_dict(sd, strict=False)
            print(f"[Finetune] Non-strict load. Missing: {len(missing)}, Unexpected: {len(unexpected)}")
        if cfg.ft_reset_head:
            _reset_head_layers(model); print("[Finetune] Reset head(s)")
        if cfg.ft_freeze_trunk:
            _freeze_trunk(model); print("[Finetune] Freeze trunk")

    criterion_wrap = CriterionWrapper(model, use_uncertainty=getattr(model, 'use_uncertainty', False) or cfg.uncertainty_weighting)
    optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=cfg.lr, weight_decay=cfg.weight_decay)
    steps_per_epoch = max(1, len(train_loader))
    scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=cfg.lr, epochs=cfg.onecycle_epochs, steps_per_epoch=steps_per_epoch) if cfg.use_onecycle else None
    scaler = torch.amp.GradScaler('cuda' if device.type=='cuda' else 'cpu', enabled=(device.type=='cuda'))

    stamp = time.strftime('%Y%m%d-%H%M%S')
    tag = 'ft' if cfg.finetune_from else 'asm_hemt_dnn'
    run_dir = os.path.join(cfg.outdir, f'{tag}_{stamp}')
    os.makedirs(run_dir, exist_ok=True)
    writer = SummaryWriter(log_dir=run_dir)

    meas_loader = None
    if cfg.meas_h5 and cfg.lambda_cyc_meas > 0.0:
        meas_loader, _ = make_meas_loader(cfg.meas_h5, x_scaler, cfg.batch_size, cfg.num_workers)

    # proxy (load or auto-train). ALWAYS keep proxy scalers INDEPENDENT of main scalers
    proxy_g = None; proxy_paths = {}
    x_scaler_p: Optional[XStandardizer] = None
    y_tf_p: Optional[YTransform] = None

    if (cfg.lambda_cyc_sim > 0.0 or cfg.lambda_cyc_meas > 0.0):
        if cfg.proxy_run:
            try:
                proxy_g, x_scaler_p, y_tf_p, meta = load_proxy_artifacts(cfg.proxy_run, device)
                proxy_paths['proxy_g.ts'] = os.path.join(cfg.proxy_run, 'proxy_g.ts')
                proxy_paths['proxy_g.pt'] = os.path.join(cfg.proxy_run, 'proxy_g.pt')
            except Exception as e:
                print(f"[Warn] load proxy from {cfg.proxy_run} failed: {e}")
        elif cfg.auto_train_proxy:
            tr_idx, va_idx, _ = splits
            # independent scalers for proxy
            X_flat = X_all.reshape(X_all.shape[0], -1).astype(np.float32)
            Y_flat = Y_all.reshape(Y_all.shape[0], len(PARAM_NAMES)).astype(np.float32)

            x_scaler_p = XStandardizer(); x_scaler_p.fit(X_flat[tr_idx])
            X_tr_std_p = x_scaler_p.transform(X_flat[tr_idx]); X_va_std_p = x_scaler_p.transform(X_flat[va_idx])

            log_mask_np = choose_log_mask(PARAM_RANGE, PARAM_NAMES)
            y_tf_p = YTransform(PARAM_NAMES, log_mask_np)
            y_tf_p.fit(torch.from_numpy(Y_flat[tr_idx]))
            Y_tr_norm_p = y_tf_p.transform(torch.from_numpy(Y_flat[tr_idx])).numpy()
            Y_va_norm_p = y_tf_p.transform(torch.from_numpy(Y_flat[va_idx])).numpy()

            # 保存 proxy 训练集的 Y_train_norm（用于 Probe-A 的 KNN 距离）
            np.save(os.path.join(run_dir, 'proxy_Ytr_norm.npy'), Y_tr_norm_p)

            proxy_g_eager, pt_path, ts_path, _ = train_proxy_g(
                X_tr_std_p, Y_tr_norm_p, X_va_std_p, Y_va_norm_p, device, run_dir,
                hidden=cfg.proxy_hidden, activation=cfg.proxy_activation, norm=cfg.proxy_norm,
                max_epochs=cfg.proxy_epochs, lr=cfg.proxy_lr, weight_decay=cfg.proxy_wd, beta=cfg.proxy_beta,
                seed=cfg.proxy_seed,
                patience=cfg.proxy_patience, min_delta=cfg.proxy_min_delta,  
                batch_size=cfg.proxy_batch_size
            )
            proxy_paths['proxy_g.ts'] = ts_path
            proxy_paths['proxy_g.pt'] = pt_path

            # store proxy-specific scalers in transforms.json
            _update_transforms_meta(run_dir, {
                'proxy_x_scaler': x_scaler_p.state_dict(),
                'proxy_y_transform': y_tf_p.state_dict(),
            })

            # IMPORTANT: replace eager with TorchScript for parity with --proxy-run
            try:
                proxy_g = torch.jit.load(ts_path, map_location=device)
                proxy_g.eval()
            except Exception as e:
                print(f"[Warn] failed to load scripted proxy, keep eager: {e}")
                proxy_g = proxy_g_eager
        else:
            print("[Warn] Consistency enabled but no proxy provided; will skip.")

    # save transforms/config
    save_state(run_dir, x_scaler, y_tf, cfg, parent_run=parent_run_dir)
    if proxy_paths:
        files = {k: os.path.basename(v) for k, v in proxy_paths.items() if v and os.path.isfile(v)}
        _update_transforms_meta(run_dir, {'proxy': {
            'arch': 'mlp', 'in_dim': len(PARAM_NAMES), 'out_dim': len(x_scaler.mean),
            'hidden': list(cfg.proxy_hidden), 'activation': cfg.proxy_activation, 'norm': cfg.proxy_norm,
            'format': 'torchscript' if 'proxy_g.ts' in files else 'state_dict', 'files': files
        }})

    # tensors for coord mapping
    x_mu_c  = torch.tensor(x_scaler.mean, device=device, dtype=torch.float32)
    x_std_c = torch.tensor(x_scaler.std,  device=device, dtype=torch.float32)

    if proxy_g is not None and x_scaler_p is not None and y_tf_p is not None:
        x_mu_p  = torch.tensor(x_scaler_p.mean, device=device, dtype=torch.float32)
        x_std_p = torch.tensor(x_scaler_p.std,  device=device, dtype=torch.float32)
        y_tf_proxy = y_tf_p
    elif proxy_g is not None:
        # fallback: use current scalers if proxy-specific not available
        x_mu_p, x_std_p = x_mu_c, x_std_c
        y_tf_proxy = y_tf
    else:
        x_mu_p = x_std_p = y_tf_proxy = None

    # y order mapping current->proxy
    if proxy_g is not None:
        y_names_curr = list(y_tf.names)
        y_names_proxy = list(y_tf_proxy.names)
        name2idx_curr = {n: i for i, n in enumerate(y_names_curr)}
        try:
            idx_list = [name2idx_curr[n] for n in y_names_proxy]
            y_idx_c_from_p = torch.tensor(idx_list, device=device, dtype=torch.long)
        except Exception as e:
            print(f"[Warn] Y name order mismatch: {e}; identity map.")
            y_idx_c_from_p = torch.arange(len(y_names_curr), device=device, dtype=torch.long)
    else:
        y_idx_c_from_p = None
    
    # === diagnostics reference set for Probe-A ===
    yref_proxy_norm = None
    if (proxy_g is not None):
        probe_path_local = os.path.join(run_dir, 'proxy_Ytr_norm.npy')
        probe_path_proxy = os.path.join(cfg.proxy_run, 'proxy_Ytr_norm.npy') if cfg.proxy_run else None
        probep = probe_path_local if os.path.isfile(probe_path_local) else (probe_path_proxy if (probe_path_proxy and os.path.isfile    (probe_path_proxy)) else None)
        if probep:
            # === PATCH B: load reference Y_train_norm for L_trust ===
            if cfg.trust_alpha > 0.0 and proxy_g is not None:
                arr = np.load(probep).astype(np.float32)
                if arr.shape[0] > cfg.trust_ref_max:
                    idx = np.random.choice(arr.shape[0], cfg.trust_ref_max, replace=False)
                    arr = arr[idx]
                yref_proxy_norm = torch.from_numpy(arr).to(device)
                print(f"[L_trust] using {arr.shape[0]} ref rows from {probep}")
            else:
                arr = np.load(probep).astype(np.float32)
                # 去重/抽样可选：如训练集过大，可下采样
                yref_proxy_norm = torch.from_numpy(arr).to(device)
        else:
            print("[Diag] proxy_Ytr_norm.npy not found; Probe-A will be skipped.")
            print("[L_trust] proxy_Ytr_norm.npy not found; L_trust will be skipped.")


    # train
    best_val = float('inf'); best_path = os.path.join(run_dir, 'best_model.pt')
    patience = cfg.patience; no_improve = 0

    for epoch in range(1, cfg.max_epochs + 1):
        lam_meas = cfg.lambda_cyc_meas * min(1.0, epoch / max(1, cfg.cyc_warmup_epochs)) if (cfg.lambda_cyc_meas > 0 and cfg.cyc_warmup_epochs > 0) else cfg.lambda_cyc_meas
        train_loss, cyc_sim, cyc_meas = train_one_epoch(
            model, train_loader, optimizer, scaler, device, CriterionWrapper(model, use_uncertainty=getattr(model, 'use_uncertainty', False) or cfg.uncertainty_weighting),
            scheduler, current_epoch=epoch, onecycle_epochs=cfg.onecycle_epochs,
            y_tf=y_tf, proxy_g=proxy_g, lambda_cyc_sim=cfg.lambda_cyc_sim,
            meas_loader=meas_loader, lambda_cyc_meas=lam_meas,
            y_tf_proxy=y_tf_proxy, x_mu_c=x_mu_c, x_std_c=x_std_c, x_mu_p=x_mu_p, x_std_p=x_std_p,
            y_idx_c_from_p=y_idx_c_from_p,
            sup_weight=cfg.sup_weight,
            prior_l2=cfg.prior_l2,
            prior_bound=cfg.prior_bound,
            prior_bound_margin=cfg.prior_bound_margin,
            trust_alpha=cfg.trust_alpha, trust_tau=cfg.trust_tau,
            yref_proxy_norm=yref_proxy_norm, trust_ref_batch=cfg.trust_ref_batch,
            trust_alpha_meas=cfg.trust_alpha_meas,
            cyc_meas_knn_weight=cfg.cyc_meas_knn_weight,
            cyc_meas_knn_gamma=cfg.cyc_meas_knn_gamma
        )
        val_metrics = evaluate_full(
            model, val_loader, device, criterion_wrap,
            y_tf=y_tf, proxy_g=proxy_g, lambda_cyc_sim=cfg.lambda_cyc_sim,
            meas_loader=meas_loader, lambda_cyc_meas=cfg.lambda_cyc_meas,
            y_tf_proxy=y_tf_p,
            x_mu_c=x_mu_c, x_std_c=x_std_c, x_mu_p=x_mu_p, x_std_p=x_std_p,
            y_idx_c_from_p=y_idx_c_from_p,
            sup_weight=cfg.sup_weight,
            prior_l2=cfg.prior_l2, prior_bound=cfg.prior_bound, prior_bound_margin=cfg.prior_bound_margin,
            enforce_bounds=cfg.enforce_bounds,
            diag_cfg=diag_cfg, yref_proxy_norm=yref_proxy_norm, diag_outdir=run_dir, diag_tag=f"val_ep{epoch:03d}"
        )

        writer.add_scalar('val/total', val_metrics['total'], epoch)
        writer.add_scalar('val/sup', val_metrics['sup'], epoch)
        writer.add_scalar('val/cyc_sim', val_metrics['cyc_sim'], epoch)
        writer.add_scalar('val/cyc_meas', val_metrics['cyc_meas'], epoch)
        writer.add_scalar('val/prior_l2', val_metrics['prior_l2'], epoch)
        writer.add_scalar('val/prior_bnd', val_metrics['prior_bnd'], epoch)

        print(f" >> Epoch {epoch:03d} | train {train_loss:.6f} | [val] total={val_metrics['total']:.6f} | sup={val_metrics['sup']:.6f} | "
            f"cyc_sim={val_metrics['cyc_sim']:.6f} | cyc_meas={val_metrics['cyc_meas']:.6f} | "
            f"prior_l2={val_metrics['prior_l2']:.6e} | prior_bnd={val_metrics['prior_bnd']:.6e} | best val {best_val:.6f} | used patience {no_improve + 1}/{patience} ")

        # 选择用于早停的指标
        es_value = val_metrics['total'] if cfg.es_metric == 'val_total' else \
                val_metrics['sup'] if cfg.es_metric == 'val_sup' else \
                val_metrics['cyc_sim'] if cfg.es_metric == 'val_cyc_sim' else \
                val_metrics['cyc_meas']  # 'val_cyc_meas'

        if es_value < best_val - cfg.es_min_delta:
            best_val = es_value
            no_improve = 0
            torch.save({'model': model.state_dict()}, best_path)
            print(f"[Update] best {cfg.es_metric} improved to {best_val:.6f} @ epoch {epoch}")
        else:
            no_improve += 1
            if no_improve >= patience:
                print(f"[EarlyStop] stop at epoch {epoch} (no improve {patience} on {cfg.es_metric})")
                break


    # === Load best and run Test (cycle-first metric) ===
    ckpt = torch.load(best_path, map_location=device)
    model.load_state_dict(ckpt['model'])
    test_metrics = evaluate_full(
        model, test_loader, device, criterion_wrap,
        y_tf=y_tf, proxy_g=proxy_g, lambda_cyc_sim=cfg.lambda_cyc_sim,
        meas_loader=meas_loader, lambda_cyc_meas=cfg.lambda_cyc_meas,
        y_tf_proxy=y_tf_p,
        x_mu_c=x_mu_c, x_std_c=x_std_c, x_mu_p=x_mu_p, x_std_p=x_std_p,
        y_idx_c_from_p=y_idx_c_from_p,
        sup_weight=cfg.sup_weight,
        prior_l2=cfg.prior_l2, prior_bound=cfg.prior_bound, prior_bound_margin=cfg.prior_bound_margin,
        enforce_bounds=cfg.enforce_bounds,
        diag_cfg=diag_cfg, yref_proxy_norm=yref_proxy_norm, diag_outdir=run_dir, diag_tag="test"
    )
    print(f"[test] total={test_metrics['total']:.6f} | sup={test_metrics['sup']:.6f} | "
        f"cyc_sim={test_metrics['cyc_sim']:.6f} | cyc_meas={test_metrics['cyc_meas']:.6f} | "
        f"prior_l2={test_metrics['prior_l2']:.6e} | prior_bnd={test_metrics['prior_bnd']:.6e}")

    writer.add_scalar('test/total', test_metrics['total'], epoch)
    writer.add_scalar('test/sup', test_metrics['sup'], epoch)
    writer.add_scalar('test/cyc_sim', test_metrics['cyc_sim'], epoch)
    writer.add_scalar('test/cyc_meas', test_metrics['cyc_meas'], epoch)
    writer.add_scalar('test/prior_l2', test_metrics['prior_l2'], epoch)
    writer.add_scalar('test/prior_bnd', test_metrics['prior_bnd'], epoch)

    # 记录超参与最终指标（注意使用新的指标键名）
    add_hparams_safe(
    writer,
    run_dir,
    {"tag": "final"},
    {
        'final/test_total':    test_metrics['total'],
        'final/test_sup':      test_metrics['sup'],
        'final/test_cyc_sim':  test_metrics['cyc_sim'],
        'final/test_cyc_meas': test_metrics['cyc_meas'],
        'final/test_prior_l2': test_metrics['prior_l2'],
        'final/test_prior_bnd':test_metrics['prior_bnd']
    }
    )

    writer.flush()
    writer.close()

    return {
        'run_dir': run_dir,
        'best_model': best_path,
        # 与训练/验证同口径的测试指标
        'test_total':    test_metrics['total'],
        'test_sup':      test_metrics['sup'],
        'test_cyc_sim':  test_metrics['cyc_sim'],
        'test_cyc_meas': test_metrics['cyc_meas'],
        'test_prior_l2': test_metrics['prior_l2'],
        'test_prior_bnd':test_metrics['prior_bnd'],
        # 早停时的最好验证值（按 cfg.es_metric）
        'best_val': best_val,
    }


# ============================
# 8) Inference & CLI
# ============================

def _flatten_X_any(X: np.ndarray) -> np.ndarray:
    """把 X 统一 reshape 到 (N, F)。
       支持 (N,7,121)、(7,121)、(F,) 等输入。
    """
    X = np.asarray(X)
    if X.ndim == 3:
        X = X.reshape(X.shape[0], -1)
    elif X.ndim == 2:
        # 可能传入单样本 7x121
        if X.shape == (7, 121):
            X = X.reshape(1, -1)
    elif X.ndim == 1:
        X = X.reshape(1, -1)
    elif X.ndim >= 4:
        X = X.reshape(X.shape[0], -1)
    return X.astype(np.float32)


def load_artifacts(run_dir: str, device):
    """加载主模型 best_model.pt & transforms.json，并按照训练时结构还原模型。"""
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
        model = MultiHeadMLP(input_dim, list(hidden), param_names,
                             dropout=cfg.get('dropout', 0.1),
                             use_uncertainty=use_unc).to(device)
    else:
        model = MLP(input_dim, list(hidden), output_dim,
                    dropout=cfg.get('dropout', 0.1)).to(device)

    ckpt = torch.load(md_path, map_location=device)
    model.load_state_dict(ckpt['model'])
    model.eval()
    return model, x_scaler, y_tf, param_names, meta


def infer_cli(args, device):
    """主模型 X->Y 推理（支持 npy/h5 输入、可选保存 CSV）。"""
    model, x_scaler, y_tf, names, meta = load_artifacts(args.infer_run, device)
    expected_F = int(meta.get('input_dim', len(x_scaler.mean)))

    # 读取输入
    if args.input_npy:
        X = np.load(args.input_npy)
    elif args.input_h5:
        with h5py.File(args.input_h5, 'r') as f:
            X = f['X'][...]
        if args.index is not None and X.ndim >= 3:
            idx = int(np.clip(args.index, 0, X.shape[0]-1))
            X = X[idx]
    else:
        raise ValueError('Please provide --input-npy or --input-h5')

    x = _flatten_X_any(X)
    if x.shape[1] != expected_F:
        raise ValueError(f"Flattened feature dim {x.shape[1]} != expected {expected_F}. Check grid/shape.")

    # 标准化 + 前向
    x_std = x_scaler.transform(x)
    with torch.no_grad():
        xt = torch.from_numpy(x_std).to(device)
        pred_norm = model(xt)
        pred_raw = y_tf.inverse(pred_norm).cpu()

        # 可选择把输出裁剪回物理边界（与训练期 metrics 一致）
        if meta.get('config', {}).get('enforce_bounds', True):
            for i, n in enumerate(names):
                lo, hi = PARAM_RANGE.get(n, (-float('inf'), float('inf')))
                pred_raw[:, i] = pred_raw[:, i].clamp(
                    min=lo if not math.isinf(lo) else None,
                    max=hi if not math.isinf(hi) else None
                )
        pred = pred_raw.numpy()

    # 打印
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

    # 保存
    if args.save_csv:
        with open(args.save_csv, 'w', newline='') as f:
            wcsv = csv.writer(f)
            if N == 1:
                wcsv.writerow(['param', 'value'])
                for i, n in enumerate(names):
                    wcsv.writerow([n, pred[0, i]])
            else:
                wcsv.writerow(['index'] + names)
                for idx in range(N):
                    wcsv.writerow([idx] + [pred[idx, i] for i in range(len(names))])
        print(f"Saved CSV -> {args.save_csv}")


def infer_proxy_cli(args, device):
    """proxy g: Y->X 推理 + （如有真值 X）误差评估。"""
    proxy_g, x_scaler, y_tf, meta = load_proxy_artifacts(args.infer_proxy_run, device)

    assert args.proxy_input_h5 is not None, "--proxy-input-h5 is required for proxy inference"
    with h5py.File(args.proxy_input_h5, 'r') as f:
        if 'Y' not in f:
            raise ValueError('H5 must contain dataset Y for proxy inference')
        Y = f['Y'][...]
        X_true = f['X'][...] if 'X' in f else None

    N = Y.shape[0]
    if args.proxy_index is not None:
        idx = int(np.clip(args.proxy_index, 0, N-1))
        Y = Y[idx:idx+1]
        if X_true is not None:
            X_true = X_true[idx:idx+1]
        N = 1

    Y = Y.reshape(N, len(PARAM_NAMES)).astype(np.float32)
    y_norm = y_tf.transform(torch.from_numpy(Y)).to(device)

    with torch.no_grad():
        xhat_std = proxy_g(y_norm).cpu().numpy()
        xhat_phys = x_scaler.inverse(xhat_std)

    if X_true is not None:
        X_true = X_true.reshape(N, -1).astype(np.float32)
        X_true_std = x_scaler.transform(X_true)
        mae_std = float(np.mean(np.abs(xhat_std - X_true_std)))
        mae_phys = float(np.mean(np.abs(xhat_phys - X_true)))
        rmse_phys = float(np.sqrt(np.mean((xhat_phys - X_true) ** 2)))
        print(f"[proxy eval] N={N}  MAE_std={mae_std:.6f}  MAE_phys={mae_phys:.6f}  RMSE_phys={rmse_phys:.6f}")
    else:
        print(f"[proxy infer] N={N}  (no ground-truth X to compare)")

    # 展示一条基本统计
    show = min(1, N)
    for s in range(show):
        xs = xhat_phys[s].reshape(7, 121)
        print(f"[proxy sample {s}] X_hat shape={xs.shape}, min={xs.min():.4g}, max={xs.max():.4g}, mean={xs.mean():.4g}")

    if args.save_xhat_npy:
        np.save(args.save_xhat_npy, xhat_phys.reshape(N, 7, 121))
        print(f"Saved X_hat (physical) to: {args.save_xhat_npy}")

def parse_args():
    p = argparse.ArgumentParser(description='ASM-HEMT DNN with proxy consistency and finetune (clean)')
    p.add_argument('--data', type=str, required=True)
    p.add_argument('--outdir', type=str, default='runs')
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--test-split', type=float, default=0.15)
    p.add_argument('--val-split', type=float, default=0.15)
    p.add_argument('--max-epochs', type=int, default=200)
    p.add_argument('--onecycle-epochs', type=int, default=0)
    p.add_argument('--batch-size', type=int, default=256)
    p.add_argument('--lr', type=float, default=1e-3)
    p.add_argument('--weight-decay', type=float, default=1e-4)
    p.add_argument('--dropout', type=float, default=0.1)
    p.add_argument('--hidden', type=str, default='960,512,256,128')
    p.add_argument('--patience', type=int, default=20)
    p.add_argument('--num-workers', type=int, default=0)
    p.add_argument('--compile', action='store_true')
    p.add_argument('--no-onecycle', action='store_true')
    p.add_argument('--aug-noise-std', type=float, default=0.015)
    p.add_argument('--aug-prob', type=float, default=0.5)
    p.add_argument('--no-multihead', action='store_true')
    p.add_argument('--no-uncertainty', action='store_true')
    p.add_argument('--no-bounds', action='store_true')
    # consistency & proxy
    p.add_argument('--meas-h5', type=str, default=None)
    p.add_argument('--lambda-cyc-sim', type=float, default=0.0)
    p.add_argument('--lambda-cyc-meas', type=float, default=0.0)
    p.add_argument('--cyc-warmup-epochs', type=int, default=15)
    p.add_argument('--proxy-run', type=str, default=None)
    p.add_argument('--auto-train-proxy', action='store_true')
    # proxy hyperparams
    p.add_argument('--proxy-hidden', type=str, default='512,512')
    p.add_argument('--proxy-activation', type=str, default='gelu', choices=['gelu','relu','silu'])
    p.add_argument('--proxy-norm', type=str, default='layernorm', choices=['layernorm','batchnorm','none'])
    p.add_argument('--proxy-epochs', type=int, default=100)
    p.add_argument('--proxy-lr', type=float, default=1e-3)
    p.add_argument('--proxy-wd', type=float, default=1e-4)
    p.add_argument('--proxy-beta', type=float, default=0.02)
    p.add_argument('--proxy-seed', type=int, default=None)
    p.add_argument('--proxy-patience', type=int, default=15,
                    help='Early stopping patience for proxy training')
    p.add_argument('--proxy-min-delta', type=float, default=1e-6,
                    help='Minimum improvement on val to reset patience for proxy')
    p.add_argument('--proxy-batch-size', type=int, default=1024,
                    help='Batch size for proxy training (default: 1024)')
    p.add_argument('--train-proxy-only', action='store_true',
                    help='Only train the proxy g then exit')
    # finetune
    p.add_argument('--finetune-from', type=str, default=None)
    p.add_argument('--ft-use-prev-transforms', action='store_true')
    p.add_argument('--ft-freeze-trunk', action='store_true')
    p.add_argument('--ft-reset-head', action='store_true')
    p.add_argument('--ft-allow-arch-change', action='store_true')
    p.add_argument('--ft-non-strict', action='store_true')

    # main model inference
    p.add_argument('--infer-run', type=str, default=None,
                        help='Run dir containing best_model.pt & transforms.json')
    p.add_argument('--input-npy', type=str, default=None,
                        help='Path to .npy of X, shape (N,F) or (N,7,121) or (7,121)')
    p.add_argument('--input-h5', type=str, default=None,
                        help='Path to .h5 with dataset X')
    p.add_argument('--index', type=int, default=None,
                        help='When --input-h5 has X with ndim>=3, pick a single sample by index')
    p.add_argument('--save-csv', type=str, default=None,
                        help='Optional path to save prediction as CSV')
    # proxy model inference
    p.add_argument('--infer-proxy-run', type=str, default=None,
                        help='Run dir containing proxy_g.ts/pt & transforms.json')
    p.add_argument('--proxy-input-h5', type=str, default=None,
                        help='H5 file with Y (and optional X) for proxy inference')
    p.add_argument('--proxy-index', type=int, default=None,
                        help='Index for --proxy-input-h5 when selecting single sample')
    p.add_argument('--save-xhat-npy', type=str, default=None,
                        help='Save predicted X (physical) as .npy with shape (N,7,121)')
    # weak L_sup and strong L_cyc
    p.add_argument('--sup-weight', type=float, default=0.05, help='weight for supervised term L_sup')
    p.add_argument('--prior-l2', type=float, default=1e-3, help='L2 prior on y_norm')
    p.add_argument('--prior-bound', type=float, default=1e-3, help='soft bound prior on physical Y')
    p.add_argument('--prior-bound-margin', type=float, default=0.0, help='margin for bound prior')
    p.add_argument('--es-metric', type=str, default='val_cyc_meas',
                choices=['val_total','val_sup','val_cyc_sim','val_cyc_meas'])
    p.add_argument('--es-min-delta', type=float, default=1e-6)
    # === PATCH A: CLI for trust loss (L_trust) ===
    p.add_argument('--trust-alpha', type=float, default=0.0,
                help='Weight of L_trust (0 disables).')
    p.add_argument('--trust-tau', type=float, default=1.6,
                help='kNN distance threshold (z-space) for L_trust.')
    p.add_argument('--trust-ref-max', type=int, default=20000,
                help='Max rows from proxy Y_train_norm used as trust reference.')
    p.add_argument('--trust-ref-batch', type=int, default=4096,
                help='Per-batch cap of reference rows sampled for cdist speed.')
    p.add_argument('--tail-optmize',type=bool, default=False, 
                   help='Whether to optimize the tail of the cyc_sim')
    # === PATCH-A1: L_trust_meas ===
    p.add_argument('--trust-alpha-meas', type=float, default=0.05,
                help='Weight of L_trust on MEAS branch (0 disables).')
    p.add_argument('--cyc-meas-knn-weight', action='store_true',
                help='If set, downweight L_cyc(meas) by kNN distance.')
    p.add_argument('--cyc-meas-knn-gamma', type=float, default=0.5,
                help='Gamma for kNN weight: w=exp(-gamma*max(0,d-tau)).')
    # === diagnostics ===
    p.add_argument('--diag', action='store_true', help='Enable 3-probe diagnostics and export CSV')
    p.add_argument('--diag-max-samples', type=int, default=256, help='Max samples for Jacobian SVD (per split)')
    p.add_argument('--diag-knn-k', type=int, default=8, help='k for KNN distance (mean of k-NN)')




    args = p.parse_args()
    hidden = tuple(int(x) for x in args.hidden.split(','))
    proxy_hidden = tuple(int(x) for x in args.proxy_hidden.split(','))

    cfg = TrainConfig(
        data=args.data,
        outdir=args.outdir,
        seed=args.seed,
        test_split=args.test_split,
        val_split=args.val_split,
        max_epochs=args.max_epochs,
        onecycle_epochs=args.onecycle_epochs if args.onecycle_epochs != 0 else args.max_epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        weight_decay=args.weight_decay,
        dropout=args.dropout,
        hidden=hidden,
        patience=args.patience,
        num_workers=args.num_workers,
        compile=args.compile,
        use_onecycle=(not args.no_onecycle),
        aug_noise_std=args.aug_noise_std,
        aug_prob=args.aug_prob,
        multihead=(not args.no_multihead),
        uncertainty_weighting=(not args.no_uncertainty),
        enforce_bounds=(not args.no_bounds),
        meas_h5=args.meas_h5,
        lambda_cyc_sim=args.lambda_cyc_sim,
        lambda_cyc_meas=args.lambda_cyc_meas,
        cyc_warmup_epochs=args.cyc_warmup_epochs,
        proxy_run=args.proxy_run,
        auto_train_proxy=args.auto_train_proxy or (args.lambda_cyc_sim>0 or args.lambda_cyc_meas>0),
        proxy_hidden=proxy_hidden,
        proxy_activation=args.proxy_activation,
        proxy_norm=args.proxy_norm,
        proxy_epochs=args.proxy_epochs,
        proxy_lr=args.proxy_lr,
        proxy_wd=args.proxy_wd,
        proxy_beta=args.proxy_beta,
        proxy_seed=args.proxy_seed,
        proxy_patience=args.proxy_patience,   
        proxy_min_delta=args.proxy_min_delta,
        proxy_batch_size=args.proxy_batch_size,
        train_proxy_only=args.train_proxy_only, 
        finetune_from=args.finetune_from,
        ft_use_prev_transforms=args.ft_use_prev_transforms,
        ft_freeze_trunk=args.ft_freeze_trunk,
        ft_reset_head=args.ft_reset_head,
        ft_allow_arch_change=args.ft_allow_arch_change,
        ft_strict=(not args.ft_non_strict),
        sup_weight=args.sup_weight,
        prior_l2=args.prior_l2,
        prior_bound=args.prior_bound,
        prior_bound_margin=args.prior_bound_margin,
        es_metric=args.es_metric,
        es_min_delta=args.es_min_delta,
        trust_alpha = args.trust_alpha,
        trust_tau = args.trust_tau,
        trust_ref_max = args.trust_ref_max,
        trust_ref_batch = args.trust_ref_batch,
        trust_alpha_meas=args.trust_alpha_meas,
        cyc_meas_knn_weight=bool(args.cyc_meas_knn_weight),
        cyc_meas_knn_gamma=args.cyc_meas_knn_gamma,
    )

    diag_cfg = {
    'enable': bool(args.diag),
    'max_samples': int(args.diag_max_samples),
    'knn_k': int(args.diag_knn_k),
    }

    return cfg,args,diag_cfg


def main():
    cfg,args,diag_cfg = parse_args()
    set_seed(cfg.seed)

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

    if cfg.data is None:
        raise SystemExit('--data is required')
    
    if cfg.train_proxy_only:
        res = run_proxy_only(cfg, device)
        print(json.dumps(res, indent=2, default=float))
        return

    if getattr(args, 'infer_proxy_run', None):
        infer_proxy_cli(args, device)
        return


    if getattr(args, 'infer_run', None):
        infer_cli(args, device)
        return


    res = run_once(cfg, diag_cfg, device)
    print(json.dumps(res, indent=2, default=float))


if __name__ == '__main__':
    main()
