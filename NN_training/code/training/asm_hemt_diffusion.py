#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import csv
import json
import math
import time
import argparse
from dataclasses import dataclass, asdict
from typing import List, Tuple, Dict, Optional, Literal

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
from collections import defaultdict

# ============================
# 0) for safe running
# ============================

def add_hparams_safe(base_writer: SummaryWriter, run_dir: str, hparams: dict, metrics: dict):
    try:
        base_writer.add_hparams(hparams, metrics)
        return
    except Exception as e:
        print(f"[warn] add_hparams failed on base writer: {e}. Falling back to dedicated hparams dir.")

    hp_dir = os.path.join(run_dir, "hparams")
    os.makedirs(hp_dir, exist_ok=True)
    try:
        with SummaryWriter(log_dir=hp_dir) as w_hp:
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
# 0.1) normalization term calculation  (FIXED)
# ============================
def NormCalc_prior_bnd(device, y_tf, y_hat_32, PARAM_RANGE,
                       prior_bound, prior_bound_margin,
                       per_sample_ena: bool = False):
    if prior_bound <= 0.0:
        return torch.zeros((), device=device) if not per_sample_ena else torch.zeros(y_hat_32.size(0), device=device)

    # 物理域
    y_phys = y_tf.inverse(y_hat_32.to(torch.float32))
    names, dev = y_tf.names, device
    lo = torch.tensor([PARAM_RANGE[n][0] for n in names], device=dev, dtype=torch.float32)
    hi = torch.tensor([PARAM_RANGE[n][1] for n in names], device=dev, dtype=torch.float32)
    width = (hi - lo).clamp_min(1e-12)
    log_mask = y_tf.log_mask.to(dev)

    if per_sample_ena:
        B = y_hat_32.size(0)
        bound_lin = torch.zeros(B, device=dev)
        bound_log = torch.zeros(B, device=dev)
    else:
        bound_lin = torch.zeros((), device=dev)
        bound_log = torch.zeros((), device=dev)

    # 线性量纲
    if (~log_mask).any():
        y_lin = y_phys[:, ~log_mask]
        lo_lin, hi_lin, w_lin = lo[~log_mask], hi[~log_mask], width[~log_mask]
        over_hi = softplus0((y_lin - (hi_lin + prior_bound_margin * w_lin)) / w_lin, beta=2.0)
        over_lo = softplus0(((lo_lin - prior_bound_margin * w_lin) - y_lin) / w_lin, beta=2.0)
        term = (over_hi + over_lo).mean(dim=1)
        bound_lin = term if per_sample_ena else term.mean()

    # log10 量纲
    if log_mask.any():
        y_log = torch.log10(y_phys[:, log_mask].clamp_min(1e-12))
        lo_log = torch.log10(lo[log_mask].clamp_min(1e-12))
        hi_log = torch.log10(hi[log_mask].clamp_min(1e-12))
        w_log  = (hi_log - lo_log).clamp_min(1e-6)
        over_hi = softplus0((y_log - (hi_log + prior_bound_margin * w_log)) / w_log, beta=2.0)
        over_lo = softplus0(((lo_log - prior_bound_margin * w_log) - y_log) / w_log, beta=2.0)
        term = (over_hi + over_lo).mean(dim=1)
        bound_log = term if per_sample_ena else term.mean()

    return bound_lin + bound_log


def NormCalc_cyc(device, proxy_g, lambda_cyc, y_tf, y_tf_proxy, y_hat_32,
                        x, x_mu_c, x_std_c, x_mu_p, x_std_p, y_idx_c_from_p, cyc_crit):
    cyc = torch.tensor(0.0, device=device)
    xhat_curr_std = x
    if (proxy_g is not None and lambda_cyc > 0.0):
        y_phys = y_tf.inverse(y_hat_32)
        if y_idx_c_from_p is not None: y_phys = y_phys.index_select(1, y_idx_c_from_p)
        y_proxy_norm = y_tf_proxy.transform(y_phys)
        xhat_proxy_std = proxy_g(y_proxy_norm)
        xhat_phys = xhat_proxy_std * x_std_p + x_mu_p
        xhat_curr_std = (xhat_phys - x_mu_c) / x_std_c
        cyc = cyc_crit(xhat_curr_std, x)
    return cyc, xhat_curr_std

# ============================
# 0.2) Diagnostic processing
# ============================
def diag_processing(model, proxy_g, device, 
                    domain_label:str, 
                    diag_rows:List[Dict], diag_count:int, diag_max:int, diag_k:int,
                    x, x_hat_std_sim_prior, x_mu_p, x_std_p, x_mu_c, x_std_c,
                    y, y_hat_prior, y_hat_prior_32, y_tf, y_tf_proxy, y_idx_c_from_p, yref_proxy_norm,
                    prior_bound, prior_bound_margin, PARAM_RANGE,
                    proxy_floor_all: Optional[List[float]] = None):
    B = x.size(0)
    # Diag Module [1] :: cyc_sim using SmoothL1 per-sample 
    beta = 0.02
    diff = (x_hat_std_sim_prior - x).reshape(B, -1)
    absd = diff.abs()
    cyc_sim_ps = torch.where(absd < beta, 0.5 * (diff**2) / beta, absd - 0.5*beta).mean(dim=1)

    # Diag Module [2] :: sup loss per-sample (simplified SmoothL1 version)
    if domain_label == 'sim':
        criterion = CriterionWrapper(model, use_uncertainty=getattr(model, 'use_uncertainty', False))
        per_elem = criterion(y_hat_prior, y, return_per_elem=True)    # [B, D]
        sup_ps = per_elem.mean(dim=1)

    # Diag Module [3] :: prior_l2 per-sample 
    prior_l2_ps = y_hat_prior.pow(2).mean(dim=1)

    # Diag Module [4] :: prior_bnd per-sample 
    prior_bnd_ps = NormCalc_prior_bnd(device, y_tf, y_hat_prior_32, PARAM_RANGE, prior_bound, prior_bound_margin, per_sample_ena=True)

    # Diag Module [5] :: proxy floor per-sample
    if domain_label == 'sim':
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
        # record in global list
        if proxy_floor_all is not None:
            proxy_floor_all.extend(proxy_floor_ps.detach().cpu().tolist())

    # Diag Module [6] :: KNN distance to proxy's Y_train_norm (if provided)
    if (yref_proxy_norm is not None):
        y_phys_m = y_tf.inverse(y_hat_prior_32)
        y_proxy_norm = y_tf_proxy.transform(y_tf.inverse(y.to(torch.float32))) if domain_label == 'sim' else y_tf_proxy.transform(
                (y_phys_m.index_select(1, y_idx_c_from_p) if y_idx_c_from_p is not None else y_phys_m)
                )
        dists = torch.cdist(y_proxy_norm, yref_proxy_norm.to(y_proxy_norm.device), p=2) 
        knn_min = dists.min(dim=1).values
        if diag_k > 1:
            knn_vals, _ = dists.topk(k=min(diag_k, dists.size(1)), largest=False, dim=1)
            knn_mean_k = knn_vals.mean(dim=1)
        else:
            knn_mean_k = knn_min
    else:
        knn_min = torch.full((B,), float('nan'), device=x.device)
        knn_mean_k = torch.full((B,), float('nan'), device=x.device)
    
    # Diag Module [7] :: Jacobian spectral norm of proxy g at y_proxy_norm (subsampled)
    if domain_label == 'sim':
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
    
    # record diag rows
    with torch.no_grad():
        for i in range(B):
            diag_rows.append({
                'domain': domain_label,
                'sup_ps': float(sup_ps[i]) if domain_label == 'sim' else float('nan'),
                'prior_l2_ps': float(prior_l2_ps[i]),
                'prior_bnd_ps': float(prior_bnd_ps[i]),
                'cyc_sim_ps': float(cyc_sim_ps[i]),
                'knn_min': float(knn_min[i]),
                'knn_mean_k': float(knn_mean_k[i]),
                'jac_sigma_max': float(jac_sig[i]) if domain_label == 'sim' else float('nan'),
                'proxy_floor_ps': float(proxy_floor_ps[i]) if 'proxy_floor_ps' in locals() else float('nan'),
            })

    return diag_rows, diag_count
# ============================
# 1) Utils & RNG isolation
# ============================

def noisy_condition(x: torch.Tensor,
                    noise_std: float,
                    noise_clip: float = 3.0) -> torch.Tensor:
    """
    对输入x加零均值高斯噪声，幅度= noise_std（以标准化域为尺度）。
    noise_clip 只是为了防止极端大跳，保证稳定训练。
    """
    if noise_std <= 0.0:
        return x
    # 生成噪声
    eps = torch.randn_like(x) * noise_std
    if noise_clip is not None and noise_clip > 0:
        eps = torch.clamp(eps, -noise_clip * noise_std, noise_clip * noise_std)
    return x + eps


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

def softplus0(t: torch.Tensor, beta: float = 2.0) -> torch.Tensor:
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
        y_norm = y_norm.to(torch.float32)
        mean = self.mean.to(y_norm.device, dtype=torch.float32)
        std  = self.std.to(y_norm.device,  dtype=torch.float32)

        y_t = y_norm * std + mean
        mask = self.log_mask.to(y_norm.device)

        if mask.any():
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

class _MLPBlock(nn.Module):
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
    
class DiffusionSchedule:
    """
    Simple beta schedule + helper to get alpha_bar_t etc.
    We keep T fixed (e.g. 1000) and precompute coefficients on device.
    """
    def __init__(self, num_steps: int = 1000, beta_start: float = 1e-4, beta_end: float = 0.02, device='cpu'):
        # linear beta schedule; you can swap to cosine later
        betas = torch.linspace(beta_start, beta_end, num_steps, dtype=torch.float32)
        alphas = 1.0 - betas
        alpha_bar = torch.cumprod(alphas, dim=0)

        self.num_steps = num_steps
        self.register(betas, alphas, alpha_bar, device)

    def register(self, betas, alphas, alpha_bar, device):
        self.betas = betas.to(device)
        self.alphas = alphas.to(device)
        self.alpha_bar = alpha_bar.to(device)

    def to(self, device):
        self.register(self.betas, self.alphas, self.alpha_bar, device)
        return self

    def sample_timesteps(self, bsz: int, device) -> torch.Tensor:
        # Uniform t in [1, T-1]
        return torch.randint(low=1, high=self.num_steps, size=(bsz,), device=device)

    def get_coeffs(self, t: torch.Tensor):
        """
        t: [B] with values in [0, T-1]
        returns sqrt_ab_t, sqrt_one_minus_ab_t
        where alpha_bar_t = prod_{s<=t} alpha_s
        """
        ab = self.alpha_bar[t]  # [B]
        sqrt_ab = torch.sqrt(ab)
        sqrt_1m = torch.sqrt(1.0 - ab)
        return sqrt_ab, sqrt_1m


class FiLM(nn.Module):
    """
    Simple FiLM-style conditioning: given cond_feat (from x),
    produce per-layer (gamma, beta) to modulate hidden activations.
    """
    def __init__(self, in_dim: int, hidden_dim: int):
        super().__init__()
        self.to_scale = nn.Linear(in_dim, hidden_dim)
        self.to_shift = nn.Linear(in_dim, hidden_dim)

    def forward(self, h, cond_vec):
        gamma = self.to_scale(cond_vec)
        beta  = self.to_shift(cond_vec)
        return h * (1 + gamma) + beta


class CondDenoiserMLP(nn.Module):
    """
    ε_theta(y_t, t, x_cond) -> predict noise on y_t.
    - y_t: noised target (B, y_dim)
    - t_emb: timestep embedding (B, d_t)
    - x_cond: condition from x (B, x_dim)
    We'll fuse [y_t, t_emb] first, then inject x_cond via FiLM at each block.
    """
    def __init__(self, x_dim: int, y_dim: int, hidden: List[int], t_embed_dim: int = 64, dropout: float = 0.1):
        super().__init__()
        self.y_dim = y_dim
        self.x_dim = x_dim

        # timestep embedding
        self.t_embed = nn.Sequential(
            nn.Linear(1, t_embed_dim),
            nn.SiLU(),
            nn.Linear(t_embed_dim, t_embed_dim),
            nn.SiLU()
        )

        layers = []
        films = []
        in_dim = y_dim + t_embed_dim
        for h in hidden:
            layers.append(nn.Sequential(
                nn.Linear(in_dim, h),
                nn.LayerNorm(h),
                nn.GELU(),
                nn.Dropout(dropout),
            ))
            films.append(FiLM(x_dim, h))
            in_dim = h
        self.layers = nn.ModuleList(layers)
        self.films  = nn.ModuleList(films)

        self.out = nn.Linear(in_dim, y_dim)

    def forward(self, y_t, t, x_cond):
        # t: (B,) int timesteps -> embed
        t = t.view(-1,1).to(y_t.dtype)
        t_feat = self.t_embed(t)  # (B, t_embed_dim)

        h = torch.cat([y_t, t_feat], dim=1)  # (B, y_dim+t_embed_dim)

        for block, film in zip(self.layers, self.films):
            h = block(h)
            h = film(h, x_cond)  # FiLM inject condition from x

        eps_pred = self.out(h)
        return eps_pred


class CondDiffusion(nn.Module):
    """
    Wrapper that holds:
    - denoiser: predicts eps
    - schedule: forward diffusion schedule

    Exposes:
      forward_train(x, y)  -> loss_terms dict
      sample(x, n_steps)   -> y_hat  (DDIM-like simple sampler)
    """
    def __init__(self, x_dim: int, y_dim: int, hidden: List[int], dropout: float = 0.1,
                 num_steps: int = 1000, beta_start: float = 1e-4, beta_end: float = 2e-2, device='cpu'):
        super().__init__()
        self.schedule = DiffusionSchedule(num_steps=num_steps,
                                          beta_start=beta_start,
                                          beta_end=beta_end,
                                          device=device)
        self.denoiser = CondDenoiserMLP(
            x_dim=x_dim,
            y_dim=y_dim,
            hidden=hidden,
            t_embed_dim=64,
            dropout=dropout
        )

    def to(self, device):
        super().to(device)
        self.schedule.to(device)
        return self

    def forward_train(self, x, y):
        """
        x: (B, x_dim) condition
        y: (B, y_dim) clean target (our y_norm)
        returns:
           eps_pred, eps_gt, t, y_t, y_pred0
        """
        B = y.size(0)
        device = y.device

        # 1) sample timestep
        t = self.schedule.sample_timesteps(B, device)  # (B,)

        # 2) forward diffuse y -> y_t
        eps = torch.randn_like(y)
        sqrt_ab, sqrt_1m = self.schedule.get_coeffs(t) # (B,)
        sqrt_ab = sqrt_ab.view(-1,1)
        sqrt_1m = sqrt_1m.view(-1,1)
        y_t = sqrt_ab * y + sqrt_1m * eps  # (B, y_dim)

        # 3) predict eps
        eps_pred = self.denoiser(y_t, t, x)

        # 4) also get a current denoised y_pred0 (for cycle/proxy losses etc.)
        # standard DDPM eps->x0 reconstruction:
        # y0_hat = (y_t - sqrt_1m * eps_pred) / sqrt_ab
        y_pred0 = (y_t - sqrt_1m * eps_pred) / (sqrt_ab + 1e-8)

        return {
            't': t,
            'y_t': y_t,
            'eps_gt': eps,
            'eps_pred': eps_pred,
            'y_pred0': y_pred0,
        }

    @torch.no_grad()
    def sample(self, x, num_steps: int = 50, guidance_w: float = 0.0, noise_scale: float = 0.0):
        """
        Stochastic sampler:
        - 保留DDIM式的direct update toward y0_hat
        - 再加一点高斯噪声，避免所有轨迹塌到同一个极小盆地
        noise_scale: 噪声强度，0表示完全确定性
        """
        device = x.device
        B = x.size(0)
        T = self.schedule.num_steps

        # 选一组反推步
        step_ids = torch.linspace(T-1, 1, num_steps, dtype=torch.long, device=device)

        # 从高斯初始化
        y_t = torch.randn(B, self.denoiser.y_dim, device=device)

        for t_idx in step_ids:
            t_batch = t_idx.repeat(B)

            # 预测噪声
            eps_pred = self.denoiser(y_t, t_batch, x)

            # 还原当前估计的无噪版本 y0_hat
            sqrt_ab, sqrt_1m = self.schedule.get_coeffs(t_batch)
            sqrt_ab = sqrt_ab.view(-1,1)
            sqrt_1m = sqrt_1m.view(-1,1)
            y0_hat = (y_t - sqrt_1m * eps_pred) / (sqrt_ab + 1e-8)

            # 目标上一时刻 t_prev 的系数
            if t_idx > 1:
                t_prev = (t_idx - 1).long()
            else:
                t_prev = torch.zeros_like(t_idx)

            sqrt_ab_prev, _ = self.schedule.get_coeffs(t_prev)
            sqrt_ab_prev = sqrt_ab_prev.view(-1,1)

            # 预测下一个状态的均值项（DDIM core）
            mean_tprev = sqrt_ab_prev * y0_hat

            # 额外噪声项：给多解一点空间
            if noise_scale > 0.0 and t_idx > 1:
                # 用 eps_pred 来刻画方向的尺度（可选），或者直接高斯
                sigma_t = noise_scale * torch.sqrt(torch.clamp(1.0 - sqrt_ab_prev**2, min=1e-8))
                noise = torch.randn_like(y_t)
                y_t = mean_tprev + sigma_t * noise
            else:
                # 几乎确定性的路径
                y_t = mean_tprev

        return y_t  # 这对应 t=0 近似解



def _make_act(name: str) -> nn.Module:
    name = (name or 'gelu').lower()
    if name == 'relu': return nn.ReLU()
    if name in ('silu', 'swish'): return nn.SiLU()
    return nn.GELU()

def _make_norm(name: str, dim: int) -> nn.Module:
    name = (name or 'layernorm').lower()
    if name in ('batchnorm', 'bn'): return nn.BatchNorm1d(dim)
    if name in ('none', 'identity'): return nn.Identity()
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
        
@dataclass
class TrainConfig:
    # Core
    data: str
    outdir: str = 'runs'
    seed: int = 42
    test_split: float = 0.15
    val_split: float = 0.15
    max_epochs: int = 300
    batch_size: int = 256
    lr: float = 1e-3
    weight_decay: float = 1e-4
    patience: int = 30
    num_workers: int = 0
    compile: bool = False
    
    # CVAE architecture
    hidden: Tuple[int, ...] = (512, 256)
    latent_dim: int = 32
    dropout: float = 0.1

    # Training strategy
    onecycle_epochs: int = 300
    use_onecycle: bool = True
    aug_noise_std: float = 0.015
    aug_prob: float = 0.5
    enforce_bounds: bool = True
    kl_beta: float = 0.1 # Weight for the KL divergence term

    # Consistency & proxy
    meas_h5: Optional[str] = None
    lambda_cyc_sim: float = 0.0
    lambda_cyc_meas: float = 0.0
    cyc_warmup_epochs: int = 15
    proxy_run: Optional[str] = None
    auto_train_proxy: bool = True
    
    # Proxy hyperparams
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
    train_proxy_only: bool = False
    
    # Finetune (Note: Finetuning a CVAE arch from an MLP is complex and not recommended without careful thought)
    finetune_from: Optional[str] = None
    
    # Priors and Loss weights
    sup_weight: float = 1.0 # Reconstruction loss weight
    prior_l2: float = 1e-3
    prior_bound: float = 1e-3
    prior_bound_margin: float = 0.0
    es_metric: str = 'val_cyc_meas'
    es_min_delta: float = 1e-6
    
    # L_trust
    trust_alpha : float = 0.0
    trust_tau : float = 1.6
    trust_ref_max : int = 20000
    trust_ref_batch : int = 4096
    
    # L_trust_meas
    trust_alpha_meas : float = 0.0
    cyc_meas_knn_weight : bool = False
    cyc_meas_knn_gamma : float = 0.5

    # Inference
    num_samples: int = 10
    sample_mode: str = 'rand'

    # Extend
    z_sample_mode: str = 'rand'
    cond_noise_std: float = 0.02   # 训练时往x里打的噪声标准差(标准化域)
    cond_noise_clip: float = 3.0   # 噪声截断
    prior_reg_scale: float = 0.5   # 先验相关loss的全局缩放
    cyc_reg_scale: float   = 0.5   # cycle一致性相关loss全局缩放


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
        X = f['X'][...]
        Y = f['Y'][...]
    N = X.shape[0]
    X = X.reshape(N, -1).astype(np.float32)
    Y = Y.reshape(N, len(PARAM_NAMES)).astype(np.float32)

    tr_idx, va_idx, te_idx = split_indices(N, cfg.test_split, cfg.val_split, cfg.seed)

    x_scaler = XStandardizer(); x_scaler.fit(X[tr_idx])
    X_tr = x_scaler.transform(X[tr_idx]); X_va = x_scaler.transform(X[va_idx]); X_te = x_scaler.transform(X[te_idx])
    log_mask_np = choose_log_mask(PARAM_RANGE, PARAM_NAMES)
    y_tf = YTransform(PARAM_NAMES, log_mask_np)
    y_tf.fit(torch.from_numpy(Y[tr_idx]))

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

    local_seed = seed if seed is not None else _seed_from_proxy_cfg(
        hidden, activation, norm, max_epochs, lr, weight_decay, beta
    )

    with scoped_rng(local_seed):
        in_dim = Y_tr_norm.shape[1]
        out_dim = X_tr_std.shape[1]
        model = ProxyMLP(in_dim, out_dim, list(hidden), activation=activation, norm=norm).to(device)
        opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

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

        ac_kwargs = (
            dict(device_type='cuda', dtype=torch.float16, enabled=True)
            if device.type == 'cuda' else
            dict(device_type='cpu', dtype=torch.bfloat16, enabled=False)
        )

        for ep in range(1, max_epochs + 1):
            model.train()
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
def calculate_kl_divergence(mu_post, logvar_post, mu_prior, logvar_prior) -> torch.Tensor:
    """KL(P_post || P_prior) for diagonal Gaussians."""
    var_post = torch.exp(logvar_post)
    var_prior = torch.exp(logvar_prior)

    # Add a small epsilon to variance to avoid division by zero or log(0)
    var_prior = var_prior + 1e-8

    kl_div = 0.5 * torch.sum(
        logvar_prior - logvar_post +
        (var_post + (mu_post - mu_prior).pow(2)) / var_prior - 1,
        dim=1
    )
    return kl_div.mean()


def train_one_epoch(model, loader, optimizer, scaler, device,
                    scheduler=None,
                    current_epoch: int = 1,
                    onecycle_epochs: int = 120,
                    kl_beta: float = 0.0,  # KL不再使用，保持接口
                    y_tf: Optional[YTransform]=None,
                    proxy_g: Optional[nn.Module]=None,
                    lambda_cyc_sim: float=0.0,
                    meas_loader: Optional[DataLoader]=None,
                    lambda_cyc_meas: float=0.0,
                    y_tf_proxy: Optional[YTransform]=None,
                    x_mu_c: Optional[torch.Tensor]=None, x_std_c: Optional[torch.Tensor]=None,
                    x_mu_p: Optional[torch.Tensor]=None, x_std_p: Optional[torch.Tensor]=None,
                    y_idx_c_from_p: Optional[torch.Tensor]=None,
                    sup_weight: float = 1.0,           # we'll treat this as weight on eps MSE? keep =1
                    prior_l2: float = 1e-3,
                    prior_bound: float = 1e-3,
                    prior_bound_margin: float = 0.0,
                    trust_alpha: float = 0.0, trust_tau: float = 1.6,
                    yref_proxy_norm: Optional[torch.Tensor] = None,
                    trust_ref_batch: int = 4096,
                    trust_alpha_meas: float = 0.0,
                    cyc_meas_knn_weight: bool = False,
                    cyc_meas_knn_gamma: float = 0.5,
                    z_sample_mode: Literal['mean','rand'] = 'mean',  # unused but kept for interface
                    cond_noise_std: float = 0.02,
                    cond_noise_clip: float = 3.0,
                    prior_reg_scale: float = 1.0,
                    cyc_reg_scale: float = 1.0,
                    ):
    model.train()

    total_loss = total_eps = total_cyc_sim = total_cyc_meas = 0.0
    total_prior_l2 = total_prior_bnd = 0.0
    n = 0

    mse_eps = nn.MSELoss(reduction='mean')
    cyc_crit = nn.SmoothL1Loss(beta=0.02, reduction='mean')
    meas_iter = iter(meas_loader) if meas_loader is not None else None

    for x, y in loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)

        x_noisy = noisy_condition(
            x,
            noise_std=cond_noise_std,
            noise_clip=cond_noise_clip
        )

        optimizer.zero_grad(set_to_none=True)

        # --- forward + main diffusion loss ---
        with torch.autocast(device_type='cuda' if device.type=='cuda' else 'cpu',
                            dtype=torch.float16 if device.type=='cuda' else torch.bfloat16):
            out = model.forward_train(x_noisy, y)
            eps_pred = out['eps_pred']      # (B, y_dim)
            eps_gt   = out['eps_gt']        # (B, y_dim)
            y_pred0  = out['y_pred0']       # (B, y_dim)
            t_batch  = out['t']             # (B,)  <-- already returned by forward_train

            # 主loss: 噪声回归，始终算
            loss_eps = mse_eps(eps_pred, eps_gt)

            # ---------------------------
            # 只在小t样本上施加物理/先验约束
            # ---------------------------
            # 假设我们把“几乎去噪完成”的区间定义为 t < t_phys_thresh
            t_phys_thresh = 50  # <<< 重要超参，可调成 cfg.t_phys_thresh later
            mask = (t_batch < t_phys_thresh)

        # --- regularizers (float32 for stability) ---
        with torch.autocast(device_type='cuda' if device.type=='cuda' else 'cpu', enabled=False):
            if mask.any():
                # 只取这些样本
                y_hat_32_phys = y_pred0[mask].to(torch.float32)
                x_phys        = x[mask]

                # --------- prior L2 / bound ----------
                prior_term_l2 = y_hat_32_phys.pow(2).mean() if prior_l2 > 0.0 else y_hat_32_phys.new_tensor(0.0)
                prior_term_bnd = NormCalc_prior_bnd(
                    device, y_tf, y_hat_32_phys,
                    PARAM_RANGE,
                    prior_bound, prior_bound_margin
                )

                # --------- cyc_sim (sim batch) ----------
                cyc_sim = y_hat_32_phys.new_tensor(0.0)
                if (proxy_g is not None and lambda_cyc_sim > 0.0):
                    cyc_sim, _ = NormCalc_cyc(
                        device, proxy_g, lambda_cyc_sim,
                        y_tf, y_tf_proxy, y_hat_32_phys,
                        x_phys, x_mu_c, x_std_c, x_mu_p, x_std_p,
                        y_idx_c_from_p,
                        cyc_crit
                    )

                # --------- cyc_meas (meas batch) ----------
                cyc_meas = y_hat_32_phys.new_tensor(0.0)
                if (proxy_g is not None and meas_iter is not None and lambda_cyc_meas > 0.0):
                    try:
                        xm = next(meas_iter)
                    except StopIteration:
                        meas_iter = iter(meas_loader); xm = next(meas_iter)
                    xm = xm.to(device, non_blocking=True)

                    # 用当前模型对测量样本反演（我们只需要 y_pred0）
                    with torch.no_grad():
                        out_meas = model.forward_train(xm, torch.zeros_like(y[:xm.size(0)]))
                        y_hat_meas = out_meas['y_pred0'].to(torch.float32)

                    ym_phys = y_tf.inverse(y_hat_meas)
                    ym_phys_proj = (ym_phys.index_select(1, y_idx_c_from_p)
                                    if y_idx_c_from_p is not None else ym_phys)
                    ym_proxy_norm = y_tf_proxy.transform(ym_phys_proj)

                    xmh_proxy_std = proxy_g(ym_proxy_norm)
                    xmh_phys = xmh_proxy_std * x_std_p + x_mu_p
                    xmh_curr_std = (xmh_phys - x_mu_c) / x_std_c

                    beta = 0.02
                    diff = (xmh_curr_std - xm).reshape(xm.size(0), -1)
                    absd = diff.abs()
                    cyc_ps_m = torch.where(
                        absd < beta,
                        0.5 * (diff**2) / beta,
                        absd - 0.5*beta
                    ).mean(dim=1)

                    # knn weighting / trust_alpha_meas 逻辑保持不变
                    dmin_m = None
                    if ((cyc_meas_knn_weight or (trust_alpha_meas > 0.0)) and (yref_proxy_norm is not None)):
                        Nref = yref_proxy_norm.shape[0]
                        idx = torch.randint(0, Nref, (min(Nref, trust_ref_batch),), device=yref_proxy_norm.device)
                        yref_sub = yref_proxy_norm.index_select(0, idx)
                        dists_m = torch.cdist(
                            y_tf_proxy.transform(ym_phys_proj),
                            yref_sub,
                            p=2
                        )
                        dmin_m = dists_m.min(dim=1).values

                    if cyc_meas_knn_weight and dmin_m is not None:
                        w_knn = torch.clamp(dmin_m / max(1e-6, trust_tau),
                                            min=0.0, max=4.0).pow(cyc_meas_knn_gamma).detach()
                        cyc_meas = (w_knn * cyc_ps_m).mean()
                    else:
                        cyc_meas = cyc_ps_m.mean()

                    if (trust_alpha_meas > 0.0) and (dmin_m is not None):
                        trust_loss_m = torch.clamp_min(dmin_m - trust_tau, 0.0).pow(2).mean()
                        cyc_meas = cyc_meas + trust_alpha_meas * trust_loss_m

                # --------- trust_alpha (sim batch) ----------
                if (trust_alpha > 0.0 and yref_proxy_norm is not None):
                    y_phys = y_tf.inverse(y_hat_32_phys)
                    y_phys_p = y_phys.index_select(1, y_idx_c_from_p) if y_idx_c_from_p is not None else y_phys
                    y_proxy_norm = y_tf_proxy.transform(y_phys_p)

                    Nref = yref_proxy_norm.shape[0]
                    idx = torch.randint(0, Nref, (min(Nref, trust_ref_batch),), device=yref_proxy_norm.device)
                    yref_sub = yref_proxy_norm.index_select(0, idx)

                    dists = torch.cdist(y_proxy_norm, yref_sub, p=2)
                    dmin = dists.min(dim=1).values
                    trust_loss = torch.clamp_min(dmin - trust_tau, 0.0).pow(2).mean()
                else:
                    trust_loss = y_hat_32_phys.new_tensor(0.0)

                prior_block = (
                    prior_l2 * prior_term_l2 +
                    prior_bound * prior_term_bnd
                )

                cyc_block = (
                    lambda_cyc_sim * cyc_sim +
                    lambda_cyc_meas * cyc_meas +
                    trust_alpha * trust_loss
                )

                loss_reg = prior_reg_scale * prior_block + cyc_reg_scale * cyc_block

                # loss_reg = (prior_l2 * prior_term_l2
                #             + prior_bound * prior_term_bnd
                #             + lambda_cyc_sim * cyc_sim
                #             + lambda_cyc_meas * cyc_meas
                #             + trust_alpha * trust_loss)
            else:
                # 如果本批次里没有小t样本，就不加正则
                loss_reg = loss_eps.new_zeros(())

        total_step_loss = sup_weight * loss_eps + loss_reg

        # backward / step
        scaler.scale(total_step_loss).backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=2.0)
        scaler.step(optimizer)
        scaler.update()
        if (scheduler is not None) and (current_epoch <= onecycle_epochs):
            scheduler.step()

        bs = x.size(0)
        n += bs
        total_loss     += total_step_loss.item() * bs
        total_eps      += loss_eps.item() * bs
        total_cyc_sim  += float(cyc_sim) * bs
        total_cyc_meas += float(cyc_meas) * bs
        total_prior_l2 += float(prior_term_l2) * bs
        total_prior_bnd+= float(prior_term_bnd)* bs

    return {
        'total': total_loss / max(1,n),
        'eps_mse': total_eps / max(1,n),
        'cyc_sim': total_cyc_sim / max(1,n),
        'cyc_meas': total_cyc_meas / max(1,n),
        'prior_l2': total_prior_l2 / max(1,n),
        'prior_bnd': total_prior_bnd / max(1,n),
        # keep keys so run_once logging won't break:
        'recon': total_eps / max(1,n),
        'kl': 0.0,
    }


@torch.no_grad()
def evaluate_full(model, loader, device, *,
                  y_tf,
                  proxy_g,
                  lambda_cyc_sim,
                  meas_loader,
                  lambda_cyc_meas,
                  y_tf_proxy,
                  x_mu_c, x_std_c,
                  x_mu_p, x_std_p,
                  y_idx_c_from_p,
                  sup_weight,
                  kl_beta,
                  prior_l2,
                  prior_bound,
                  prior_bound_margin,
                  enforce_bounds=False,
                  diag_cfg=None,
                  yref_proxy_norm=None,
                  diag_outdir=None,
                  diag_tag=None,
                  z_sample_mode='mean'):

    model.eval()
    recon_crit = nn.SmoothL1Loss(beta=0.02, reduction='mean')
    cyc_crit   = nn.SmoothL1Loss(beta=0.02, reduction='mean')

    meter = defaultdict(float)

    # -------------------------
    # (1) 评估有标签的验证集
    # -------------------------
    for x, y in loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)

        # 关键改动：用真正的采样结果而不是 y_pred0
        # 这一步等价于“扩散模型最终给出的参数解”
        y_eval = model.sample(x, num_steps=50)        # (B, y_dim)
        y_eval = y_eval.to(torch.float32)

        # 下面这些和你原来的 evaluate_full 基本一致，但对象换成 y_eval
        sup_prior = recon_crit(y_eval, y)

        prior_l2_term  = y_eval.pow(2).mean()
        prior_bnd_term = NormCalc_prior_bnd(
            device, y_tf, y_eval, PARAM_RANGE,
            prior_bound, prior_bound_margin
        )

        cyc_sim_term, x_hat_std_sim_prior = NormCalc_cyc(
            device, proxy_g, lambda_cyc_sim,
            y_tf, y_tf_proxy, y_eval,
            x, x_mu_c, x_std_c, x_mu_p, x_std_p,
            y_idx_c_from_p,
            cyc_crit
        )

        meter['sup_prior']         += sup_prior.item()         * x.size(0)
        meter['prior_l2_prior']    += prior_l2_term.item()     * x.size(0)
        meter['prior_bnd_prior']   += prior_bnd_term.item()    * x.size(0)
        meter['cyc_sim_prior']     += cyc_sim_term.item()      * x.size(0)
        meter['n']                 += x.size(0)

    # -------------------------
    # (2) 评估无标签测量集 meas_loader
    # -------------------------
    if (proxy_g is not None and meas_loader is not None and lambda_cyc_meas > 0.0):
        for xm in meas_loader:
            xm = xm.to(device, non_blocking=True)

            # 同样用最终采样，而不是 forward_train()
            y_hat_meas = model.sample(xm, num_steps=50)
            y_hat_meas = y_hat_meas.to(torch.float32)

            cyc_meas_term, _ = NormCalc_cyc(
                device, proxy_g, lambda_cyc_meas,
                y_tf, y_tf_proxy, y_hat_meas,
                xm, x_mu_c, x_std_c, x_mu_p, x_std_p,
                y_idx_c_from_p,
                cyc_crit
            )

            meter['cyc_meas'] += cyc_meas_term.item() * xm.size(0)
            meter['n_meas']   += xm.size(0)

    n      = meter['n']
    n_meas = meter.get('n_meas', 0)

    kl_val = 0.0  # no KL in diffusion

    metrics = {
        # 这些 key 名是 run_once() 里早停和日志在用的，我们保持不变
        'val_sup_post':        meter['sup_prior']/n,
        'val_cyc_sim_post':    meter['cyc_sim_prior']/n,
        'val_prior_l2_post':   meter['prior_l2_prior']/n,
        'val_prior_bnd_post':  meter['prior_bnd_prior']/n,
        'val_kl':              kl_val,
        'val_total_post':
            (meter['sup_prior']/n
             + prior_l2    * (meter['prior_l2_prior']/n)
             + prior_bound * (meter['prior_bnd_prior']/n)
             + lambda_cyc_sim  * (meter['cyc_sim_prior']/n)
             + lambda_cyc_meas * ((meter['cyc_meas']/max(1,n_meas)) if n_meas>0 else 0.0)
            ),

        # “prior”这组字段我们也用同一个 y_eval 来填
        'val_sup_prior':       meter['sup_prior']/n,
        'val_cyc_sim_prior':   meter['cyc_sim_prior']/n,
        'val_cyc_meas':        (meter['cyc_meas']/max(1,n_meas)) if n_meas>0 else 0.0,
        'val_prior_l2_prior':  meter['prior_l2_prior']/n,
        'val_prior_bnd_prior': meter['prior_bnd_prior']/n,
        'val_total_prior':
            (meter['sup_prior']/n
             + prior_l2    * (meter['prior_l2_prior']/n)
             + prior_bound * (meter['prior_bnd_prior']/n)
             + lambda_cyc_sim  * (meter['cyc_sim_prior']/n)
             + lambda_cyc_meas * ((meter['cyc_meas']/max(1,n_meas)) if n_meas>0 else 0.0)
            ),
    }

    return metrics


def save_state(outdir: str, x_scaler: XStandardizer, y_tf: YTransform, cfg: TrainConfig):
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

# ============================
# 7) Run once / main loop
# ============================
def run_proxy_only(cfg: TrainConfig, device):
    """Only trains and saves the proxy g(Y->X)."""
    if cfg.data is None: raise SystemExit('--data is required')
    _, _, _, _, _, splits, X_all, Y_all = load_and_prepare(cfg.data, cfg)
    tr_idx, va_idx, _ = splits
    N, x_dim = X_all.shape[0], X_all.shape[1] * X_all.shape[2]
    y_dim = Y_all.shape[1]
    X_flat = X_all.reshape(N, -1).astype(np.float32)

    x_scaler_p = XStandardizer(); x_scaler_p.fit(X_flat[tr_idx])
    X_tr_std_p, X_va_std_p = x_scaler_p.transform(X_flat[tr_idx]), x_scaler_p.transform(X_flat[va_idx])
    
    log_mask_np = choose_log_mask(PARAM_RANGE, PARAM_NAMES)
    y_tf_p = YTransform(PARAM_NAMES, log_mask_np)
    y_tf_p.fit(torch.from_numpy(Y_all[tr_idx]))
    Y_tr_norm_p = y_tf_p.transform(torch.from_numpy(Y_all[tr_idx])).numpy()
    Y_va_norm_p = y_tf_p.transform(torch.from_numpy(Y_all[va_idx])).numpy()
    
    stamp = time.strftime('%Y%m%d-%H%M%S')
    run_dir = os.path.join(cfg.outdir, f'proxy_run_{stamp}')
    os.makedirs(run_dir, exist_ok=True)

    save_state(run_dir, x_scaler_p, y_tf_p, cfg)
    np.save(os.path.join(run_dir, 'proxy_Ytr_norm.npy'), Y_tr_norm_p)
    
    proxy_g, pt_path, ts_path, _ = train_proxy_g(
        X_tr_std_p, Y_tr_norm_p, X_va_std_p, Y_va_norm_p, device, run_dir,
        hidden=cfg.proxy_hidden, activation=cfg.proxy_activation, norm=cfg.proxy_norm,
        max_epochs=cfg.proxy_epochs, lr=cfg.proxy_lr, weight_decay=cfg.proxy_wd, beta=cfg.proxy_beta,
        seed=cfg.proxy_seed, patience=cfg.proxy_patience, min_delta=cfg.proxy_min_delta,
        batch_size=cfg.proxy_batch_size)

    files = {'proxy_g.pt': os.path.basename(pt_path), 'proxy_g.ts': os.path.basename(ts_path)}
    _update_transforms_meta(run_dir, {'proxy': {
        'arch': 'mlp', 'in_dim': y_dim, 'out_dim': x_dim, 'hidden': list(cfg.proxy_hidden),
        'activation': cfg.proxy_activation, 'norm': cfg.proxy_norm, 'format': 'torchscript', 'files': files
    }})
    print(f"[ProxyOnly] Saved to: {run_dir}")
    return {'run_dir': run_dir}


def run_once(cfg: TrainConfig, diag_cfg: dict, device):
    set_seed(cfg.seed)
    train_ds, val_ds, test_ds, x_scaler, y_tf, splits, X_all, Y_all = load_and_prepare(cfg.data, cfg)
    train_loader, val_loader, test_loader = make_loaders(train_ds, val_ds, test_ds, cfg.batch_size, cfg.num_workers)

    x_dim, y_dim = train_ds.x.shape[1], train_ds.y.shape[1]
    
    model = CondDiffusion(
        x_dim=x_dim,
        y_dim=y_dim,
        hidden=list(cfg.hidden),
        dropout=cfg.dropout,
        num_steps=1000,
        beta_start=1e-4,
        beta_end=2e-2,
        device=device
    ).to(device)

    if cfg.compile and hasattr(torch, 'compile'):
        try: model = torch.compile(model)
        except Exception as e: print(f"Torch compile failed: {e}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=cfg.lr, epochs=cfg.onecycle_epochs, steps_per_epoch=len(train_loader)) if cfg.use_onecycle else None
    scaler = torch.amp.GradScaler('cuda', enabled=(device.type=='cuda'))

    stamp = time.strftime('%Y%m%d-%H%M%S')
    run_dir = os.path.join(cfg.outdir, f'cvae_{stamp}')
    writer = SummaryWriter(log_dir=run_dir)

    meas_loader = None
    if cfg.meas_h5:
        meas_loader, _ = make_meas_loader(cfg.meas_h5, x_scaler, cfg.batch_size, cfg.num_workers)

    # Proxy setup
    proxy_g, x_scaler_p, y_tf_p = None, None, None
    if (cfg.lambda_cyc_sim > 0.0 or cfg.lambda_cyc_meas > 0.0):
        if cfg.proxy_run:
            proxy_g, x_scaler_p, y_tf_p, _ = load_proxy_artifacts(cfg.proxy_run, device)
        elif cfg.auto_train_proxy:
            # Re-running proxy training logic here, scoped to this run
            print("[Info] Auto-training proxy model...")
            tr_idx, va_idx, _ = splits
            X_flat = X_all.reshape(X_all.shape[0], -1)
            x_scaler_p = XStandardizer(); x_scaler_p.fit(X_flat[tr_idx])
            X_tr_std_p = x_scaler_p.transform(X_flat[tr_idx]); X_va_std_p = x_scaler_p.transform(X_flat[va_idx])
            y_tf_p = YTransform(PARAM_NAMES, choose_log_mask(PARAM_RANGE, PARAM_NAMES))
            y_tf_p.fit(torch.from_numpy(Y_all[tr_idx]))
            Y_tr_norm_p = y_tf_p.transform(torch.from_numpy(Y_all[tr_idx])).numpy()
            Y_va_norm_p = y_tf_p.transform(torch.from_numpy(Y_all[va_idx])).numpy()
            np.save(os.path.join(run_dir, 'proxy_Ytr_norm.npy'), Y_tr_norm_p)

            proxy_g_eager, pt_path, ts_path, _ = train_proxy_g(
                X_tr_std_p, Y_tr_norm_p, X_va_std_p, Y_va_norm_p, device, run_dir,
                hidden=cfg.proxy_hidden, activation=cfg.proxy_activation, norm=cfg.proxy_norm,
                max_epochs=cfg.proxy_epochs, lr=cfg.proxy_lr, weight_decay=cfg.proxy_wd,
                beta=cfg.proxy_beta, seed=cfg.proxy_seed, patience=cfg.proxy_patience,
                min_delta=cfg.proxy_min_delta, batch_size=cfg.proxy_batch_size)
            
            try:
                proxy_g = torch.jit.load(ts_path, map_location=device).eval()
            except Exception as e:
                print(f"[Warn] Failed to load scripted proxy, using eager: {e}")
                proxy_g = proxy_g_eager
    
    save_state(run_dir, x_scaler, y_tf, cfg)
    if x_scaler_p and y_tf_p:
        _update_transforms_meta(run_dir, {
            'proxy_x_scaler': x_scaler_p.state_dict(),
            'proxy_y_transform': y_tf_p.state_dict(),
        })

    x_mu_c  = torch.tensor(x_scaler.mean, device=device, dtype=torch.float32)
    x_std_c = torch.tensor(x_scaler.std,  device=device, dtype=torch.float32)
    x_mu_p, x_std_p, y_tf_proxy, y_idx_c_from_p = None, None, None, None
    if proxy_g:
        x_mu_p = torch.tensor(x_scaler_p.mean, device=device, dtype=torch.float32)
        x_std_p = torch.tensor(x_scaler_p.std, device=device, dtype=torch.float32)
        y_tf_proxy = y_tf_p
        name2idx_curr = {n: i for i, n in enumerate(y_tf.names)}
        idx_list = [name2idx_curr[n] for n in y_tf_proxy.names]
        y_idx_c_from_p = torch.tensor(idx_list, device=device, dtype=torch.long)

    yref_proxy_norm = None
    if (proxy_g and (cfg.trust_alpha > 0.0 or cfg.trust_alpha_meas > 0.0)):
        probe_path = os.path.join(cfg.proxy_run or run_dir, 'proxy_Ytr_norm.npy')
        if os.path.isfile(probe_path):
            arr = np.load(probe_path).astype(np.float32)
            if arr.shape[0] > cfg.trust_ref_max:
                idx = np.random.choice(arr.shape[0], cfg.trust_ref_max, replace=False)
                arr = arr[idx]
            yref_proxy_norm = torch.from_numpy(arr).to(device)
            print(f"[L_trust] using {arr.shape[0]} ref rows from {probe_path}")
        else:
            print("[L_trust] Warning: proxy_Ytr_norm.npy not found, L_trust disabled.")
            cfg.trust_alpha, cfg.trust_alpha_meas = 0.0, 0.0

    # Training loop
    best_val, no_improve = float('inf'), 0
    best_path = os.path.join(run_dir, 'best_model.pt')
    for epoch in range(1, cfg.max_epochs + 1):
        warm_ratio = min(1.0, epoch / max(1, cfg.cyc_warmup_epochs))

        lam_sim_epoch  = cfg.lambda_cyc_sim  * warm_ratio
        lam_meas_epoch = cfg.lambda_cyc_meas * warm_ratio

        sup_w_epoch = cfg.sup_weight * (1.0 - 0.7 * warm_ratio)
        # 例如 cfg.sup_weight=0.1:
        # epoch早期 warm_ratio小 -> sup_w_epoch ~0.1
        # 后期 warm_ratio→1   -> sup_w_epoch ~0.03
        reg_scale_epoch = min(1.0, warm_ratio)  # 前期小，后期到1
        prior_reg_scale_epoch = cfg.prior_reg_scale * reg_scale_epoch
        cyc_reg_scale_epoch   = cfg.cyc_reg_scale   * reg_scale_epoch

        train_metrics = train_one_epoch(
            model, train_loader, optimizer, scaler, device,
            scheduler=scheduler,
            current_epoch=epoch,
            onecycle_epochs=cfg.onecycle_epochs,
            kl_beta=cfg.kl_beta,
            y_tf=y_tf,
            proxy_g=proxy_g,
            lambda_cyc_sim=lam_sim_epoch,
            meas_loader=meas_loader,
            lambda_cyc_meas=lam_meas_epoch,
            y_tf_proxy=y_tf_proxy,
            x_mu_c=x_mu_c, x_std_c=x_std_c,
            x_mu_p=x_mu_p, x_std_p=x_std_p,
            y_idx_c_from_p=y_idx_c_from_p,
            sup_weight=sup_w_epoch,
            prior_l2=cfg.prior_l2,
            prior_bound=cfg.prior_bound,
            prior_bound_margin=cfg.prior_bound_margin,
            trust_alpha=cfg.trust_alpha * warm_ratio,
            trust_tau=cfg.trust_tau,
            yref_proxy_norm=yref_proxy_norm,
            trust_ref_batch=cfg.trust_ref_batch,
            trust_alpha_meas=cfg.trust_alpha_meas * warm_ratio,
            cyc_meas_knn_weight=cfg.cyc_meas_knn_weight,
            cyc_meas_knn_gamma=cfg.cyc_meas_knn_gamma,
            z_sample_mode=cfg.z_sample_mode,
            cond_noise_std=cfg.cond_noise_std,
            cond_noise_clip=cfg.cond_noise_clip,
            prior_reg_scale=prior_reg_scale_epoch,
            cyc_reg_scale=cyc_reg_scale_epoch,
        )
        val_metrics = evaluate_full(
            model, val_loader, device, y_tf=y_tf, proxy_g=proxy_g, lambda_cyc_sim=cfg.lambda_cyc_sim,
            meas_loader=meas_loader, lambda_cyc_meas=cfg.lambda_cyc_meas, y_tf_proxy=y_tf_proxy,
            x_mu_c=x_mu_c, x_std_c=x_std_c, x_mu_p=x_mu_p, x_std_p=x_std_p,
            y_idx_c_from_p=y_idx_c_from_p, sup_weight=cfg.sup_weight, kl_beta=cfg.kl_beta,
            prior_l2=cfg.prior_l2, prior_bound=cfg.prior_bound, prior_bound_margin=cfg.prior_bound_margin,
            enforce_bounds=cfg.enforce_bounds, diag_cfg=diag_cfg, yref_proxy_norm=yref_proxy_norm,
            diag_outdir=run_dir, diag_tag=f"val_ep{epoch:03d}",
            z_sample_mode=cfg.z_sample_mode
        )
        
        for k, v in val_metrics.items(): writer.add_scalar(f'val/{k}', v, epoch)
        for k, v in train_metrics.items(): writer.add_scalar(f'train/{k}', v, epoch)

        print(f" >> Epoch {epoch:03d} | Train Total={train_metrics['total']:.4f} "
              f"| Val post/prior Total={val_metrics['val_total_post']:.4f}/{val_metrics['val_total_prior']:.4f}, Recon post/prior={val_metrics['val_sup_post']:.4f}/{val_metrics['val_sup_prior']:.4f}, KL={val_metrics['val_kl']:.4f}, "
              f"CycSim post/prior={val_metrics['val_cyc_sim_post']:.4f}/{val_metrics['val_cyc_sim_prior']:.4f}, CycMeas={val_metrics['val_cyc_meas']:.4f}, "
              f"PriorBnd post/prior={val_metrics['val_prior_bnd_post']:.2e}/{val_metrics['val_prior_bnd_prior']:.2e} | Best {best_val:.4f} | Patience {no_improve+1}/{cfg.patience}")

        es_value = val_metrics.get(cfg.es_metric, val_metrics['val_total_post'])
        if es_value < best_val - cfg.es_min_delta:
            best_val = es_value
            no_improve = 0
            torch.save({'model': model.state_dict()}, best_path)
            print(f"[Update] best {cfg.es_metric} improved to {best_val:.6f} @ epoch {epoch}")
        else:
            no_improve += 1
            if no_improve >= cfg.patience:
                print(f"[EarlyStop] at epoch {epoch}")
                break

    # Final Test Evaluation
    ckpt = torch.load(best_path, map_location=device)
    model.load_state_dict(ckpt['model'])
    test_metrics = evaluate_full(model, test_loader, device, y_tf=y_tf, proxy_g=proxy_g, lambda_cyc_sim=cfg.lambda_cyc_sim,
                                 meas_loader=meas_loader, lambda_cyc_meas=cfg.lambda_cyc_meas, y_tf_proxy=y_tf_proxy,
                                 x_mu_c=x_mu_c, x_std_c=x_std_c, x_mu_p=x_mu_p, x_std_p=x_std_p, y_idx_c_from_p=y_idx_c_from_p,
                                 sup_weight=cfg.sup_weight, kl_beta=cfg.kl_beta, prior_l2=cfg.prior_l2, prior_bound=cfg.prior_bound,
                                 prior_bound_margin=cfg.prior_bound_margin, enforce_bounds=cfg.enforce_bounds,
                                 diag_cfg=diag_cfg, yref_proxy_norm=yref_proxy_norm, diag_outdir=run_dir, diag_tag="test",
                                 z_sample_mode=cfg.z_sample_mode)

    print(f"[Test] prior total={test_metrics['val_total_prior']:.6f} | prior recon={test_metrics['val_sup_prior']:.6f} | kl={test_metrics['val_kl']:.6f} | "
          f"prior cyc_sim={test_metrics['val_cyc_sim_prior']:.6f} | cyc_meas={test_metrics['val_cyc_meas']:.6f} | "
          f"prior_l2={test_metrics['val_prior_l2_prior']:.6f} | prior_bnd={test_metrics['val_prior_bnd_prior']:.6f} | ")
    
    final_metrics = {f'final/test_{k}': v for k, v in test_metrics.items()}
    add_hparams_safe(writer, run_dir, {"tag": "final"}, final_metrics)
    writer.close()
    return {'run_dir': run_dir, 'best_model': best_path, **test_metrics}

# ============================
# 8) Inference & CLI
# ============================
def load_diffusion_artifacts(run_dir: str, device):
    tr_path = os.path.join(run_dir, 'transforms.json')
    md_path = os.path.join(run_dir, 'best_model.pt')
    assert os.path.isfile(tr_path), f"transforms.json not found: {tr_path}"
    assert os.path.isfile(md_path), f"best_model.pt not found: {md_path}"

    with open(tr_path, 'r') as f:
        meta = json.load(f)

    # restore scalers / transforms
    x_scaler = XStandardizer.from_state_dict(meta['x_scaler'])
    y_tf     = YTransform.from_state_dict(meta['y_transform'])

    cfg_saved = meta['config']
    x_dim = meta['input_dim']
    y_dim = len(y_tf.names)

    # recreate diffusion model skeleton
    model = CondDiffusion(
        x_dim=x_dim,
        y_dim=y_dim,
        hidden=list(cfg_saved['hidden']),
        dropout=cfg_saved['dropout'],
        num_steps=1000,
        beta_start=1e-4,
        beta_end=2e-2,
        device=device
    ).to(device)

    ckpt = torch.load(md_path, map_location=device)
    model.load_state_dict(ckpt['model'])
    model.eval()

    return model, x_scaler, y_tf, meta


def infer_cli(args, device):
    # 1. 加载训练好的 CondDiffusion 模型 + scaler/transform
    model, x_scaler, y_tf, meta = load_diffusion_artifacts(args.infer_run, device)
    model.eval()

    # 2. 读输入数据
    if args.input_npy:
        X = np.load(args.input_npy)
    elif args.input_h5:
        with h5py.File(args.input_h5, 'r') as f:
            X = f['X'][...]
        # 如果是多条样本但用户想指定 index
        if args.index is not None and X.ndim >= 3:
            X = X[int(args.index)]
    else:
        raise ValueError("Provide --input-npy or --input-h5")

    # 把输入拉平成一批 x: [B, x_dim]
    if X.ndim < 3:
        x = X.reshape(1, -1)
    else:
        x = X.reshape(X.shape[0], -1)

    # 标准化到训练域
    x_std = x_scaler.transform(x)  # numpy -> same scaling used in training
    xt = torch.tensor(x_std, dtype=torch.float32, device=device)  # [B, x_dim]

    # 3. 多次采样
    K = max(1, int(args.num_samples))  # 多少个候选解
    all_pred_norm = []
    with torch.no_grad():
        for k in range(K):
            # 每次调用 sample() 都会重新从 torch.randn(...) 开始
            # 这才是扩散里的 "随机性"
            y_k = model.sample(
                xt,            # [B, x_dim]
                num_steps=50,  # 反向扩散步数(DDIM steps); 可以做成args.sample_steps
                guidance_w=0.0, # 目前我们没做classifier-free guidance，先0
                noise_scale=0.2,
            )                 # -> [B, y_dim]
            all_pred_norm.append(y_k.unsqueeze(0))  # [1, B, y_dim]

    # 堆叠: [K, B, y_dim]
    pred_norm = torch.cat(all_pred_norm, dim=0).cpu()  # to CPU tensor

    # 4. 反标准化回物理参数域
    S, B, Dy = pred_norm.shape  # S=K
    pred_phys = y_tf.inverse(
        pred_norm.reshape(-1, Dy)
    ).reshape(S, B, Dy)  # still torch tensor, float32

    # 5. 打印统计信息（只看第一个输入样本）
    first_input_idx = 0
    sols_first = pred_phys[:, first_input_idx, :]  # [K, y_dim]
    mean_sol = sols_first.mean(dim=0)
    std_sol  = sols_first.std(dim=0)

    print("\n--- Solution Statistics (input 0 across {} samples) ---".format(K))
    colw = max(len(n) for n in y_tf.names)
    for i, name in enumerate(y_tf.names):
        print(f"  {name:<{colw}} : mean={float(mean_sol[i]):.4g}  std={float(std_sol[i]):.4g}")

    # 6. 写 CSV
    if args.save_csv:
        # (a) 全部样本逐行写出
        with open(args.save_csv, 'w', newline='') as f:
            wcsv = csv.writer(f)
            header = ['input_idx', 'sample_idx'] + y_tf.names
            wcsv.writerow(header)
            for bi in range(B):
                for si in range(S):
                    row = [bi, si] + pred_phys[si, bi, :].tolist()
                    wcsv.writerow(row)
        print(f"\nSaved {S * B} solutions to {args.save_csv}")

        # (b) 统计均值/方差 (对第一个输入)
        csv_stats = args.save_csv[:-4] + "_mean_std.csv"
        with open(csv_stats, 'w', newline='') as f:
            wcsv = csv.writer(f)
            header = ['name', 'mean', 'std']
            wcsv.writerow(header)
            for i, name in enumerate(y_tf.names):
                row = [name, f"{float(mean_sol[i]):.4g}", f"{float(std_sol[i]):.4g}"]
                wcsv.writerow(row)
        print(f"Saved stats to {csv_stats}")





def parse_args():
    p = argparse.ArgumentParser(description='ASM-HEMT CVAE Training and Inference')
    # Inherited arguments
    p.add_argument('--data', type=str, help='Path to the training data H5 file. Required for training.')
    p.add_argument('--outdir', type=str, default='runs')
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--test-split', type=float, default=0.15)
    p.add_argument('--val-split', type=float, default=0.15)
    p.add_argument('--max-epochs', type=int, default=300)
    p.add_argument('--onecycle-epochs', type=int, default=0)
    p.add_argument('--batch-size', type=int, default=256)
    p.add_argument('--lr', type=float, default=1e-3)
    p.add_argument('--weight-decay', type=float, default=1e-4)
    p.add_argument('--patience', type=int, default=30)
    p.add_argument('--num-workers', type=int, default=0)
    p.add_argument('--compile', action='store_true')
    p.add_argument('--no-onecycle', action='store_true')
    p.add_argument('--aug-noise-std', type=float, default=0.015)
    p.add_argument('--aug-prob', type=float, default=0.5)
    p.add_argument('--no-bounds', action='store_true')
    p.add_argument('--meas-h5', type=str)
    p.add_argument('--lambda-cyc-sim', type=float, default=0.0)
    p.add_argument('--lambda-cyc-meas', type=float, default=0.0)
    p.add_argument('--cyc-warmup-epochs', type=int, default=15)
    p.add_argument('--proxy-run', type=str)
    p.add_argument('--auto-train-proxy', action='store_true')
    p.add_argument('--proxy-hidden', type=str, default='512,512')
    p.add_argument('--proxy-activation', type=str, default='gelu', choices=['gelu','relu','silu'])
    p.add_argument('--proxy-norm', type=str, default='layernorm', choices=['layernorm','batchnorm','none'])
    p.add_argument('--proxy-epochs', type=int, default=100)
    p.add_argument('--proxy-lr', type=float, default=1e-3)
    p.add_argument('--proxy-wd', type=float, default=1e-4)
    p.add_argument('--proxy-beta', type=float, default=0.02)
    p.add_argument('--proxy-seed', type=int)
    p.add_argument('--proxy-patience', type=int, default=15)
    p.add_argument('--proxy-min-delta', type=float, default=1e-6)
    p.add_argument('--proxy-batch-size', type=int, default=1024)
    p.add_argument('--train-proxy-only', action='store_true')
    p.add_argument('--sup-weight', type=float, default=1.0)
    p.add_argument('--prior-l2', type=float, default=1e-3)
    p.add_argument('--prior-bound', type=float, default=1e-3)
    p.add_argument('--prior-bound-margin', type=float, default=0.0)
    p.add_argument('--es-metric', type=str, default='val_cyc_sim_prior', choices=[
        'val_total_prior', 'val_total_post',
        'val_sup_prior', 'val_sup_post',
        'val_cyc_sim_prior', 'val_cyc_sim_post',
        'val_cyc_meas', # only on prior method
        'val_kl',
        'val_prior_l2_prior', 'val_prior_l2_post',
        'val_prior_bnd_prior', 'val_prior_bnd_post',
        ])
    p.add_argument('--es-min-delta', type=float, default=1e-6)
    p.add_argument('--trust-alpha', type=float, default=0.0)
    p.add_argument('--trust-tau', type=float, default=1.6)
    p.add_argument('--trust-ref-max', type=int, default=20000)
    p.add_argument('--trust-ref-batch', type=int, default=4096)
    p.add_argument('--trust-alpha-meas', type=float, default=0.0)
    p.add_argument('--cyc-meas-knn-weight', action='store_true')
    p.add_argument('--cyc-meas-knn-gamma', type=float, default=0.5)
    p.add_argument('--diag', action='store_true', help='Enable 3-probe diagnostics and export CSV')
    p.add_argument('--diag-max-samples', type=int, default=256, help='Max samples for Jacobian SVD (per split)')
    p.add_argument('--diag-knn-k', type=int, default=8, help='k for KNN distance (mean of k-NN)')
    
    # New/Modified Diffusion arguments
    p.add_argument('--hidden', type=str, default='512,256', help='Hidden layers for CVAE components')
    p.add_argument('--latent-dim', type=int, default=32, help='Dimension of the latent space z')
    p.add_argument('--dropout', type=float, default=0.1)
    p.add_argument('--kl-beta', type=float, default=0.1, help='Weight for the KL divergence term in the loss')
    p.add_argument('--cond-noise-std', type=float, default=0.02)
    p.add_argument('--cond-noise-clip', type=float, default=3.0)
    p.add_argument('--prior-reg-scale', type=float, default=0.5)
    p.add_argument('--cyc-reg-scale', type=float, default=0.5)

    # Inference arguments
    p.add_argument('--infer-run', type=str, help='Run dir for inference')
    p.add_argument('--input-npy', type=str, help='Input .npy file for inference')
    p.add_argument('--input-h5', type=str, help='Input .h5 file for inference')
    p.add_argument('--index', type=int, help='Sample index for .h5 input')
    p.add_argument('--save-csv', type=str, help='Path to save inference results')
    p.add_argument('--num-samples', type=int, default=1, help='Number of solutions to sample during inference')
    p.add_argument('--sample-mode', type=str, default='mean', choices=['rand', 'mean'], help='Sampling mode for inference')
    p.add_argument('--z-sample-mode', type=str, default='mean', choices=['rand', 'mean'], help='Sampling mode of latent space for training and testing')



    args = p.parse_args()
    cfg = TrainConfig(
        data=args.data,
        outdir=args.outdir,
        seed=args.seed,
        test_split=args.test_split,
        val_split=args.val_split,
        max_epochs=args.max_epochs,
        onecycle_epochs=args.onecycle_epochs if args.onecycle_epochs > 0 else args.max_epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        weight_decay=args.weight_decay,
        patience=args.patience,
        num_workers=args.num_workers,
        compile=args.compile,
        use_onecycle=not args.no_onecycle,
        aug_noise_std=args.aug_noise_std,
        aug_prob=args.aug_prob,
        enforce_bounds=not args.no_bounds,
        hidden=tuple(int(x) for x in args.hidden.split(',')),
        latent_dim=args.latent_dim,
        dropout=args.dropout,
        kl_beta=args.kl_beta,
        meas_h5=args.meas_h5,
        lambda_cyc_sim=args.lambda_cyc_sim,
        lambda_cyc_meas=args.lambda_cyc_meas,
        cyc_warmup_epochs=args.cyc_warmup_epochs,
        proxy_run=args.proxy_run,
        auto_train_proxy=args.auto_train_proxy or (args.lambda_cyc_sim > 0 or args.lambda_cyc_meas > 0),
        proxy_hidden=tuple(int(x) for x in args.proxy_hidden.split(',')),
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
        sup_weight=args.sup_weight,
        prior_l2=args.prior_l2,
        prior_bound=args.prior_bound,
        prior_bound_margin=args.prior_bound_margin,
        es_metric=args.es_metric,
        es_min_delta=args.es_min_delta,
        trust_alpha=args.trust_alpha,
        trust_tau=args.trust_tau,
        trust_ref_max=args.trust_ref_max,
        trust_ref_batch=args.trust_ref_batch,
        trust_alpha_meas=args.trust_alpha_meas,
        cyc_meas_knn_weight=args.cyc_meas_knn_weight,
        cyc_meas_knn_gamma=args.cyc_meas_knn_gamma,
        num_samples=args.num_samples,
        sample_mode=args.sample_mode,
        z_sample_mode=args.z_sample_mode,
        cond_noise_std=args.cond_noise_std,
        cond_noise_clip=args.cond_noise_clip,
        prior_reg_scale=args.prior_reg_scale,   # 先验相关loss的全局缩放
        cyc_reg_scale=args.cyc_reg_scale   # cycle一致性相关loss全局缩放
    )
    diag_cfg = {'enable': args.diag, 'max_samples': args.diag_max_samples, 'knn_k': args.diag_knn_k}
    return cfg, args, diag_cfg


def main():
    cfg, args, diag_cfg = parse_args()

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    if device.type == 'cuda':
        print(f'Using CUDA device 0: {torch.cuda.get_device_name(0)}')
    else:
        print('CUDA not available, using CPU')

    # --- FIX START: Only require --data for training modes ---
    is_training_mode = not args.infer_run and not args.train_proxy_only
    if is_training_mode and cfg.data is None:
        raise SystemExit("ArgumentError: '--data' is required for training the main CVAE model.")
    # --- FIX END ---

    if args.train_proxy_only:
        # Also check for data in proxy-only mode
        if cfg.data is None:
            raise SystemExit("ArgumentError: '--data' is required for --train-proxy-only.")
        run_proxy_only(cfg, device)
        return

    if args.infer_run:
        infer_cli(args, device)
        return
        
    run_once(cfg, diag_cfg, device)

if __name__ == '__main__':
    main()


