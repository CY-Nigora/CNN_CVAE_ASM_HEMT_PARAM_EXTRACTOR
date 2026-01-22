#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ASM-HEMT CVAE for inverse problem, building upon the two-stage DNN framework.
This version replaces the deterministic MLP with a Conditional VAE to handle
the multi-solution nature of the inverse problem.

Key Architectural Changes:
- Main model 'f' is now a CVAE composed of an Encoder, a Prior Network, and a Decoder.
- Encoder: P(z|x,y) - learns posterior from features and ground truth.
- Prior Network: P(z|x) - learns prior from features only, used for inference.
- Decoder: P(y|x,z) - generates solutions from features and latent samples.
- Loss: Reconstruction + KL(P(z|x,y) || P(z|x)) + cycle consistency + other priors.
- Inference: Samples multiple solutions 'y' by sampling 'z' from the prior P(z|x).
"""
import os
import csv
import json
import math
import time
import argparse
from dataclasses import dataclass, asdict
from typing import List, Tuple, Dict, Optional, Literal
import sys, io, atexit
from datetime import datetime

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
# 0) Print → Console + File (tee) 
# ============================

class _Tee(io.TextIOBase):
    """Write to multiple streams at once (e.g., console + a logfile)."""
    def __init__(self, *streams):
        # streams: e.g., (sys.stdout, file_handle)
        self.streams = list(streams)
        # 选择一个可用的编码作为暴露给外部的 encoding
        enc = None
        for st in self.streams:
            enc = getattr(st, "encoding", None)
            if enc:
                break
        self._encoding = enc or "utf-8"

    # 用只读 property 暴露 encoding，避免直接赋值触发只读报错
    @property
    def encoding(self):
        return self._encoding

    # 某些库也会访问 errors；给个合理默认
    @property
    def errors(self):
        return "strict"

    def write(self, s):
        # 兼容非字符串对象
        if not isinstance(s, str):
            s = str(s)
        wrote = 0
        for st in self.streams:
            try:
                n = st.write(s)
                if isinstance(n, int):
                    wrote = max(wrote, n)
                st.flush()
            except UnicodeEncodeError:
                # 针对非 UTF-8 控制台：降级为 ASCII/本地编码的安全字符串
                enc = getattr(st, "encoding", None) or "ascii"
                safe = s.encode(enc, errors="replace").decode(enc, errors="replace")
                try:
                    n2 = st.write(safe)
                    if isinstance(n2, int):
                        wrote = max(wrote, n2)
                    st.flush()
                except Exception:
                    pass
            except Exception:
                pass
        return wrote

    def flush(self):
        for st in self.streams:
            try:
                st.flush()
            except Exception:
                pass

    def add_stream(self, stream):
        # 运行中追加一个新的输出目的地（比如 run_dir 打开后再加 train.log）
        self.streams.append(stream)

    # 一些环境会检查 isatty；这里统一返回 False
    def isatty(self):
        return False


def _setup_print_tee(out_dir: str, filename: str):
    os.makedirs(out_dir, exist_ok=True)
    log_path = os.path.join(out_dir, filename)
    fh = open(log_path, mode="a", encoding="utf-8", buffering=1)

    if isinstance(sys.stdout, _Tee):
        # 防重复添加：只有当这个文件句柄不在 streams 里时才追加
        if fh not in sys.stdout.streams:
            sys.stdout.add_stream(fh)
        if fh not in sys.stderr.streams:
            sys.stderr.add_stream(fh)
    else:
        sys.stdout = _Tee(sys.stdout, fh)
        sys.stderr = _Tee(sys.stderr, fh)

    atexit.register(lambda: (fh.flush(), fh.close()))
    from datetime import datetime
    print(f"[log] tee initialized at {datetime.now():%Y-%m-%d %H:%M:%S} -> {log_path}")




# ============================
# 0) NaN/Inf sanitizers & safe distance 
# ============================

def _sanitize(t: torch.Tensor, clip: float|None = 1e6) -> torch.Tensor:
    # repalce NaN/Inf totally, and limite the amplitude to avoide 
    # parameter explode in the following stages
    t = torch.nan_to_num(t, nan=0.0, posinf=1e38, neginf=-1e38)
    if clip is not None:
        t = torch.clamp(t, min=-clip, max=clip)
    return t

def _safe_cdist(a: torch.Tensor, b: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    # Avoid negative square roots caused by
    # value cancellation → NaN
    # a: [B,D], b: [K,D]
    a = _sanitize(a); b = _sanitize(b)
    a2 = (a * a).sum(dim=1, keepdim=True)       # [B,1]
    b2 = (b * b).sum(dim=1, keepdim=True).T     # [1,K]
    dist2 = a2 + b2 - 2.0 * (a @ b.T)
    return dist2.clamp_min(eps).sqrt()


# ============================
# 0) for safe running
# ============================

def add_hparams_safe(base_writer: SummaryWriter, run_dir: str, hparams: dict, metrics: dict):
    """
    Simplified: avoid TensorBoard.add_hparams to get rid of weird Windows path issues.
    Just dump hparams + metrics to JSON for record.
    """
    hp_dir = os.path.join(run_dir, "hparams")
    os.makedirs(hp_dir, exist_ok=True)
    try:
        with open(os.path.join(hp_dir, "hparams.json"), "w", encoding="utf-8") as f:
            json.dump({"hparams": hparams, "metrics": metrics}, f, indent=2)
    except Exception as e:
        print(f"[warn] writing hparams.json failed: {e}")


# ============================
# 0) normalization term calculation  (FIXED)
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
        y_log = torch.log10(y_phys[:, log_mask].clamp_min(1e-21)) # FIX: 1e-12 -> 1e-21
        lo_log = torch.log10(lo[log_mask].clamp_min(1e-21))       # FIX: 1e-12 -> 1e-21
        hi_log = torch.log10(hi[log_mask].clamp_min(1e-21))       # FIX: 1e-12 -> 1e-21
        w_log  = (hi_log - lo_log).clamp_min(1e-6)
        over_hi = softplus0((y_log - (hi_log + prior_bound_margin * w_log)) / w_log, beta=2.0)
        over_lo = softplus0(((lo_log - prior_bound_margin * w_log) - y_log) / w_log, beta=2.0)
        term = (over_hi + over_lo).mean(dim=1)
        bound_log = term if per_sample_ena else term.mean()

    return bound_lin + bound_log


def NormCalc_cyc(device, proxy_g, lambda_cyc, y_tf, y_tf_proxy, y_hat_32,
                        x, x_mu_c, x_std_c, x_mu_p, x_std_p, y_idx_c_from_p, cyc_crit):
    # sanitize inputs first
    y_hat_32 = _sanitize(y_hat_32)
    x = _sanitize(x)

    cyc = torch.tensor(0.0, device=device)
    xhat_curr_std = x
    if (proxy_g is not None and lambda_cyc > 0.0):
        y_phys = _sanitize(y_tf.inverse(y_hat_32), clip=None)
        if y_idx_c_from_p is not None:
            y_phys = y_phys.index_select(1, y_idx_c_from_p)

        y_proxy_norm = _sanitize(y_tf_proxy.transform(y_phys))
        xhat_proxy_std = _sanitize(proxy_g(y_proxy_norm))
        xhat_phys = _sanitize(xhat_proxy_std * x_std_p + x_mu_p)
        xhat_curr_std = _sanitize((xhat_phys - x_mu_c) / x_std_c)

        cyc = cyc_crit(xhat_curr_std, x)
    return cyc, xhat_curr_std




# ============================
# 0) Diagnostic processing
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
        # dists = torch.cdist(y_proxy_norm, yref_proxy_norm.to(y_proxy_norm.device), p=2) 
        y_proxy_norm = _sanitize(y_proxy_norm)
        yref_norm_dev = _sanitize(yref_proxy_norm.to(y_proxy_norm.device)) if (yref_proxy_norm is not None) else None
        dists = _safe_cdist(y_proxy_norm, yref_norm_dev)


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


def _effective_bok_k(cfg, phase: str, epoch: int) -> int:
    K = int(getattr(cfg, 'best_of_k', 0) or 0)
    if K <= 1: 
        return 1
    applies = {s.strip() for s in str(getattr(cfg, 'bok_apply', 'train')).split(',') if s.strip()}
    if phase not in applies:
        return 1
    W = int(getattr(cfg, 'bok_warmup_epochs', 0) or 0)
    if W > 0:
        e = max(1, min(int(epoch), W))
        return max(1, int(round(1 + (K - 1) * (e / W))))
    return K


def _bok_flags(cfg, phase: str, epoch: int):
    """return effective_K, use_bok_sim, use_bok_meas"""
    K = _effective_bok_k(cfg, phase, epoch)
    if K <= 1 or str(getattr(cfg, 'bok_target','sim')) == 'none':
        return 1, False, False
    tgt = str(getattr(cfg, 'bok_target','sim'))
    use_sim = (tgt in ('sim','both'))
    use_meas = (tgt in ('meas','both'))
    return K, use_sim, use_meas

def _smooth_l1_per_sample(diff: torch.Tensor, beta: float = 0.02) -> torch.Tensor:
    # diff: [N, D]  -> returns [N]
    absd = diff.abs()
    return torch.where(absd < beta, 0.5*(diff**2)/beta, absd - 0.5*beta).mean(dim=1)


def bok_prior_select_and_cyc(
    model, device, K: int,
    x,                      # [B, Xd] (normalized current x used by cycle)
    mu_prior, logvar_prior, # [B, L]
    y_tf, y_tf_proxy, proxy_g,
    x_mu_c, x_std_c, x_mu_p, x_std_p,
    y_idx_c_from_p,
    cyc_crit,
):
    """
    Returns: y_best32 [B, Dy], cyc_sim_scalar (tensor), x_hat_std_sim_prior (for diagnostics)
    """
    B, L = mu_prior.shape
    # 1) sample K latents for prior
    muK = mu_prior.unsqueeze(1).expand(B, K, L)
    lvK = logvar_prior.unsqueeze(1).expand(B, K, L)
    eps = torch.randn_like(muK)
    zK  = muK + eps * torch.exp(0.5 * lvK)   # [B,K,L]
    xK  = x.unsqueeze(1).expand(B, K, x.shape[-1])  # [B,K,Xd]

    # 2) decoder forward (vectorized B*K)
    in_dec = torch.cat([xK.reshape(B*K, -1), zK.reshape(B*K, -1)], dim=1)  # [B*K, Xd+L]
    yK = model.decoder(in_dec)                                             # [B*K, Dy]
    yK32 = _sanitize(yK.to(torch.float32))                                 # [B*K, Dy]

    # 3) proxy path to get per-sample cycle score to x
    y_physK = _sanitize(y_tf.inverse(yK32))
    if y_idx_c_from_p is not None:
        y_physK = y_physK.index_select(1, y_idx_c_from_p)                  # align columns
    y_proxy_normK   = _sanitize(y_tf_proxy.transform(y_physK))
    xhat_proxy_stdK = _sanitize(proxy_g(y_proxy_normK))                    # [B*K, Xd]
    xhat_physK      = _sanitize(xhat_proxy_stdK * x_std_p + x_mu_p)
    xhat_curr_stdK  = _sanitize((xhat_physK - x_mu_c) / x_std_c)           # compare in current-norm space

    x_ref = _sanitize(xK.reshape(B*K, -1))
    valid = torch.isfinite(xhat_curr_stdK).all(dim=1) & torch.isfinite(x_ref).all(dim=1)
    diff = xhat_curr_stdK - x_ref
    ps = _smooth_l1_per_sample(diff, beta=0.02)                            # [B*K]
    ps = torch.where(valid, ps, torch.full_like(ps, 1e9))

    # 4) pick best index per sample
    cyc_mat = ps.view(B, K)                                                # [B,K]
    best_idx = torch.argmin(cyc_mat, dim=1)                                # [B]
    Dy = yK32.shape[-1]
    y_best32 = yK32.view(B, K, Dy).gather(1, best_idx.view(B,1,1).expand(-1,1,Dy)).squeeze(1)  # [B,Dy]

    # 5) compute canonical cyc_sim with your existing pipeline
    cyc_sim, x_hat_std_sim_prior = NormCalc_cyc(
        device, proxy_g, torch.tensor(1.0, device=device),  # lambda factor outside
        y_tf, y_tf_proxy, y_best32,
        _sanitize(x), x_mu_c, x_std_c, x_mu_p, x_std_p, y_idx_c_from_p, cyc_crit
    )
    return y_best32, cyc_sim, x_hat_std_sim_prior


def bok_prior_select_and_cyc_meas(
    model, device, K: int,
    xm,                     # [B, Xd]   meas-side x in current-normalized space
    y_tf, y_tf_proxy, proxy_g,
    x_mu_c, x_std_c, x_mu_p, x_std_p,
    y_idx_c_from_p,
    cyc_meas_knn_weight: bool = False, cyc_meas_knn_gamma: float = 0.5,
    yref_proxy_norm: Optional[torch.Tensor] = None, trust_tau: float = 1.6,
):
    """
    Returns:
      ym_best32:           [B, Dy]  best parameter candidate (norm space, float32)
      cyc_meas_scalar:     ()       SmoothL1 scalar (aggregated with KNN weighting if enabled)
      xmh_curr_std_best:   [B, Xd]  proxy-generated x (current-norm space) for the best candidate
      y_proxy_norm_best:   [B, Dp]  proxy input vector (norm space) for the best candidate (for trust metrics)
      valid_mask:          [B]      per-sample validity (no NaN/Inf through the path)
      dmin_best:           [B] or None  nearest-neighbor distance to reference set if provided (for trust/KNN weight)
    """
    B = xm.size(0)
    xmK = xm.unsqueeze(1).expand(B, K, xm.shape[-1]).reshape(B*K, -1)
    mu_lv = model.prior_net(xmK)
    muK, lvK = mu_lv.chunk(2, dim=-1)
    muK = muK.view(B, K, -1)
    lvK = lvK.view(B, K, -1)
    eps = torch.randn_like(muK)
    zK  = muK + eps * torch.exp(0.5 * lvK)             # [B,K,L]
    xmK = xm.unsqueeze(1).expand(B, K, xm.shape[-1])   # [B,K,Xd]

    # 2) decode all candidates in a single vectorized pass
    in_dec = torch.cat([xmK.reshape(B*K, -1), zK.reshape(B*K, -1)], dim=1)   # [B*K, Xd+L]
    yK = model.decoder(in_dec)                                               # [B*K, Dy]
    yK32 = _sanitize(yK.to(torch.float32))                                   # safety cast / clamp if needed

    # 3) proxy pipeline: y(norm) -> phys -> (select columns) -> proxy_in -> proxy -> phys -> current-norm
    y_physK = _sanitize(y_tf.inverse(yK32), clip=None)
    if y_idx_c_from_p is not None:
        y_physK = y_physK.index_select(1, y_idx_c_from_p)
    y_proxy_normK   = _sanitize(y_tf_proxy.transform(y_physK))               # [B*K, Dp]
    xhat_proxy_stdK = _sanitize(proxy_g(y_proxy_normK))                      # [B*K, Xd]
    xhat_physK      = _sanitize(xhat_proxy_stdK * x_std_p + x_mu_p)
    xhat_curr_stdK  = _sanitize((xhat_physK - x_mu_c) / x_std_c)             # compare in current-norm space

    # 4) per-sample SmoothL1 score against xm (current-norm)
    xm_ref = _sanitize(xmK.reshape(B*K, -1))
    validK = torch.isfinite(xhat_curr_stdK).all(dim=1) & torch.isfinite(xm_ref).all(dim=1)
    diffK  = xhat_curr_stdK - xm_ref
    psK    = _smooth_l1_per_sample(diffK, beta=0.02)                         # [B*K]
    psK    = torch.where(validK, psK, torch.full_like(psK, 1e9))             # mask invalids with large penalty

    # 5) pick the best candidate per sample (BoK)
    ps_mat   = psK.view(B, K)                                                # [B,K]
    best_idx = torch.argmin(ps_mat, dim=1)                                   # [B]
    Dy = yK32.shape[-1];  Xd = xm.shape[-1];  Dp = y_proxy_normK.shape[-1]
    gather_y   = best_idx.view(B,1,1).expand(-1,1,Dy)
    gather_x   = best_idx.view(B,1,1).expand(-1,1,Xd)
    gather_dpn = best_idx.view(B,1,1).expand(-1,1,Dp)

    ym_best32          = yK32.view(B, K, Dy).gather(1, gather_y).squeeze(1)           # [B,Dy]
    xmh_curr_std_best  = xhat_curr_stdK.view(B, K, Xd).gather(1, gather_x).squeeze(1) # [B,Xd]
    y_proxy_norm_best  = y_proxy_normK.view(B, K, Dp).gather(1, gather_dpn).squeeze(1)# [B,Dp]

    # 6) compute scalar cyc_meas for the selected best candidate (supports KNN weighting)
    valid_best = torch.isfinite(xmh_curr_std_best).all(dim=1) & torch.isfinite(xm).all(dim=1)
    if valid_best.any():
        diff_best = (xmh_curr_std_best[valid_best] - xm[valid_best]).reshape(valid_best.sum(), -1)
        beta = 0.02
        absd = diff_best.abs()
        cyc_ps = torch.where(absd < beta, 0.5 * (diff_best**2) / beta, absd - 0.5*beta).mean(dim=1)  # [Nv]

        dmin_best = None
        if ((cyc_meas_knn_weight or (yref_proxy_norm is not None)) and (yref_proxy_norm is not None)):
            # use a subsampled reference to avoid O(N^2)
            Nref = yref_proxy_norm.shape[0]
            idx  = torch.randint(0, Nref, (min(Nref,  trust_tau if isinstance(trust_tau,int) else 4096),), device=yref_proxy_norm.device)
            yref_sub = yref_proxy_norm.index_select(0, idx)
            dists_m  = _safe_cdist(y_proxy_norm_best[valid_best], yref_sub)  # [Nv, Nsub]
            dmin_best= dists_m.min(dim=1).values

        if cyc_meas_knn_weight and (dmin_best is not None):
            w_knn = torch.clamp(dmin_best / max(1e-6, trust_tau), min=0.0, max=4.0).pow(cyc_meas_knn_gamma).detach()
            cyc_meas_scalar = (w_knn * cyc_ps).mean()
        else:
            cyc_meas_scalar = cyc_ps.mean()
    else:
        cyc_meas_scalar = xm.new_tensor(0.0)
        dmin_best = None

    return ym_best32, cyc_meas_scalar, xmh_curr_std_best, y_proxy_norm_best, valid_best, dmin_best

# ============================
# 2) Data & transforms
# ============================
PARAM_NAMES = [
    'VOFF', 'U0', 'NS0ACCS', 'NFACTOR', 'ETA0',
    'VSAT', 'VDSCALE', 'CDSCD', 'LAMBDA', 'MEXPACCD', 'DELTA' , 'UA', 'UB', 'U0ACCS'
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
    'DELTA': (2, 100),
    'UA': (1e-10, 1e-8),
    'UB': (1e-21, 3e-16),
    'U0ACCS': (5e-2, 0.25)
}

_param_lo = []
_param_hi = []
for name in PARAM_NAMES:
    lo, hi = PARAM_RANGE[name]
    _param_lo.append(lo)
    _param_hi.append(hi)

# 注册为全局张量（惰性搬到 device）
PARAM_LO = torch.tensor(_param_lo, dtype=torch.float32)  # shape [Dy]
PARAM_HI = torch.tensor(_param_hi, dtype=torch.float32)

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
        y_t = y.to(torch.float32).clone()
        mask = self.log_mask.to(y_t.device)
        if mask.any():
            # FIX: Change 1e-12 to 1e-21 to accommodate UB (1e-21)
            y_t[:, mask] = torch.log10(y_t[:, mask].clamp_min(1e-21)) 
        self.mean = y_t.mean(dim=0).detach().cpu().to(torch.float32)
        self.std  = y_t.std(dim=0).clamp_min(1e-8).detach().cpu().to(torch.float32)

    def transform(self, y: torch.Tensor) -> torch.Tensor:
        y_t = y.to(torch.float32)
        mask = self.log_mask.to(y.device)
        if mask.any():
            y_t = y_t.clone()
            y_t[:, mask] = torch.log10(y_t[:, mask].clamp_min(1e-21))
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
    def __init__(self, x: np.ndarray, y: np.ndarray,
                 augment_std: float = 0.0, augment_prob: float = 0.0,
                 weights: Optional[np.ndarray] = None,
                 grid_shape: Optional[Tuple[int,int]] = None,
                 gain_std: float = 0.0,
                 row_gain_std: float = 0.0,
                 smooth_window: int = 0):
        assert x.shape[0] == y.shape[0]
        self.x = x.astype(np.float32)
        self.y = y.astype(np.float32)
        self.w = None if weights is None else weights.astype(np.float32).reshape(-1)
        self.augment_std = float(augment_std)
        self.augment_prob = float(augment_prob)
        self.grid_shape = grid_shape
        self.gain_std = float(gain_std)
        self.row_gain_std = float(row_gain_std)
        self.smooth_window = int(smooth_window)
        self._aug_scale = 1.0  # 允许训练过程中动态调节

    def set_aug_scale(self, s: float):
        self._aug_scale = float(max(0.0, s))

    @staticmethod
    def _smooth1d(v: np.ndarray, win: int) -> np.ndarray:
        # 简单滑动平均，沿着最后一维做卷积
        if win <= 1: return v
        win = int(win)
        if win % 2 == 0: win += 1
        k = np.ones((win,), dtype=np.float32) / float(win)
        # v: (..., L)
        out = np.apply_along_axis(lambda a: np.convolve(a, k, mode='same'), -1, v)
        return out.astype(np.float32)

    def __len__(self): return self.x.shape[0]

    def __getitem__(self, idx):
        xi = self.x[idx].copy()
        yi = self.y[idx].copy()

        if self.augment_prob > 0.0 and (self.augment_std > 0.0 or self.gain_std > 0.0 or self.row_gain_std > 0.0):
            import random
            if random.random() < self.augment_prob:
                # 1) 乘性增益（整条样本）
                if self.gain_std > 0.0:
                    # LogNormal(0, sigma)：保证正增益；sigma 经 _aug_scale 缩放
                    sigma = self.gain_std * self._aug_scale
                    alpha = np.float32(np.random.lognormal(mean=0.0, sigma=sigma))
                    xi = (xi * alpha).astype(np.float32)


                # 2) 行级乘性增益（按 Vgs 行）
                if (self.grid_shape is not None) and (self.row_gain_std > 0.0):
                    R, C = self.grid_shape
                    x2 = xi.reshape(R, C).copy()
                    sigma_r = self.row_gain_std * self._aug_scale
                    alphas = np.random.lognormal(mean=0.0, sigma=sigma_r, size=(R,)).astype(np.float32)
                    x2 = (x2.T * alphas).T  # 每行一个增益
                    xi = x2.reshape(-1)

                # 3) 加性白噪（标准化域）
                if self.augment_std > 0.0:
                    std = np.float32(self.augment_std * self._aug_scale)
                    eps = np.random.randn(*xi.shape).astype(np.float32) * std
                    if (self.grid_shape is not None) and (self.smooth_window > 1):
                        # 沿列方向平滑（不破坏行/曲线形状）
                        R, C = self.grid_shape
                        eps2 = eps.reshape(R, C)
                        eps2 = self._smooth1d(eps2, self.smooth_window)
                        eps = eps2.reshape(-1)
                    xi = (xi + eps).astype(np.float32)

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

class CVAE(nn.Module):
    """
    Conditional VAE with a separate Prior network.
    - Encoder P(z|x,y): Encodes latent space from x and y (training).
    - PriorNet P(z|x): Predicts latent space from x only (inference).
    - Decoder P(y|x,z): Reconstructs y from x and a latent sample z.
    """
    def __init__(self, x_dim: int, y_dim: int, hidden: List[int], latent_dim: int, dropout: float = 0.1):
        super().__init__()
        self.y_dim = y_dim
        self.latent_dim = latent_dim

        # Encoder P(z|x, y)
        self.encoder = _MLPBlock(x_dim + y_dim, hidden, latent_dim * 2, dropout)
        # Prior network P(z|x)
        self.prior_net = _MLPBlock(x_dim, hidden, latent_dim * 2, dropout)
        # Decoder P(y|x, z)
        self.decoder = _MLPBlock(x_dim + latent_dim, hidden, y_dim, dropout)

    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x: torch.Tensor, y: Optional[torch.Tensor] = None) -> Tuple:
        # Prior P(z|x) - always needed for KL loss
        prior_out = self.prior_net(x)
        mu_prior, logvar_prior = prior_out.chunk(2, dim=-1)

        if y is not None: # Training mode with ground truth y
            # Posterior P(z|x,y)
            encoder_input = torch.cat([x, y], dim=1)
            encoder_out = self.encoder(encoder_input)
            mu_post, logvar_post = encoder_out.chunk(2, dim=-1)

            # Sample from posterior for reconstruction
            z = self.reparameterize(mu_post, logvar_post)

            # Decode
            decoder_input = torch.cat([x, z], dim=1)
            y_hat = self.decoder(decoder_input)

            return y_hat, (mu_post, logvar_post), (mu_prior, logvar_prior)
        else: # Inference mode without y
            z = self.reparameterize(mu_prior, logvar_prior)
            decoder_input = torch.cat([x, z], dim=1)
            y_hat = self.decoder(decoder_input)
            # Return y_hat and prior stats (posterior is not available)
            return y_hat, (None, None), (mu_prior, logvar_prior)

    def sample(self, x: torch.Tensor, num_samples: int = 1, sample_mode: str = 'rand') -> torch.Tensor:
        self.eval()
        with torch.no_grad():
            B = x.size(0)
            prior_out = self.prior_net(x)
            mu_prior, logvar_prior = prior_out.chunk(2, dim=-1)

            y_samples = []
            for _ in range(num_samples):
                if sample_mode == 'mean':
                    z = mu_prior
                else: # 'rand'
                    z = self.reparameterize(mu_prior, logvar_prior)

                decoder_input = torch.cat([x, z], dim=1)
                y_hat = self.decoder(decoder_input)
                y_samples.append(y_hat.unsqueeze(0))

        return torch.cat(y_samples, dim=0)


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

@contextmanager
def dropout_mode(model: nn.Module, enabled: bool):
    """Temporarily enable/disable ONLY Dropout layers without touching others (e.g., BatchNorm stays eval)."""
    states = []
    for m in model.modules():
        if isinstance(m, nn.Dropout):
            states.append((m, m.training))
            m.train(enabled)   # True: enable dropout  | False: disable dropout
    try:
        yield
    finally:
        for m, was_training in states:
            m.train(was_training)


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

    # add noise in X
    aug_gain_std: float = 0.0           # overall multiplicative gain is LogNormal(0, aug_gain_std)
    aug_row_gain_std: float = 0.0       # row/curve level multiplicative gain (mesh shape required)
    aug_smooth_window: int = 0          # window length (along the column direction) for 1D sliding smoothing of additive noise
    aug_schedule: str = "none"          # ["none", "linear_decay", "cosine"]
    aug_final_scale: float = 0.5        # end-of-schedule Noise Scaling Factor

    # additional dropout while eval/infer
    dropout_val: bool = False
    dropout_test: bool = False
    dropout_infer: bool = False

    # Best of K in prior path
    best_of_k: int = 0                 # 0/1 => off, >1 => enable BoK
    bok_warmup_epochs: int = 0         # warm up K from 1 -> best_of_k in first W epochs
    bok_target: str = "sim"            # 'sim' | 'meas' | 'both' | 'none'
    bok_apply: str = "train"           # comma list among: train, val, test


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
    grid_shape = None
    if X.ndim == 3:                 # (N, rows, cols)
        grid_shape = (int(X.shape[1]), int(X.shape[2]))
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

    train_ds = ArrayDataset(
            X_tr, Y_tr,
            augment_std=cfg.aug_noise_std,
            augment_prob=cfg.aug_prob,
            grid_shape=grid_shape,                     # 新增
            gain_std=cfg.aug_gain_std,                 # 新增
            row_gain_std=cfg.aug_row_gain_std,         # 新增
            smooth_window=cfg.aug_smooth_window        # 新增
        )
    val_ds   = ArrayDataset(X_va, Y_va)
    test_ds  = ArrayDataset(X_te, Y_te)

    return train_ds, val_ds, test_ds, x_scaler, y_tf, (tr_idx, va_idx, te_idx), X, Y


def make_loaders(train_ds, val_ds, test_ds, batch_size: int, num_workers: int):
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,  num_workers=num_workers, pin_memory=True, persistent_workers=False)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True, persistent_workers=False)
    test_loader  = DataLoader(test_ds,  batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True, persistent_workers=False)
    return train_loader, val_loader, test_loader


def make_meas_loader(meas_h5: str, x_scaler: XStandardizer, batch_size: int, num_workers: int = 0):
    assert os.path.isfile(meas_h5), f"Meas file not found: {meas_h5}"
    with h5py.File(meas_h5, 'r') as f:
        Xm = f['X'][...]                       # ← 注意这里用 [...], 不是 [.]
    Nm = Xm.shape[0]
    if Nm == 0:
        print("[warn] meas set is empty (Nm=0); MEAS branch will be disabled.")
        return None, 0

    Xm = Xm.reshape(Nm, -1).astype(np.float32)
    Xm_std = x_scaler.transform(Xm)
    ds = MeasDataset(Xm_std)

    bs_meas = min(batch_size, Nm)              # ← 自适应 batch size，永远 ≥1
    loader = DataLoader(
        ds,
        batch_size=bs_meas,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=False,
        drop_last=False                        # ← 不丢尾，避免 0 batch
    )
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

        def _eval(Xs, Ys, eval_bs: int = 2048):
            model.eval()
            n = Xs.shape[0]
            total = 0.0
            cnt = 0
            with torch.no_grad():
                for i in range(0, n, eval_bs):
                    xs_chunk = Xs[i:i+eval_bs]
                    ys_chunk = Ys[i:i+eval_bs]

                    xt = torch.from_numpy(xs_chunk).to(device, non_blocking=True)
                    yt = torch.from_numpy(ys_chunk).to(device, non_blocking=True)

                    xhat = model(yt)
                    loss = crit(xhat, xt)

                    bs = xt.size(0)
                    total += loss.item() * bs
                    cnt   += bs

                    del xt, yt, xhat, loss
            model.train()
            return total / max(1, cnt)

        best = float('inf')
        no_improve = 0
        best_path = os.path.join(outdir, 'proxy_g.pt')
        os.makedirs(outdir, exist_ok=True)

        B = int(batch_size)
        n = X_tr_std.shape[0]
        rng = np.random.default_rng(local_seed)

        amp_dtype = (torch.bfloat16 if (device.type=='cuda' and torch.cuda.is_bf16_supported())
                    else (torch.float16 if device.type=='cuda' else torch.bfloat16))
        ac_kwargs = (
            dict(device_type='cuda', dtype=amp_dtype, enabled=True)
            if device.type == 'cuda' else
            dict(device_type='cpu', dtype=torch.bfloat16, enabled=False)
        )


        for ep in range(1, max_epochs + 1):
            model.train()
            total = 0.0
            cnt = 0

            # 用“抽样近似一个 epoch”，避免一次性分配 n 个 index
            steps = max(1, math.ceil(n / B))

            for step in range(steps):
                # 每个 step 随机采样一个 batch 的下标（有放回采样）
                idx = rng.integers(low=0, high=n, size=B, endpoint=False)

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
                    kl_beta: float = 0.1,
                    y_tf: Optional[YTransform]=None,
                    proxy_g: Optional[nn.Module]=None,
                    lambda_cyc_sim: float=0.0,
                    meas_loader: Optional[DataLoader]=None,
                    lambda_cyc_meas: float=0.0,
                    y_tf_proxy: Optional[YTransform]=None,
                    x_mu_c: Optional[torch.Tensor]=None, x_std_c: Optional[torch.Tensor]=None,
                    x_mu_p: Optional[torch.Tensor]=None, x_std_p: Optional[torch.Tensor]=None,
                    y_idx_c_from_p: Optional[torch.Tensor]=None,
                    sup_weight: float = 1.0,
                    prior_l2: float = 1e-3,
                    prior_bound: float = 1e-3,
                    prior_bound_margin: float = 0.0,
                    trust_alpha: float = 0.0, trust_tau: float = 1.6,
                    yref_proxy_norm: Optional[torch.Tensor] = None,
                    trust_ref_batch: int = 4096,
                    trust_alpha_meas: float = 0.0,
                    cyc_meas_knn_weight: bool = False,
                    cyc_meas_knn_gamma: float = 0.5,
                    z_sample_mode: Literal['mean', 'rand'] = 'mean',
                    best_of_k: int = 1, bok_use_sim: bool = False, bok_use_meas: bool = False):
    model.train()
    total_loss, n = 0.0, 0
    total_recon, total_kl, total_cyc_sim, total_cyc_meas = 0.0, 0.0, 0.0, 0.0
    total_prior_l2, total_prior_bnd = 0.0, 0.0
    total_cyc_meas_sum = 0.0  
    total_cyc_meas_n   = 0    # effective samples


    recon_crit = nn.SmoothL1Loss(beta=0.02, reduction='mean')
    cyc_crit = nn.SmoothL1Loss(beta=0.02, reduction='mean')
    meas_iter = iter(meas_loader) if meas_loader is not None else None

    for x, y in loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        
        # before run, pre define:
        # bf16 first, AMP can be turned off for the first few epochs as needed
        amp_dtype = (torch.bfloat16 if (device.type == 'cuda' and torch.cuda.is_bf16_supported())
                    else (torch.float16 if device.type == 'cuda' else torch.bfloat16))
        amp_enabled = True   # for greater stability, turn it off during the first 10 epochs: (current_epoch > 10)

        # 1) Main CVAE forward pass + loss
        with torch.autocast(device_type=('cuda' if device.type=='cuda' else 'cpu'),
                    dtype=amp_dtype, enabled=amp_enabled):
            y_hat_post, (mu_post, logvar_post), (mu_prior, logvar_prior) = model(x, y)
            # sampling mode choose:
            # 1. traditionally use mean
            # 2. advanced way with rand 
            if z_sample_mode == 'mean':
                z_prior = mu_prior
            else: # z_sample_mode == 'rand'
                z_prior = model.reparameterize(mu_prior, logvar_prior)
            y_hat_prior = model.decoder(torch.cat([x, z_prior], dim=1))

            # for safety, remove nan/inf
            y_hat_post  = _sanitize(y_hat_post)
            y_hat_prior = _sanitize(y_hat_prior)
            
            recon_loss = recon_crit(y_hat_post, y)
            kl_loss = calculate_kl_divergence(mu_post, logvar_post, mu_prior, logvar_prior)
            
            loss = sup_weight * recon_loss + kl_beta * kl_loss

        # 2) Consistency + Priors (in FP32 for stability)
        with torch.autocast(device_type='cuda' if device.type=='cuda' else 'cpu', enabled=False):
            # y_hat_32 = y_hat_post.to(torch.float32)
            y_hat_post_32  = _sanitize(y_hat_post.to(torch.float32))
            y_hat_prior_32 = _sanitize(y_hat_prior.to(torch.float32))

            # L2 Prior
            prior_term_l2 = y_hat_post_32.pow(2).mean() if prior_l2 > 0.0 else y_hat_post_32.new_tensor(0.0)

            # Soft Boundary Prior
            prior_term_bnd = NormCalc_prior_bnd(device, y_tf, y_hat_post_32, PARAM_RANGE, prior_bound, prior_bound_margin )
            
            loss += prior_l2 * prior_term_l2 + prior_bound * prior_term_bnd

            cyc_sim  = y_hat_post_32.new_tensor(0.0)  
            cyc_meas = y_hat_post_32.new_tensor(0.0)  

            # Cycle-Sim Consistency
            if (proxy_g is not None) and (lambda_cyc_sim > 0.0):
                if bok_use_sim and best_of_k > 1:
                    y_best32, cyc_sim_unit, x_hat_std_sim_prior = bok_prior_select_and_cyc(
                        model, device, best_of_k,
                        x, mu_prior, logvar_prior,
                        y_tf, y_tf_proxy, proxy_g,
                        x_mu_c, x_std_c, x_mu_p, x_std_p,
                        y_idx_c_from_p,
                        cyc_crit,
                    )
                    # the selected best candidate replaces the prior output, 
                    # facilitating unified diagnostics and logging
                    y_hat_prior_32 = y_best32
                    loss = loss + lambda_cyc_sim * cyc_sim_unit
                    cyc_sim = cyc_sim_unit
                else:
                    cyc_sim, x_hat_std_sim_prior = NormCalc_cyc(
                        device, proxy_g, lambda_cyc_sim,
                        y_tf, y_tf_proxy, y_hat_prior_32,
                        x, x_mu_c, x_std_c, x_mu_p, x_std_p, y_idx_c_from_p, cyc_crit
                    )
                    loss += lambda_cyc_sim * cyc_sim

            # Cycle-Meas Consistency
            cyc_meas = y_hat_post_32.new_tensor(0.0)
            if (proxy_g is not None and meas_iter is not None and lambda_cyc_meas > 0.0):
                xm = None
                try:
                    xm = next(meas_iter)
                except StopIteration:
                    try:
                        meas_iter = iter(meas_loader)
                        xm = next(meas_iter)
                    except StopIteration:
                        xm = None     # ← 彻底为空：这一轮跳过 MEAS 分支
                if xm is not None:
                    xm = xm.to(device, non_blocking=True)

                if bok_use_meas and best_of_k > 1:
                    # === Best-of-K for MEAS branch ===
                    ym_best32, cyc_meas_unit, xmh_curr_std_best, y_proxy_norm_best, valid_best, dmin_best = bok_prior_select_and_cyc_meas(
                        model, device, best_of_k,
                        _sanitize(xm),
                        y_tf, y_tf_proxy, proxy_g,
                        x_mu_c, x_std_c, x_mu_p, x_std_p,
                        y_idx_c_from_p,
                        cyc_meas_knn_weight=cyc_meas_knn_weight, cyc_meas_knn_gamma=cyc_meas_knn_gamma,
                        yref_proxy_norm=yref_proxy_norm, trust_tau=trust_tau
                    )
                    cyc_meas = cyc_meas_unit
                    loss += lambda_cyc_meas * cyc_meas

                    # accumulate metrics using only valid samples (aligns with your original accounting)
                    if valid_best.any():
                        valid_count = int(valid_best.sum().item())
                        total_cyc_meas_sum += cyc_meas.detach().item() * valid_count
                        total_cyc_meas_n   += valid_count

                    # optional trust loss on MEAS side, consistent with your previous definition
                    if (trust_alpha_meas > 0.0) and (dmin_best is not None) and (dmin_best.numel() > 0):
                        trust_loss_m = torch.clamp_min(dmin_best - trust_tau, 0.0).pow(2).mean()
                        loss += trust_alpha_meas * trust_loss_m

                else:
                    # === original single-sample path (kept intact) ===
                    was_training = model.training
                    if was_training: model.eval()
                    ym_hat, _, _ = model(xm)  # forward(x) in inference mode returns y_hat and prior stats
                    if was_training: model.train()

                    ym_hat = _sanitize(ym_hat)
                    ym_phys = y_tf.inverse(ym_hat.to(torch.float32))
                    ym_phys_proj = (ym_phys.index_select(1, y_idx_c_from_p) if y_idx_c_from_p is not None else ym_phys)
                    ym_proxy_norm = _sanitize(y_tf_proxy.transform(ym_phys_proj))
                    xmh_proxy_std = _sanitize(proxy_g(ym_proxy_norm))
                    xmh_phys = _sanitize(xmh_proxy_std * x_std_p + x_mu_p)
                    xmh_curr_std = _sanitize((xmh_phys - x_mu_c) / x_std_c)
                    xm = _sanitize(xm)

                    valid = torch.isfinite(xmh_curr_std).all(dim=1) & torch.isfinite(xm).all(dim=1)
                    if valid.any():
                        diff = (xmh_curr_std[valid] - xm[valid]).reshape(valid.sum(), -1)
                        beta = 0.02
                        absd = diff.abs()
                        cyc_ps_m = torch.where(absd < beta, 0.5 * (diff**2) / beta, absd - 0.5 * beta).mean(dim=1)

                        dmin_m = None
                        if ((cyc_meas_knn_weight or (trust_alpha_meas > 0.0)) and (yref_proxy_norm is not None)):
                            Nref = yref_proxy_norm.shape[0]
                            idx = torch.randint(0, Nref, (min(Nref, trust_ref_batch),), device=yref_proxy_norm.device)
                            yref_sub = yref_proxy_norm.index_select(0, idx)
                            dists_m = _safe_cdist(ym_proxy_norm[valid], yref_sub)
                            dmin_m = dists_m.min(dim=1).values

                        if cyc_meas_knn_weight and dmin_m is not None:
                            w_knn = torch.clamp(dmin_m / max(1e-6, trust_tau), min=0.0, max=4.0).pow(cyc_meas_knn_gamma).detach()
                            cyc_meas = (w_knn * cyc_ps_m).mean()
                        else:
                            cyc_meas = cyc_ps_m.mean()
                    else:
                        cyc_meas = y_hat_post_32.new_tensor(0.0)

                    loss += lambda_cyc_meas * cyc_meas

                    if 'valid' in locals():
                        valid_count = int(valid.sum().item())
                        if valid_count > 0:
                            total_cyc_meas_sum += cyc_meas.detach().item() * valid_count
                            total_cyc_meas_n   += valid_count

                    if (trust_alpha_meas > 0.0) and ('dmin_m' in locals()) and (dmin_m is not None) and dmin_m.numel() > 0:
                        trust_loss_m = torch.clamp_min(dmin_m - trust_tau, 0.0).pow(2).mean()
                        loss += trust_alpha_meas * trust_loss_m



            # L_trust
            if (trust_alpha > 0.0 and yref_proxy_norm is not None):
                y_phys = y_tf.inverse(y_hat_post_32)
                y_phys_p = y_phys.index_select(1, y_idx_c_from_p) if y_idx_c_from_p is not None else y_phys
                y_proxy_norm = _sanitize(y_tf_proxy.transform(y_phys_p))
                
                Nref = yref_proxy_norm.shape[0]
                idx = torch.randint(0, Nref, (min(Nref, trust_ref_batch),), device=yref_proxy_norm.device)
                yref_sub = yref_proxy_norm.index_select(0, idx)
                
                dists = _safe_cdist(y_proxy_norm, yref_sub)
                dmin = dists.min(dim=1).values
                trust_loss = torch.clamp_min(dmin - trust_tau, 0.0).pow(2).mean()
                loss += trust_alpha * trust_loss

        if not torch.isfinite(loss):
            print(f"[nan-guard] non-finite loss: recon={float(recon_loss):.4g}, kl={float(kl_loss):.4g}, "
                  f"cyc_sim={float(cyc_sim):.4g}, cyc_meas={float(cyc_meas):.4g}, prior_l2={float(prior_term_l2):.4g}, "
                  f"prior_bnd={float(prior_term_bnd):.4g} -> skip step")
            continue

        scaler.scale(loss).backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=2.0)
        scaler.step(optimizer)
        scaler.update()
        if (scheduler is not None) and (current_epoch <= onecycle_epochs):
            scheduler.step()

        bs = int(x.size(0))
        total_loss += loss.item() * bs; n += bs
        total_recon += recon_loss.item() * bs
        total_kl += kl_loss.item() * bs
        total_prior_l2  += prior_term_l2.detach().item()  * bs
        total_prior_bnd += prior_term_bnd.detach().item() * bs
        total_cyc_sim   += cyc_sim.detach().item()        * bs
        # total_cyc_meas  += cyc_meas.detach().item()       * bs

    return {
        'total': total_loss / max(1, n),
        'recon': total_recon / max(1, n),
        'kl': total_kl / max(1, n),
        'cyc_sim': total_cyc_sim / max(1, n),
        'cyc_meas': (total_cyc_meas_sum / max(1, total_cyc_meas_n)),
        'prior_l2': total_prior_l2 / max(1, n),
        'prior_bnd': total_prior_bnd / max(1, n),
    }

@torch.no_grad()
def evaluate_full(model, loader, device,
                  *,
                  y_tf: YTransform,
                  proxy_g: Optional[nn.Module],
                  lambda_cyc_sim: float,
                  meas_loader: Optional[DataLoader],
                  lambda_cyc_meas: float,
                  y_tf_proxy: YTransform,
                  x_mu_c: torch.Tensor, x_std_c: torch.Tensor,
                  x_mu_p: torch.Tensor, x_std_p: torch.Tensor,
                  y_idx_c_from_p: torch.Tensor,
                  sup_weight: float, kl_beta: float, prior_l2: float, prior_bound: float, prior_bound_margin: float,
                  enforce_bounds: bool = False,
                  diag_cfg: Optional[Dict] = None,
                  yref_proxy_norm: Optional[torch.Tensor] = None,
                  diag_outdir: Optional[str] = None,
                  diag_tag: Optional[str] = None,
                  z_sample_mode: Literal['mean', 'rand'] = 'mean',
                  dropout_in_eval: bool = False,
                  best_of_k: int = 1, bok_use_sim: bool = False, bok_use_meas: bool = False
                  ) -> Dict[str, float]:
    model.eval()
    recon_crit = nn.SmoothL1Loss(beta=0.02, reduction='mean')
    cyc_crit = nn.SmoothL1Loss(beta=0.02, reduction='mean')
    total_recon, total_kl, total_prior_l2, total_prior_bnd, total_cyc_sim = 0.0, 0.0, 0.0, 0.0, 0.0
    n_sup, n_sim = 0, 0

    # Diagnostics
    diag_rows = []
    diag_enabled = bool(diag_cfg and diag_cfg.get('enable', False))
    diag_max = int(diag_cfg.get('max_samples', 64)) if diag_cfg else 0
    diag_k = int(diag_cfg.get('knn_k', 8)) if diag_cfg else 0
    diag_count = 0
    proxy_floor_all = []
    
    # On validation/test set with labels (y)
    meter = defaultdict(float)
    print('[Diag] Start to calcaulte diag in test preocess... ') if diag_enabled and (y_tf is not None) and (y_tf_proxy is not None) and (proxy_g is not None) and (diag_tag == "test") else None
    with dropout_mode(model, enabled=dropout_in_eval):
        for x, y in loader:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)

            with torch.no_grad():
                # Posterior 路径（训练/ELBO一致）：q(z|x,y) ---
                y_hat_post, (mu_post, logvar_post), (mu_prior, logvar_prior) = model(x, y)
                if z_sample_mode == 'mean':
                # Prior 路径（部署口径）：p(z|x) → 这里用 prior-mean 更稳定 ---
                    z_prior = mu_prior                       # 也可改为 sample：mu_prior + torch.randn_like(mu_prior)*torch.exp(0.5*logvar_prior)
                else: # random sampling
                    z_prior = model.reparameterize(mu_prior, logvar_prior)
                y_hat_prior = model.decoder(torch.cat([x, z_prior], dim=1))
                
                # Sup（重构项，和你现有的 loss 一致口径） ----------
                sup_post  = recon_crit(y_hat_post,  y)
                sup_prior = recon_crit(y_hat_prior, y)

                # KL(post||prior)（用于posterior口径的val_total） ----------
                kl = calculate_kl_divergence(mu_post, logvar_post, mu_prior, logvar_prior).mean()

                # Priors (calculated on y_hat)
                # y_hat_post_32 = y_hat_post.to(torch.float32)
                # y_hat_prior_32 = y_hat_prior.to(torch.float32)
                y_hat_post_32  = _sanitize(y_hat_post.to(torch.float32))
                y_hat_prior_32 = _sanitize(y_hat_prior.to(torch.float32))
                prior_l2_post  = y_hat_post_32.pow(2).mean()
                prior_l2_prior = y_hat_prior_32.pow(2).mean()
                prior_bnd_post = NormCalc_prior_bnd(device, y_tf, y_hat_post_32, PARAM_RANGE, prior_bound, prior_bound_margin)
                prior_bnd_prior = NormCalc_prior_bnd(device, y_tf, y_hat_prior_32, PARAM_RANGE, prior_bound, prior_bound_margin)

                
                # Cycle-Sim
                cyc_sim_post, x_hat_std_sim_post = NormCalc_cyc(device, proxy_g, lambda_cyc_sim, y_tf, y_tf_proxy, y_hat_post_32,
                                    _sanitize(x), x_mu_c, x_std_c, x_mu_p, x_std_p, y_idx_c_from_p, cyc_crit)
                cyc_sim_prior, x_hat_std_sim_prior = NormCalc_cyc(device, proxy_g, lambda_cyc_sim, y_tf, y_tf_proxy, y_hat_prior_32,
                                    _sanitize(x), x_mu_c, x_std_c, x_mu_p, x_std_p, y_idx_c_from_p, cyc_crit)

                # record
                meter['sup_post']  += sup_post.item()  * x.size(0)
                meter['sup_prior'] += sup_prior.item() * x.size(0)
                meter['cyc_sim_post']  += cyc_sim_post.item()  * x.size(0)
                meter['cyc_sim_prior'] += cyc_sim_prior.item() * x.size(0)
                meter['prior_l2_post']  += prior_l2_post.item()  * x.size(0)
                meter['prior_l2_prior'] += prior_l2_prior.item() * x.size(0)
                meter['prior_bnd_post']  += prior_bnd_post.item()  * x.size(0)
                meter['prior_bnd_prior'] += prior_bnd_prior.item() * x.size(0)
                meter['kl'] += kl.item() * x.size(0)
                meter['n']  += x.size(0)
            
            # diagnostics recording
            domain_label = 'sim'
            diag_rows, diag_count = diag_processing(
                        model, proxy_g, device, 
                        domain_label, 
                        diag_rows, diag_count, diag_max, diag_k,
                        x, x_hat_std_sim_prior, x_mu_p, x_std_p, x_mu_c, x_std_c,
                        y, y_hat_prior, y_hat_prior_32, y_tf, y_tf_proxy, y_idx_c_from_p, yref_proxy_norm,
                        prior_bound, prior_bound_margin, PARAM_RANGE,
                        proxy_floor_all=proxy_floor_all
                        ) if diag_enabled and (y_tf is not None) and (y_tf_proxy is not None) and (proxy_g is not None) and (diag_tag == "test") else (diag_rows, diag_count)
            

    # On measurement set (no labels)
    if (proxy_g is not None and meas_loader is not None and lambda_cyc_meas > 0.0):
        for xm in meas_loader:
            # 1) clean up the input
            xm = _sanitize(xm.to(device, non_blocking=True))

            # 2) prior based inference
            ym_hat, _, _ = model(xm)
            ym_hat_32 = _sanitize(ym_hat.to(torch.float32))

            # 3) manually unpack the NormCalc_cyc chain, but sanitize *every step*
            y_phys_m = _sanitize(y_tf.inverse(ym_hat_32), clip=None)
            if 'PARAM_LO' in globals() and 'PARAM_HI' in globals():
                y_phys_m = torch.max(torch.min(y_phys_m, PARAM_HI.to(y_phys_m.device)), PARAM_LO.to(y_phys_m.device))
            y_phys_m = y_phys_m.index_select(1, y_idx_c_from_p) if y_idx_c_from_p is not None else y_phys_m
            ym_proxy_norm = _sanitize(y_tf_proxy.transform(y_phys_m))
            xmh_proxy_std = _sanitize(proxy_g(ym_proxy_norm))
            xmh_phys = _sanitize(xmh_proxy_std * x_std_p + x_mu_p)
            xmh_curr_std = _sanitize((xmh_phys - x_mu_c) / x_std_c)

            # 4) per-sample validity mask (x and x̂ must both be finite)
            valid = torch.isfinite(xmh_curr_std).all(dim=1) & torch.isfinite(xm).all(dim=1)

            if valid.any():
                # use the same SmoothL1 algorithm as during training (beta=0.02),
                # but only calculate the mean for valid samples
                cyc_meas = cyc_crit(xmh_curr_std[valid], xm[valid])
            else:
                # this batch is entirely invalid
                # set it to 0 (you can also choose to skip this batch)
                cyc_meas = xm.new_tensor(0.0)

            valid_count = int(valid.sum().item())
            if valid_count > 0:
                cyc_meas_sum = float(cyc_meas.item()) * valid_count  # mean × 有效样本数 = 有效样本求和
                meter['cyc_meas'] += cyc_meas_sum
                meter['n_meas']   += valid_count

            # 5) diagnosis: Input x̂ of the current normalized domain,
            #  and also use the sanitized tensor.
            if diag_enabled and (y_tf is not None) and (y_tf_proxy is not None) and (y_idx_c_from_p is not None) and (diag_tag == "test"):
                domain_label = 'meas'
                diag_rows, diag_count = diag_processing(
                    model, proxy_g, device,
                    domain_label,
                    diag_rows, diag_count, diag_max, diag_k,
                    _sanitize(xm), _sanitize(xmh_curr_std), x_mu_p, x_std_p, x_mu_c, x_std_c,   
                    None, ym_hat, ym_hat_32, y_tf, y_tf_proxy, y_idx_c_from_p, yref_proxy_norm,
                    prior_bound, prior_bound_margin, PARAM_RANGE,
                    proxy_floor_all=proxy_floor_all
                )


    # record diag, if enabled while testing
    if diag_enabled and diag_outdir and diag_tag and len(diag_rows) > 0:
        os.makedirs(diag_outdir, exist_ok=True)
        path = os.path.join(diag_outdir, f"diag_{diag_tag}.csv")
        with open(path, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(diag_rows[0].keys()))
            w.writeheader()
            w.writerows(diag_rows)
        print(f"[Diag] wrote {len(diag_rows)} rows to {path}")

    # print proxy floor with percentiles btw
    if proxy_floor_all:
        arr = np.asarray(proxy_floor_all, dtype=np.float32)
        p50, p90, p95, p99 = np.percentile(arr, [50, 90, 95, 99]).tolist()
        print(f"[Diag] proxy_floor_ps  P50={p50:.4f}  P90={p90:.4f}  P95={p95:.4f}  P99={p99:.4f}")


    # record metrics, for both val/test
    n = meter['n']
    metrics = {
        # posterior 口径（ELBO一致）
        'val_sup_post':        meter['sup_post']/n,
        'val_cyc_sim_post':    meter['cyc_sim_post']/n,
        'val_prior_l2_post':   meter['prior_l2_post']/n,
        'val_prior_bnd_post':  meter['prior_bnd_post']/n,
        'val_kl':              meter['kl']/n,
        'val_total_post': (meter['sup_post']/n
                            + kl_beta * (meter['kl']/n)
                            + prior_l2 * (meter['prior_l2_post']/n)
                            + prior_bound * (meter['prior_bnd_post']/n)
                            + lambda_cyc_sim * (meter['cyc_sim_post']/n)
                            + lambda_cyc_meas * 0.0  # 若此函数只评 SIM，可按需加）
        ),
        # prior 口径（部署一致）—— 用这个做早停/对比 infer_cli
        'val_sup_prior':       meter['sup_prior']/n,
        'val_cyc_sim_prior':   meter['cyc_sim_prior']/n,
        'val_cyc_meas':        (meter['cyc_meas']/max(1, meter.get('n_meas', 0))) if meter.get('n_meas', 0) > 0 else 0.0,
        'val_prior_l2_prior':  meter['prior_l2_prior']/n,
        'val_prior_bnd_prior': meter['prior_bnd_prior']/n,
        'val_total_prior': (meter['sup_prior']/n
                            + prior_l2 * (meter['prior_l2_prior']/n)
                            + prior_bound * (meter['prior_bnd_prior']/n)
                            + lambda_cyc_sim * (meter['cyc_sim_prior']/n)
                            + lambda_cyc_meas * ((meter['cyc_meas']/max(1, meter.get('n_meas', 0))) if meter.get('n_meas', 0) > 0 else 0.0)
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
    assert X_all.ndim in (2, 3), f"Unexpected X_all.ndim={X_all.ndim}; expected 2 or 3."
    N = X_all.shape[0]
    if X_all.ndim == 3:
        x_dim  = X_all.shape[1] * X_all.shape[2]
        X_flat = X_all.reshape(N, -1).astype(np.float32)
    else:  # 已经是二维 (N, features)
        x_dim  = X_all.shape[1]
        X_flat = X_all.astype(np.float32)

    y_dim = Y_all.shape[1]

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
    _setup_print_tee(run_dir, "proxy.log")

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


def _apply_aug_schedule(ds: ArrayDataset, epoch: int, cfg: TrainConfig):
    if not hasattr(ds, "set_aug_scale"): return
    if cfg.aug_schedule == "none":
        ds.set_aug_scale(1.0); return
    # warmup 之后开始衰减
    e0 = int(cfg.cyc_warmup_epochs)
    if epoch <= e0:
        s = 1.0
    else:
        t = (epoch - e0) / max(1, (cfg.max_epochs - e0))
        t = np.clip(t, 0.0, 1.0)
        if cfg.aug_schedule == "linear_decay":
            s = 1.0 + (cfg.aug_final_scale - 1.0) * t
        elif cfg.aug_schedule == "cosine":
            # 从 1.0 → aug_final_scale 的余弦退火
            s = cfg.aug_final_scale + (1.0 - cfg.aug_final_scale) * 0.5 * (1 + math.cos(math.pi * t))
        else:
            s = 1.0
    ds.set_aug_scale(float(s))


def run_once(cfg: TrainConfig, diag_cfg: dict, device):
    set_seed(cfg.seed)
    train_ds, val_ds, test_ds, x_scaler, y_tf, splits, X_all, Y_all = load_and_prepare(cfg.data, cfg)
    train_loader, val_loader, test_loader = make_loaders(train_ds, val_ds, test_ds, cfg.batch_size, cfg.num_workers)

    x_dim, y_dim = train_ds.x.shape[1], train_ds.y.shape[1]
    
    model = CVAE(x_dim, y_dim, list(cfg.hidden), cfg.latent_dim, cfg.dropout).to(device)
    if cfg.compile and hasattr(torch, 'compile'):
        try: model = torch.compile(model)
        except Exception as e: print(f"Torch compile failed: {e}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=cfg.lr, epochs=cfg.onecycle_epochs, steps_per_epoch=len(train_loader)) if cfg.use_onecycle else None
    scaler = torch.amp.GradScaler('cuda', enabled=(device.type=='cuda'))

    stamp = time.strftime('%Y%m%d-%H%M%S')
    run_dir = os.path.join(cfg.outdir, f'cvae_{stamp}')
    writer = SummaryWriter(log_dir=run_dir)
    _setup_print_tee(run_dir, "train.log")

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
        # 让在线噪声按 epoch 调度
        _apply_aug_schedule(train_loader.dataset, epoch, cfg)
        lam_meas = cfg.lambda_cyc_meas * min(1.0, epoch / max(1, cfg.cyc_warmup_epochs))
        
        K_eff, use_bok_sim, use_bok_meas = _bok_flags(cfg, phase='train', epoch=epoch)
        train_metrics = train_one_epoch(
            model, train_loader, optimizer, scaler, device,
            scheduler=scheduler, current_epoch=epoch, onecycle_epochs=cfg.onecycle_epochs, kl_beta=cfg.kl_beta,
            y_tf=y_tf, proxy_g=proxy_g, lambda_cyc_sim=cfg.lambda_cyc_sim, meas_loader=meas_loader,
            lambda_cyc_meas=lam_meas, y_tf_proxy=y_tf_proxy, x_mu_c=x_mu_c, x_std_c=x_std_c,
            x_mu_p=x_mu_p, x_std_p=x_std_p, y_idx_c_from_p=y_idx_c_from_p, sup_weight=cfg.sup_weight,
            prior_l2=cfg.prior_l2, prior_bound=cfg.prior_bound, prior_bound_margin=cfg.prior_bound_margin,
            trust_alpha=cfg.trust_alpha, trust_tau=cfg.trust_tau, yref_proxy_norm=yref_proxy_norm,
            trust_ref_batch=cfg.trust_ref_batch, trust_alpha_meas=cfg.trust_alpha_meas,
            cyc_meas_knn_weight=cfg.cyc_meas_knn_weight, cyc_meas_knn_gamma=cfg.cyc_meas_knn_gamma,
            z_sample_mode=cfg.z_sample_mode,
            best_of_k=K_eff,
            bok_use_sim=use_bok_sim,
            bok_use_meas=use_bok_meas,
        )

        K_eff_val, use_bok_sim_val, use_bok_meas_val = _bok_flags(cfg, phase='val', epoch=epoch)
        val_metrics = evaluate_full(
            model, val_loader, device, y_tf=y_tf, proxy_g=proxy_g, lambda_cyc_sim=cfg.lambda_cyc_sim,
            meas_loader=meas_loader, lambda_cyc_meas=cfg.lambda_cyc_meas, y_tf_proxy=y_tf_proxy,
            x_mu_c=x_mu_c, x_std_c=x_std_c, x_mu_p=x_mu_p, x_std_p=x_std_p,
            y_idx_c_from_p=y_idx_c_from_p, sup_weight=cfg.sup_weight, kl_beta=cfg.kl_beta,
            prior_l2=cfg.prior_l2, prior_bound=cfg.prior_bound, prior_bound_margin=cfg.prior_bound_margin,
            enforce_bounds=cfg.enforce_bounds, diag_cfg=diag_cfg, yref_proxy_norm=yref_proxy_norm,
            diag_outdir=run_dir, diag_tag=f"val_ep{epoch:03d}",
            z_sample_mode=cfg.z_sample_mode,
            dropout_in_eval=getattr(cfg, 'dropout_val', False),
            best_of_k=K_eff_val,
            bok_use_sim=use_bok_sim_val,
            bok_use_meas=use_bok_meas_val
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

    K_eff_test, use_bok_sim_test, use_bok_meas_test = _bok_flags(cfg, phase='test', epoch=epoch)
    test_metrics = evaluate_full(model, test_loader, device, y_tf=y_tf, proxy_g=proxy_g, lambda_cyc_sim=cfg.lambda_cyc_sim,
                                meas_loader=meas_loader, lambda_cyc_meas=cfg.lambda_cyc_meas, y_tf_proxy=y_tf_proxy,
                                x_mu_c=x_mu_c, x_std_c=x_std_c, x_mu_p=x_mu_p, x_std_p=x_std_p, y_idx_c_from_p=y_idx_c_from_p,
                                sup_weight=cfg.sup_weight, kl_beta=cfg.kl_beta, prior_l2=cfg.prior_l2, prior_bound=cfg.prior_bound,
                                prior_bound_margin=cfg.prior_bound_margin, enforce_bounds=cfg.enforce_bounds,
                                diag_cfg=diag_cfg, yref_proxy_norm=yref_proxy_norm, diag_outdir=run_dir, diag_tag="test",
                                z_sample_mode=cfg.z_sample_mode,
                                dropout_in_eval=getattr(cfg, 'dropout_test', False),
                                best_of_k=K_eff_test,
                                bok_use_sim=use_bok_sim_test,
                                bok_use_meas=use_bok_meas_test
                                )

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
def load_cvae_artifacts(run_dir: str, device):
    tr_path = os.path.join(run_dir, 'transforms.json')
    md_path = os.path.join(run_dir, 'best_model.pt')
    assert os.path.isfile(tr_path), f"transforms.json not found: {tr_path}"
    assert os.path.isfile(md_path), f"best_model.pt not found: {md_path}"

    with open(tr_path, 'r') as f: meta = json.load(f)
    x_scaler = XStandardizer.from_state_dict(meta['x_scaler'])
    y_tf = YTransform.from_state_dict(meta['y_transform'])
    cfg = meta['config']
    
    model = CVAE(x_dim=meta['input_dim'], y_dim=len(y_tf.names),
                 hidden=cfg['hidden'], latent_dim=cfg['latent_dim'], dropout=cfg['dropout']).to(device)
    
    ckpt = torch.load(md_path, map_location=device)
    model.load_state_dict(ckpt['model'])
    model.eval()
    return model, x_scaler, y_tf, meta

def infer_cli(args, device):
    model, x_scaler, y_tf, meta = load_cvae_artifacts(args.infer_run, device)
    
    if args.input_npy: X = np.load(args.input_npy)
    elif args.input_h5:
        with h5py.File(args.input_h5, 'r') as f: X = f['X'][...]
        if args.index is not None and X.ndim >= 3: X = X[int(args.index)]
    else: raise ValueError("Provide --input-npy or --input-h5")

    x = X.reshape(1, -1) if X.ndim < 3 else X.reshape(X.shape[0], -1)
    x_std = x_scaler.transform(x)
    with torch.no_grad():
        # --- FIX START: Explicitly set dtype to torch.float32 ---
        # Original line that caused the error:
        # xt = torch.from_numpy(x_std).to(device)
        # Corrected line:
        xt = torch.tensor(x_std, dtype=torch.float32, device=device)
        # --- FIX END ---

        # The model is in eval mode from load_artifacts
        with dropout_mode(model, enabled=bool(getattr(args, 'dropout_infer', False))):
            pred_norm = model.sample(xt, num_samples=args.num_samples, sample_mode=args.sample_mode)

        # pred_norm_samples shape: [num_samples, N, y_dim]
        # Reshape for inverse transform: [num_samples * N, y_dim]

    # transform back to physical space
    S, B, Dy = pred_norm.shape
    pred_phys = y_tf.inverse(pred_norm.reshape(-1, Dy)).reshape(S, B, Dy)

    sample_idx = 0
    solutions = pred_phys[:, sample_idx, :] # Shape: [num_samples, y_dim]
    mean_sol = solutions.mean(axis=0)
    std_sol = solutions.std(axis=0)
    
    print("\n--- Solution Statistics (for first input) ---")
    w = max(len(n) for n in y_tf.names)
    for i, name in enumerate(y_tf.names):
        print(f"  {name:<{w}} : mean={mean_sol[i]:.4g}  std={std_sol[i]:.4g}")

    if args.save_csv:
         # save inference result (predicted parameters)
        with open(args.save_csv, 'w', newline='') as f:
            wcsv = csv.writer(f)
            header = ['input_idx', 'sample_idx'] + y_tf.names
            wcsv.writerow(header)
            for i in range(pred_phys.shape[1]): # Loop over inputs
                for j in range(pred_phys.shape[0]): # Loop over samples
                    row = [i, j] + pred_phys[j, i, :].tolist()
                    wcsv.writerow(row)
        print(f"\nSaved all {args.num_samples * xt.size(0)} sampled solutions to {args.save_csv}")
        # save inference result (mean + std)
        path2 = args.save_csv[:-4] + '_mean_std.csv'
        with open(path2, 'w', newline='') as f:
            wcsv = csv.writer(f)
            header = ['name', 'mean', 'std']
            wcsv.writerow(header)
            for i, name in enumerate(y_tf.names):
                row = [name, f"{mean_sol[i]:.4g}", f"{std_sol[i]:.4g}"]
                wcsv.writerow(row)
        print(f"Saved inferenced statistical feature to {path2}")



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
    
    # add noise in X
    p.add_argument('--aug-gain-std', type=float, default=0.0, help='overall multiplicative gain is LogNormal(0, aug_gain_std)')
    p.add_argument('--aug-row-gain-std', type=float, default=0.0, help='row/curve level multiplicative gain (mesh shape required)')
    p.add_argument('--aug-smooth-window', type=int, default=0, help='window length (along the column direction) for 1D sliding smoothing of additive noise')
    p.add_argument('--aug-schedule', type=str, default='none', choices=["none", "linear_decay", "cosine"])
    p.add_argument('--aug-final-scale', type=float, default=0.0, help='end-of-schedule Noise Scaling Factor')

    # New/Modified CVAE arguments
    p.add_argument('--hidden', type=str, default='512,256', help='Hidden layers for CVAE components')
    p.add_argument('--latent-dim', type=int, default=32, help='Dimension of the latent space z')
    p.add_argument('--dropout', type=float, default=0.1)
    p.add_argument('--kl-beta', type=float, default=0.1, help='Weight for the KL divergence term in the loss')

    # Best of K in prior path (towards cyc)
    p.add_argument('--best-of-k', type=int, default=0, help='K for Best-of-K on prior path (0/1 disables).')
    p.add_argument('--bok-warmup-epochs', type=int, default=0, help='Warm-up epochs to ramp K from 1 to best-of-k.')
    p.add_argument('--bok-target', choices=['sim','meas','both','none'], default='sim', help='Apply BoK to which cycle: sim / meas / both / none.')
    p.add_argument('--bok-apply', type=str, default='train', help='Comma list among {train,val,test} where BoK is used.')

    # Inference arguments
    p.add_argument('--infer-run', type=str, help='Run dir for inference')
    p.add_argument('--input-npy', type=str, help='Input .npy file for inference')
    p.add_argument('--input-h5', type=str, help='Input .h5 file for inference')
    p.add_argument('--index', type=int, help='Sample index for .h5 input')
    p.add_argument('--save-csv', type=str, help='Path to save inference results')
    p.add_argument('--num-samples', type=int, default=1, help='Number of solutions to sample during inference')
    p.add_argument('--sample-mode', type=str, default='mean', choices=['rand', 'mean'], help='Sampling mode for inference')
    p.add_argument('--z-sample-mode', type=str, default='mean', choices=['rand', 'mean'], help='Sampling mode of latent space for training and testing')
    p.add_argument('--dropout-val', action='store_true', help='Enable dropout during validation') # normally off
    p.add_argument('--dropout-test', action='store_true', help='Enable dropout during test')      # normally off
    p.add_argument('--dropout-infer', action='store_true', help='Enable dropout during inference')# optional



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
        aug_gain_std=args.aug_gain_std,           
        aug_row_gain_std=args.aug_row_gain_std,
        aug_smooth_window=args.aug_smooth_window,     
        aug_schedule=args.aug_schedule,        
        aug_final_scale=args.aug_final_scale,   
        dropout_val=args.dropout_val,
        dropout_test=args.dropout_test,
        dropout_infer=args.dropout_infer,
        best_of_k = args.best_of_k,
        bok_warmup_epochs = args.bok_warmup_epochs,
        bok_target = args.bok_target,
        bok_apply = args.bok_apply,
    )
    diag_cfg = {'enable': args.diag, 'max_samples': args.diag_max_samples, 'knn_k': args.diag_knn_k}
    return cfg, args, diag_cfg


def main():
    cfg, args, diag_cfg = parse_args()

    os.environ.setdefault("PYTHONIOENCODING", "utf-8")
    os.environ.setdefault("PYTHONUTF8", "1")
    for _s in (sys.stdout, sys.stderr):
        try:
            _s.reconfigure(encoding="utf-8")  
        except Exception:
            pass
    _setup_print_tee(cfg.outdir, f"session_{time.strftime('%Y%m%d-%H%M%S')}.log")

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


