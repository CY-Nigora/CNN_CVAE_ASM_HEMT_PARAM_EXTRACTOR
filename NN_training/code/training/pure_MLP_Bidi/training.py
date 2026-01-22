# training.py
from typing import Optional, Literal, List, Dict, Tuple
import os, csv
import numpy as np
import torch
import torch.nn as nn
from utils import dropout_mode

# ----------------------------
# Criterion
# ----------------------------
class CriterionWrapper:
    def __init__(self, beta: float = 0.02):
        self.crit = nn.SmoothL1Loss(beta=beta, reduction="none")

    def __call__(self, y_hat, y):
        return self.crit(y_hat, y).mean()

# ----------------------------
# MLP Training Loop (Baseline)
# ----------------------------
def train_one_epoch_mlp(
    model, loader, optimizer, scaler, device, 
    scheduler=None, 
    # Keep interfaces to allow main.py to pass args without crashing
    **kwargs 
):
    model.train()
    criterion = CriterionWrapper(beta=0.02)
    
    meter = dict(n=0, total=0.0)

    for x_iv, x_gm, y in loader:
        x_iv = x_iv.to(device, non_blocking=True)
        x_gm = x_gm.to(device, non_blocking=True)
        y    = y.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        with torch.cuda.amp.autocast(enabled=(device.type=='cuda')):
            # Forward
            y_pred = model(x_iv, x_gm)
            
            # Loss (Pure Supervised)
            loss = criterion(y_pred, y)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        
        if scheduler is not None:
            scheduler.step()

        bs = x_iv.size(0)
        meter["n"] += bs
        meter["total"] += float(loss.item()) * bs

    n = max(1, meter["n"])
    return {"total": meter["total"]/n}

# ----------------------------
# MLP Evaluation Loop (Baseline)
# ----------------------------
@torch.no_grad()
def evaluate_mlp(
    model, loader, device, 
    dropout_in_eval: bool = False,
    # Keep interfaces to allow main.py to pass args without crashing
    **kwargs
):
    model.eval()
    criterion = CriterionWrapper(beta=0.02)
    meter = dict(n=0, total=0.0)

    with dropout_mode(model, enabled=dropout_in_eval):
        for x_iv, x_gm, y in loader:
            x_iv = x_iv.to(device)
            x_gm = x_gm.to(device)
            y    = y.to(device)

            y_pred = model(x_iv, x_gm)
            loss = criterion(y_pred, y)

            bs = x_iv.size(0)
            meter["n"] += bs
            meter["total"] += float(loss.item()) * bs

    n = max(1, meter["n"])
    # Return metrics dict with keys compatible with main.py logging
    return {
        "val_total_post": meter["total"]/n, 
        # Map same value to other keys if downstream requires them, 
        # but for MLP, post/prior separation doesn't exist.
        "val_sup_post": meter["total"]/n 
    }