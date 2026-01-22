# utils.py
import os, io, sys, json, atexit, time, math
from contextlib import contextmanager
from datetime import datetime
from typing import Optional, Dict, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
import re, glob


# ============================
# 0) Print → Console + File (tee)   (from your original)
# ============================
class _Tee(io.TextIOBase):
    def __init__(self, *streams):
        self.streams = list(streams)
        enc = None
        for st in self.streams:
            enc = getattr(st, "encoding", None)
            if enc:
                break
        self._encoding = enc or "utf-8"

    @property
    def encoding(self): return self._encoding
    @property
    def errors(self): return "strict"

    def write(self, s):
        if not isinstance(s, str):
            s = str(s)
        wrote = 0
        for st in self.streams:
            try:
                n = st.write(s)
                if isinstance(n, int): wrote = max(wrote, n)
                st.flush()
            except UnicodeEncodeError:
                enc = getattr(st, "encoding", None) or "ascii"
                safe = s.encode(enc, errors="replace").decode(enc, errors="replace")
                try:
                    n2 = st.write(safe)
                    if isinstance(n2, int): wrote = max(wrote, n2)
                    st.flush()
                except Exception:
                    pass
            except Exception:
                pass
        return wrote

    def flush(self):
        for st in self.streams:
            try: st.flush()
            except Exception: pass

    def add_stream(self, stream):
        if stream not in self.streams:
            self.streams.append(stream)


def _setup_print_tee(out_dir: str, filename: str):
    os.makedirs(out_dir, exist_ok=True)
    log_path = os.path.join(out_dir, filename)
    fh = open(log_path, mode="a", encoding="utf-8", buffering=1)

    if isinstance(sys.stdout, _Tee):
        if fh not in sys.stdout.streams:
            sys.stdout.add_stream(fh)
        if fh not in sys.stderr.streams:
            sys.stderr.add_stream(fh)
    else:
        sys.stdout = _Tee(sys.stdout, fh)
        sys.stderr = _Tee(sys.stderr, fh)

    atexit.register(lambda: (fh.flush(), fh.close()))
    print(f"[log] tee initialized at {datetime.now():%Y-%m-%d %H:%M:%S} -> {log_path}")


# ============================
# 0) Dropout on demand (val/test/infer)
# ============================
@contextmanager
def dropout_mode(model: nn.Module, enabled: bool = False):
    if not enabled:
        yield
        return
    train_modes = {}
    try:
        for m in model.modules():
            if isinstance(m, nn.Dropout):
                train_modes[m] = m.training
                m.train(True)
        yield
    finally:
        for m, st in train_modes.items():
            m.train(st)


# ============================
# 0) hparams record (same behavior as your original)
# ============================
def add_hparams_safe(base_writer: SummaryWriter, run_dir: str, hparams: dict, metrics: dict):
    hp_dir = os.path.join(run_dir, "hparams")
    os.makedirs(hp_dir, exist_ok=True)
    try:
        with open(os.path.join(hp_dir, "hparams.json"), "w", encoding="utf-8") as f:
            json.dump({"hparams": hparams, "metrics": metrics}, f, indent=2)
    except Exception as e:
        print(f"[warn] writing hparams.json failed: {e}")


# ============================
# Best-of-K warmup & flags (same interface)
# ============================
def _bok_flags(cfg, phase: str, epoch: int):
    """
    Returns:
      K_eff, bok_use_sim, bok_use_meas
    phase in {'train','val','test'}
    """
    apply_set = set([s.strip().lower() for s in (cfg.bok_apply or "train").split(",") if s.strip()])
    use_here = (phase.lower() in apply_set)

    if (not use_here) or (cfg.best_of_k is None) or (cfg.best_of_k <= 1):
        return 1, False, False

    if cfg.bok_warmup_epochs and cfg.bok_warmup_epochs > 0:
        W = int(cfg.bok_warmup_epochs)
        if epoch <= W:
            K_eff = 1 + int((cfg.best_of_k - 1) * (epoch / max(1, W)))
        else:
            K_eff = cfg.best_of_k
    else:
        K_eff = cfg.best_of_k

    targ = (cfg.bok_target or "sim").lower()
    bok_use_sim  = (targ in ("sim","both"))
    bok_use_meas = (targ in ("meas","both"))
    return max(1, int(K_eff)), bok_use_sim, bok_use_meas



def get_best_val_from_dir(resume_path: str) -> float:
    """
    From the directory where the resume weights are located, 
    find the unique .log file and extract the last occurrence of the Best value.
    """
    # get path
    run_dir = os.path.dirname(resume_path)
    # Use glob to search for all .log files in this directory
    # os.path.join(run_dir, "*.log") will match any file 
    # in this directory that ends with .log
    log_files = glob.glob(os.path.join(run_dir, "*.log"))
    # checik if we found any .log files
    if not log_files:
        print(f"[Warn] No .log file found in directory: {run_dir}. Resetting best_val to inf.")
        return float('inf')
    target_log_path = log_files[0]
    print(f"[Info] Found log file: {os.path.basename(target_log_path)}")

    best_val = float('inf')
    # Reverse search
    try:
        with open(target_log_path, 'r', encoding='utf-8', errors='ignore') as f:
            lines = f.readlines()
        for line in reversed(lines):
            # Regular expression matching pattern: "| Best 0.0332 |"
            # \s* tolerates whitespace, ([\d\.]+) captures floating-point numbers
            match = re.search(r'\|\s*Best\s+([\d\.]+)\s*\|', line)
            if match:
                best_val = float(match.group(1))
                print(f"[Info] Resuming Best Metric from log: {best_val}")
                return best_val
    except Exception as e:
        print(f"[Warn] Failed to parse log file {target_log_path}: {e}")
    print("[Info] Could not find valid 'Best' value in log, resetting to inf.")
    return best_val