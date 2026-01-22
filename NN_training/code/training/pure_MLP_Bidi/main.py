import os, json, time, argparse, sys
from dataclasses import dataclass, asdict
from typing import Tuple, Optional, List

import numpy as np
import h5py
import torch
from torch.utils.tensorboard import SummaryWriter

from data import (
    load_and_prepare_dual, make_loaders, 
    XStandardizer, YTransform
)
# 引用上面的 pure MLP 模型
from models import DualInputMLP
# 引用 pure MLP 的训练循环 (假设 training.py 已包含 train_one_epoch_mlp 和 evaluate_mlp)
from training import train_one_epoch_mlp, evaluate_mlp
from utils import _setup_print_tee, add_hparams_safe, get_best_val_from_dir

PARAM_NAMES = [
    'VOFF', 'U0', 'NS0ACCS', 'NFACTOR', 'ETA0',
    'VSAT', 'VDSCALE', 'CDSCD', 'LAMBDA', 'MEXPACCD', 'DELTA'
]
# 这里的 RANGE 仅用于数据预处理（log mask），不用于 CVAE 的物理边界 loss
PARAM_RANGE = {
    'VOFF': (-4.7, -3), 'U0': (0.1, 2.2), 'NS0ACCS': (1e15, 1e20),
    'NFACTOR': (0, 10), 'ETA0': (0, 1), 'VSAT': (5e4, 1e7),
    'VDSCALE': (0.5, 1e6), 'CDSCD': (0, 0.75), 'LAMBDA': (0, 0.2),
    'MEXPACCD': (0, 12), 'DELTA': (2, 100)
}

def set_seed(seed=42):
    import random
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = True

@dataclass
class TrainConfig:
    data: Optional[str] = None
    outdir: str = "runs_mlp_baseline"
    seed: int = 42
    test_split: float = 0.15
    val_split: float = 0.15
    
    # === 新增/修改的接口 ===
    max_epochs: int = 300       # 对应 --max-epochs
    onecycle_epochs: int = 300  # 对应 --onecycle-epochs
    patience: int = 100         # 对应 --patience
    dropout: float = 0.1        # 对应 --dropout
    weight_decay: float = 1e-4  # 对应 --weight-decay
    # =====================

    batch_size: int = 512
    lr: float = 1e-4
    num_workers: int = 0
    compile: bool = False
    use_onecycle: bool = True

    # MLP Architecture
    hidden: Tuple[int, ...] = (512, 256, 128)
    feat_dim: int = 256

    # Augmentation
    aug_noise_std: float = 0.015
    aug_prob: float = 0.5
    aug_gain_std: float = 0.0
    
    es_metric: str = 'val_total_post'
    es_min_delta: float = 1e-6
    
    resume: Optional[str] = None

# ... (imports and run_once function above) ...

def load_mlp_artifacts(run_dir: str, device):
    """
    加载训练好的 MLP 模型及预处理工具
    """
    tr_path = os.path.join(run_dir, 'transforms.json')
    md_path = os.path.join(run_dir, 'best_model.pt')
    
    if not os.path.isfile(tr_path):
        raise FileNotFoundError(f"transforms.json not found in {run_dir}")
    if not os.path.isfile(md_path):
        raise FileNotFoundError(f"best_model.pt not found in {run_dir}")

    with open(tr_path, 'r') as f:
        meta = json.load(f)
    
    # 恢复标准化器
    x_scaler = XStandardizer.from_state_dict(meta['x_scaler'])
    y_tf = YTransform.from_state_dict(meta['y_transform'])
    cfg_dict = meta['config']
    
    # 初始化模型结构
    # 注意：必须使用保存时的配置来初始化
    model = DualInputMLP(
        y_dim=len(y_tf.names),
        hidden=cfg_dict["hidden"],
        feat_dim=cfg_dict.get("feat_dim", 256),
        dropout=cfg_dict.get("dropout", 0.0), # 加载时的 dropout 仅用于结构占位，eval模式下不起作用
        iv_shape=tuple(meta["input_meta"]["iv_shape"]),
        gm_shape=tuple(meta["input_meta"]["gm_shape"])
    ).to(device)

    # 加载权重
    ckpt = torch.load(md_path, map_location=device)
    model.load_state_dict(ckpt['model'])
    model.eval()
    
    return model, x_scaler, y_tf, meta

def infer_cli(args, device):
    """
    命令行推理主函数 (Pure MLP 版本)
    """
    # 1. 加载模型
    model, x_scaler, y_tf, meta = load_mlp_artifacts(args.infer_run, device)
    
    iv_shape = tuple(meta["input_meta"]["iv_shape"])
    gm_shape = tuple(meta["input_meta"]["gm_shape"])
    L_iv = int(np.prod(iv_shape))

    # 2. 加载输入数据
    if args.input_npy:
        if args.input_npy.endswith(".npz"):
            z = np.load(args.input_npy)
            X_iv = z["X_iv"]
            X_gm = z["X_gm"]
        else:
            raise ValueError("Input .npy must be .npz format with keys 'X_iv' and 'X_gm'")
    elif args.input_h5:
        with h5py.File(args.input_h5, "r") as f:
            X_iv = f["X_iv"][...]
            X_gm = f["X_gm"][...]
        # 如果指定了 index，只推理某一个样本
        if args.index is not None:
            idx = int(args.index)
            X_iv = X_iv[idx:idx+1]
            X_gm = X_gm[idx:idx+1]
    else:
        raise ValueError("Please provide --input-npy or --input-h5")
    
    # 3. 数据预处理
    N = X_iv.shape[0]
    print(f"[Infer] Processing {N} samples...")
    
    # Flatten & Standardize
    X_concat = np.concatenate([X_iv.reshape(N,-1), X_gm.reshape(N,-1)], axis=1).astype(np.float32)
    X_std = x_scaler.transform(X_concat)
    
    # Reshape back to curve format for CNN/MLP Encoder
    X_iv_std = X_std[:, :L_iv].reshape(N, *iv_shape)
    X_gm_std = X_std[:, L_iv:].reshape(N, *gm_shape)

    xt_iv = torch.tensor(X_iv_std, dtype=torch.float32, device=device)
    xt_gm = torch.tensor(X_gm_std, dtype=torch.float32, device=device)

    # 4. 模型推理
    with torch.no_grad():
        # Pure MLP 是确定性的，直接输出 normalized parameters
        # 输出形状: [N, y_dim]
        pred_norm = model(xt_iv, xt_gm)
    
    # 5. 反归一化 -> 物理数值
    pred_phys = y_tf.inverse(pred_norm).cpu().numpy()

    # 6. 打印结果 (打印第一个样本详情)
    print("\n--- Prediction Result (First Sample) ---")
    w = max(len(n) for n in y_tf.names)
    for i, name in enumerate(y_tf.names):
        val = pred_phys[0, i]
        print(f"  {name:<{w}} : {val:.4g}")

    # 7. 保存结果到 CSV
    if args.save_csv:
        import csv
        # 自动创建目录
        os.makedirs(os.path.dirname(os.path.abspath(args.save_csv)), exist_ok=True)
        
        with open(args.save_csv, 'w', newline='') as f:
            wcsv = csv.writer(f)
            # Header: index + param_names
            header = ['sample_idx'] + y_tf.names
            wcsv.writerow(header)
            
            for i in range(N):
                row = [i] + pred_phys[i].tolist()
                wcsv.writerow(row)
        print(f"\n[Done] Saved predictions to {args.save_csv}")


def save_state(outdir: str, x_scaler: XStandardizer, y_tf: YTransform, cfg: TrainConfig, meta: dict):
    os.makedirs(outdir, exist_ok=True)
    dd = {
        'x_scaler': x_scaler.state_dict(),
        'y_transform': y_tf.state_dict(),
        'config': asdict(cfg),
        'param_names': PARAM_NAMES,
        'input_meta': meta
    }
    with open(os.path.join(outdir, 'transforms.json'), 'w') as f:
        json.dump(dd, f, indent=2)

def run_once(cfg: TrainConfig, device):
    set_seed(cfg.seed)
    # 加载数据
    train_ds, val_ds, test_ds, x_scaler, y_tf, splits, X_all, Y_all, meta = \
        load_and_prepare_dual(cfg.data, cfg, PARAM_NAMES, PARAM_RANGE)
    train_loader, val_loader, test_loader = make_loaders(train_ds, val_ds, test_ds, cfg.batch_size, cfg.num_workers)

    # 初始化纯 MLP 模型
    # 这里将 cfg.dropout 传入模型
    model = DualInputMLP(
        y_dim=len(y_tf.names),
        hidden=list(cfg.hidden),
        feat_dim=cfg.feat_dim,
        dropout=cfg.dropout,  # <--- 使用传入的 dropout 参数
        iv_shape=tuple(meta["iv_shape"]),
        gm_shape=tuple(meta["gm_shape"])
    ).to(device)

    if cfg.compile and hasattr(torch, 'compile'):
        try: model = torch.compile(model)
        except Exception: pass
    
    # 配置优化器
    # 这里将 cfg.weight_decay 传入
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    
    # 配置调度器
    # 这里使用 cfg.onecycle_epochs
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=cfg.lr, epochs=cfg.onecycle_epochs, steps_per_epoch=len(train_loader)
    ) if cfg.use_onecycle else None
    
    scaler = torch.cuda.amp.GradScaler(enabled=(device.type=='cuda'))

    stamp = time.strftime('%Y%m%d-%H%M%S')
    run_dir = os.path.join(cfg.outdir, f'mlp_{stamp}')
    writer = SummaryWriter(log_dir=run_dir)
    _setup_print_tee(run_dir, "train.log")

    save_state(run_dir, x_scaler, y_tf, cfg, meta)

    best_val, no_improve = float('inf'), 0
    best_path = os.path.join(run_dir, 'best_model.pt')

    if cfg.resume and os.path.isfile(cfg.resume):
        print(f"[Info] Resuming from {cfg.resume}")
        ckpt = torch.load(cfg.resume, map_location=device)
        model.load_state_dict(ckpt['model'])
        prev_best = get_best_val_from_dir(cfg.resume)
        if prev_best < float('inf'): best_val = prev_best

    # 训练循环
    # 使用 cfg.max_epochs
    for epoch in range(1, cfg.max_epochs + 1):
        # 训练一个 epoch
        train_metrics = train_one_epoch_mlp(
            model, train_loader, optimizer, scaler, device, 
            scheduler=scheduler
        )

        # 验证
        val_metrics = evaluate_mlp(
            model, val_loader, device, 
            dropout_in_eval=False # 通常验证时不开启 dropout，除非是为了不确定性估计
        )

        for k,v in train_metrics.items(): writer.add_scalar(f"train/{k}", v, epoch)
        for k,v in val_metrics.items(): writer.add_scalar(f"val/{k}", v, epoch)

        print(f" >> Epoch {epoch:03d}/{cfg.max_epochs} | Train Loss={train_metrics['total']:.6f} | "
              f"Val Loss={val_metrics['val_total_post']:.6f} | "
              f"Best {best_val:.6f} | Patience {no_improve}/{cfg.patience}")

        # 早停逻辑
        # 使用 cfg.patience
        es_value = val_metrics[cfg.es_metric]
        if es_value < best_val - cfg.es_min_delta:
            best_val = es_value
            no_improve = 0
            torch.save({'model': model.state_dict()}, best_path)
            print(f"[Update] New best model saved.")
        else:
            no_improve += 1
            if no_improve >= cfg.patience:
                print(f"[EarlyStop] Patience {cfg.patience} reached at epoch {epoch}")
                break

    # 最终测试
    if os.path.isfile(best_path):
        ckpt = torch.load(best_path, map_location=device)
        model.load_state_dict(ckpt['model'])
    
    test_metrics = evaluate_mlp(model, test_loader, device)
    print(f"[Test] Loss={test_metrics['val_total_post']:.6f}")

    add_hparams_safe(writer, run_dir, {"tag": "final_mlp"}, test_metrics)
    writer.close()


def parse_args():
    p = argparse.ArgumentParser(description='Pure MLP Baseline Training & Inference')
    
    # === Training Arguments ===
    p.add_argument('--data', type=str, help='Path to H5 data (for training)')
    p.add_argument('--outdir', type=str, default='runs_mlp_baseline')
    
    # Hyperparameters
    p.add_argument('--dropout', type=float, default=0.1, help='Dropout rate')
    p.add_argument('--weight-decay', type=float, default=1e-4, help='Weight decay')
    p.add_argument('--max-epochs', type=int, default=300)
    p.add_argument('--onecycle-epochs', type=int, default=300)
    p.add_argument('--patience', type=int, default=100)
    
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--batch-size', type=int, default=512)
    p.add_argument('--lr', type=float, default=1e-4)
    p.add_argument('--hidden', type=str, default='512,256,128')
    p.add_argument('--feat-dim', type=int, default=256)
    p.add_argument('--resume', type=str)
    
    # Augmentation
    p.add_argument('--aug-prob', type=float, default=0.5)
    p.add_argument('--aug-noise-std', type=float, default=0.015)
    p.add_argument('--aug-gain-std', type=float, default=0.0)
    p.add_argument('--num-workers', type=int, default=0)

    # === Inference Arguments (NEW) ===
    p.add_argument('--infer-run', type=str, help='Path to training run folder (containing best_model.pt)')
    p.add_argument('--input-h5', type=str, help='Path to H5 input file for inference')
    p.add_argument('--input-npy', type=str, help='Path to .npz input file for inference')
    p.add_argument('--index', type=int, help='Index of sample to infer (optional)')
    p.add_argument('--save-csv', type=str, help='Path to save result csv')
    
    args = p.parse_args()

    # Config 创建逻辑 (仅用于训练模式，推理模式下可以忽略 Config 的部分字段)
    cfg = TrainConfig(
        data=args.data, outdir=args.outdir, seed=args.seed,
        dropout=args.dropout, weight_decay=args.weight_decay,
        max_epochs=args.max_epochs, onecycle_epochs=args.onecycle_epochs,
        patience=args.patience,
        batch_size=args.batch_size, lr=args.lr,
        hidden=tuple(map(int, args.hidden.split(','))),
        feat_dim=args.feat_dim,
        aug_prob=args.aug_prob, aug_noise_std=args.aug_noise_std, aug_gain_std=args.aug_gain_std,
        resume=args.resume
    )
    return cfg, args

if __name__ == '__main__':
    cfg, args = parse_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    if args.infer_run:
        # 如果提供了 --infer-run，则进入推理模式
        infer_cli(args, device)
    else:
        # 否则进入训练模式 (需要检查 --data)
        if cfg.data is None:
            print("Error: Training mode requires --data.")
            sys.exit(1)
        run_once(cfg, device)