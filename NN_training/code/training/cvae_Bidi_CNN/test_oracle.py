import torch
import numpy as np
import h5py
import os
import json
import argparse
from proxy import load_dual_proxy_artifacts
from data import XStandardizer, YTransform

# ==========================================
# 1. 定义绝对物理边界 (Strict Hard Constraints)
# ==========================================
# 这里的范围必须是你希望模型绝对遵守的界限
PARAM_RANGE = {
    # 'VOFF': (-4.7, -4),
    'VOFF': (-100, 5),
    'U0': (0.1, 2.2),
    'NS0ACCS': (1e15, 1e20),
    'NFACTOR': (0, 10),
    'ETA0': (0, 1),
    'VSAT': (5e4, 1e7),
    'VDSCALE': (0.5, 1e6),
    'CDSCD': (0, 0.75),
    'LAMBDA': (0, 0.2),
    'MEXPACCD': (0, 12),
    'DELTA': (2, 100)
    # 'UA': (0, 1e-8),
    # 'UB': (0, 1e-16),
    # 'U0ACCS': (0.01, 0.4)
}

PROXY_RUN = "E:/personal_Data/Document of School/Uni Stuttgart/Masterarbeit/Code/param_regression/ADS_Parameter_Fitting/IV_param_regression/NN_training/temp_Bidi_cvae_11param_2channel/proxy_dual" 
CONFIG_SOURCE_RUN = "E:/personal_Data/Document of School/Uni Stuttgart/Masterarbeit/Code/param_regression/ADS_Parameter_Fitting/IV_param_regression/NN_training/temp_Bidi_cvae_11param_2channel/version_1_1"
MEAS_FILE = "E:/personal_Data/Document of School/Uni Stuttgart/Masterarbeit/Code/param_regression/ADS_Parameter_Fitting/IV_param_regression/NN_training/dataset/training/meas_smoothed_Bidi_2Channel.h5"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SAMPLE_IDX = 0 # 测试第 0 个样本

def load_scalers_from_json(run_dir):
    """单独加载 Scaler，不加载模型权重，避免报错"""
    tr_path = os.path.join(run_dir, 'transforms.json')
    if not os.path.isfile(tr_path):
        raise FileNotFoundError(f"{tr_path} not found")
    
    with open(tr_path, 'r') as f:
        meta = json.load(f)
    
    # 兼容不同的键名
    xs_key = 'proxy_x_scaler' if 'proxy_x_scaler' in meta else 'x_scaler'
    yt_key = 'proxy_y_transform' if 'proxy_y_transform' in meta else 'y_transform'
    
    x_scaler = XStandardizer.from_state_dict(meta[xs_key])
    y_tf = YTransform.from_state_dict(meta[yt_key])
    
    # 获取 iv_shape
    if 'iv_shape' in meta:
        iv_shape = meta['iv_shape']
    elif 'model_meta' in meta and 'iv_shape' in meta['model_meta']:
        iv_shape = meta['model_meta']['iv_shape']
    else:
        print("[Warn] iv_shape not found in json, using default (9, 61)")
        iv_shape = (9, 61)
        
    return x_scaler, y_tf, iv_shape

def test_oracle_strict():
    print(f"[Info] Device: {DEVICE}")
    
    # 1. 加载配置和 Scaler
    print(f">>> Loading Config from {CONFIG_SOURCE_RUN}...")
    x_scaler, y_tf, iv_shape = load_scalers_from_json(CONFIG_SOURCE_RUN)
    L_iv = int(np.prod(iv_shape))
    print(f"[Info] L_iv: {L_iv}")

    # 2. 加载 Proxy 网络
    print(f">>> Loading Proxy from {PROXY_RUN}...")
    # 注意：这里我们忽略 load_dual_proxy_artifacts 返回的 scaler，使用上面加载的
    proxy_iv, proxy_gm, _, _, _ = load_dual_proxy_artifacts(PROXY_RUN, DEVICE)
    
    proxy_iv.eval(); proxy_gm.eval()
    for p in proxy_iv.parameters(): p.requires_grad = False
    for p in proxy_gm.parameters(): p.requires_grad = False

    # 3. 加载测量数据 Target
    print(f">>> Loading Meas Data: {MEAS_FILE}, Sample={SAMPLE_IDX}")
    with h5py.File(MEAS_FILE, 'r') as f:
        raw_iv = f['X_iv'][SAMPLE_IDX]
        raw_gm = f['X_gm'][SAMPLE_IDX]
    
    # Standardize Target
    x_concat_raw = np.concatenate([raw_iv.flatten(), raw_gm.flatten()])[None, :]
    target_std = x_scaler.transform(x_concat_raw)
    
    target_iv_t = torch.tensor(target_std[:, :L_iv], device=DEVICE, dtype=torch.float32)
    target_gm_t = torch.tensor(target_std[:, L_iv:], device=DEVICE, dtype=torch.float32)

    # 4. 初始化优化变量
    param_dim = len(y_tf.names)
    # 从均值 (0) 开始
    opt_params = torch.zeros((1, param_dim), device=DEVICE, requires_grad=True)

    # 优化器
    optimizer = torch.optim.Adam([opt_params], lr=0.05)
    # 调度器
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=50)

    print(f"\n>>> Starting Strict Optimization (Projected Gradient Descent)...")
    
    best_loss = float('inf')
    best_phys_params = None

    for step in range(3000): # 增加步数，因为硬约束可能导致收敛变慢
        optimizer.zero_grad()
        
        # --- A. Forward ---
        # 此时 opt_params 还在 Normalized 空间，可能暂时是"不合法"的
        # 但我们先计算梯度，告诉它"想去哪"
        pred_iv = proxy_iv(opt_params)
        pred_gm = proxy_gm(opt_params)
        
        # --- B. Loss ---
        # 权重 IV:5.0, GM:1.0
        loss_iv = torch.nn.functional.mse_loss(pred_iv, target_iv_t)
        loss_gm = torch.nn.functional.mse_loss(pred_gm, target_gm_t)
        total_loss = 5.0 * loss_iv + 1.0 * loss_gm
        
        total_loss.backward()
        optimizer.step()
        
        # --- C. Projection (严格约束核心) ---
        # 这一步将参数强行拉回物理边界内
        with torch.no_grad():
            # 1. 反归一化 -> 物理空间
            p_phys = y_tf.inverse(opt_params)
            
            # 2. 对每个参数进行 Hard Clamp
            for i, name in enumerate(y_tf.names):
                if name in PARAM_RANGE:
                    low, high = PARAM_RANGE[name]
                    # In-place Clamp
                    p_phys[:, i].clamp_(min=low, max=high)
            
            # 3. 重新归一化 -> 覆盖优化变量
            p_norm_clamped = y_tf.transform(p_phys)
            opt_params.data.copy_(p_norm_clamped)

        # --- D. 记录最佳结果 ---
        scheduler.step(total_loss)
        
        if total_loss.item() < best_loss:
            best_loss = total_loss.item()
            # 保存对应的物理参数
            best_phys_params = y_tf.inverse(opt_params).detach().cpu().numpy()[0]

        if step % 500 == 0:
            lr = optimizer.param_groups[0]['lr']
            print(f"Step {step:04d} | Loss: {total_loss.item():.6f} | LR: {lr:.2e}")

    print(f"\n>>> Optimization Finished.")
    print(f"Final Best Loss (Strict Bounded): {best_loss:.6f}")
    
    # 打印最终参数检查
    print("-" * 65)
    print(f"{'Parameter':<10} | {'Optimized Value':<15} | {'Range':<20} | {'Status'}")
    print("-" * 65)
    
    for i, name in enumerate(y_tf.names):
        val = best_phys_params[i]
        status = "OK"
        bound_str = "N/A"
        
        if name in PARAM_RANGE:
            low, high = PARAM_RANGE[name]
            bound_str = f"[{low}, {high}]"
            # 允许微小的浮点误差
            if val < low - 1e-6 or val > high + 1e-6:
                status = "!!! OUT !!!"
            elif np.isclose(val, low, atol=1e-5):
                status = "Hit Min"
            elif np.isclose(val, high, atol=1e-5):
                status = "Hit Max"
        
        print(f"{name:<10} | {val:.4e}      | {bound_str:<20} | {status}")
    print("-" * 65)

if __name__ == "__main__":
    test_oracle_strict()