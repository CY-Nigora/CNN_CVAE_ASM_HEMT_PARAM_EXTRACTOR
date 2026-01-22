import torch
import numpy as np
import h5py
import os
import argparse
import pandas as pd
from tqdm import tqdm

from proxy import load_dual_proxy_artifacts
from data import XStandardizer, YTransform
from main import load_cvae_artifacts_dual

# ==========================================
# 1. define a strict physical boundary (same with that in main.py)
# ==========================================
PARAM_RANGE = {
    # 'VOFF': (-4.7, -4),
    'VOFF': (-4.7, -3),
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

def run_tto_optimization(
    target_iv, target_gm, 
    init_params_norm, 
    proxy_iv, proxy_gm, 
    y_tf, 
    device, 
    steps=200, 
    lr=0.01
):
    """
    Test-Time Optimization with Strict Boundary Projection.
    """
    # 1. 设置优化变量 (从 CVAE 预测值开始)
    # init_params_norm: [1, D]
    opt_params = init_params_norm.clone().detach().to(device).requires_grad_(True)
    
    # 2. 优化器
    optimizer = torch.optim.Adam([opt_params], lr=lr)
    
    # 3. 迭代优化
    for i in range(steps):
        optimizer.zero_grad()
        
        # --- A. Forward Proxy (在归一化空间进行) ---
        # 此时 opt_params 可能是越界的，但 Proxy 是神经网络，越界也能算
        pred_iv = proxy_iv(opt_params)
        pred_gm = proxy_gm(opt_params)
        
        # --- B. 计算 Loss ---
        # 强力拟合 Meas 数据 (权重 IV:GM = 5:1)
        loss = 5.0 * torch.nn.functional.mse_loss(pred_iv, target_iv) + \
               1.0 * torch.nn.functional.mse_loss(pred_gm, target_gm)
        
        loss.backward()
        optimizer.step()
        
        # --- C. [核心] 投影操作 (Projected Gradient) ---
        # 强制将更新后的参数拉回物理边界内
        with torch.no_grad():
            # 1. 反归一化到物理空间
            p_phys = y_tf.inverse(opt_params)
            
            # 2. 对每一个参数进行 Clamp
            for idx, name in enumerate(y_tf.names):
                if name in PARAM_RANGE:
                    low, high = PARAM_RANGE[name]
                    # In-place clamp
                    p_phys[:, idx].clamp_(min=low, max=high)
            
            # 3. 重新归一化并写回优化变量
            # 这一步保证了下一次 forward 时使用的是合法的参数
            p_norm_clamped = y_tf.transform(p_phys)
            opt_params.data.copy_(p_norm_clamped) # 直接覆盖 data
    
    # 4. 返回最终结果
    with torch.no_grad():
        final_phys = y_tf.inverse(opt_params)
        final_loss = loss.item()
                
    return final_phys, final_loss

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--cvae-run', type=str, required=True, help='Path to Stage 1/2 CVAE run dir')
    parser.add_argument('--proxy-run', type=str, required=True, help='Path to Proxy run dir')
    parser.add_argument('--meas-h5', type=str, required=True)
    parser.add_argument('--save-to', type=str, default='tto_strict_results.csv')
    parser.add_argument('--steps', type=int, default=300, help='Optimization steps per sample')
    parser.add_argument('--lr', type=float, default=0.05, help='Learning rate for TTO')
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 1. 加载模型
    print(f"[Info] Loading CVAE from {args.cvae_run}")
    # 注意：如果你的 main.py 需要 use_physics_stats，请确保 load 函数能处理
    cvae, x_scaler, y_tf, meta = load_cvae_artifacts_dual(args.cvae_run, device)
    
    print(f"[Info] Loading Proxy from {args.proxy_run}")
    proxy_iv, proxy_gm, _, _, _ = load_dual_proxy_artifacts(args.proxy_run, device)
    
    # 冻结所有网络，只把 Input (opt_params) 当作变量
    cvae.eval(); proxy_iv.eval(); proxy_gm.eval()
    for p in cvae.parameters(): p.requires_grad = False
    for p in proxy_iv.parameters(): p.requires_grad = False
    for p in proxy_gm.parameters(): p.requires_grad = False

    # 2. 加载数据
    print(f"[Info] Loading Meas Data from {args.meas_h5}")
    with h5py.File(args.meas_h5, 'r') as f:
        Xm_iv = f['X_iv'][...]
        Xm_gm = f['X_gm'][...]
    
    # 3. 准备结果容器
    results = []
    N = Xm_iv.shape[0]
    print(f"[Info] Starting Strict TTO for {N} samples...")
    print(f"[Info] Constraints: {PARAM_RANGE.keys()}")
    
    # 4. 循环处理
    for i in tqdm(range(N)):
        # --- Data Prep ---
        x_raw = np.concatenate([Xm_iv[i].flatten(), Xm_gm[i].flatten()])[None, :]
        x_std = x_scaler.transform(x_raw)
        
        L_iv = np.prod(meta['input_meta']['iv_shape'])
        target_iv = torch.tensor(x_std[:, :L_iv], device=device, dtype=torch.float32)
        target_gm = torch.tensor(x_std[:, L_iv:], device=device, dtype=torch.float32)
        
        # Prepare Network Input
        x_iv_cnn = torch.tensor(x_std[:, :L_iv].reshape(1, 1, *meta['input_meta']['iv_shape']), device=device, dtype=torch.float32)
        x_gm_cnn = torch.tensor(x_std[:, L_iv:].reshape(1, 1, *meta['input_meta']['gm_shape']), device=device, dtype=torch.float32)

        # --- Step 1: CVAE Initial Guess ---
        with torch.no_grad():
            h = cvae.encode_x(x_iv_cnn, x_gm_cnn)
            prior_out = cvae.prior_net(h)
            mu_prior, _ = prior_out.chunk(2, dim=-1)
            z_init = mu_prior 
            
            dec_in = torch.cat([h, z_init], dim=1)
            y_init_norm = cvae.decoder(dec_in) # CVAE 预测的初始参数

        # --- Step 2: Strict TTO ---
        # 传入 y_init_norm 作为起点，进行有约束的梯度下降
        final_params_phys, final_loss = run_tto_optimization(
            target_iv, target_gm, 
            y_init_norm, 
            proxy_iv, proxy_gm, 
            y_tf, device, 
            steps=args.steps, lr=args.lr
        )
        
        # --- Save Result ---
        row = final_params_phys.cpu().numpy()[0].tolist()
        row.append(final_loss)
        results.append(row)

    # 5. 保存结果
    cols = y_tf.names + ['final_loss']
    df = pd.DataFrame(results, columns=cols)
    df.to_csv(args.save_to, index=False)
    print(f"[Done] Results saved to {args.save_to}")
    print(f"Average Final Loss: {df['final_loss'].mean():.5f}")

if __name__ == "__main__":
    main()