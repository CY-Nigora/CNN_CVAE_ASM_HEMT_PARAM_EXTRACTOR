import tkinter as tk
from tkinter import ttk
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import numpy as np

import sys
sys.path.append('E:/personal_Data/Document of School/Uni Stuttgart/Masterarbeit/Code/param_regression/ADS_Parameter_Fitting/IV_param_regression/data_gen')
from log_data_gen import param_random_generator


# —— 参数范围（决定每条数轴的左右端点 & 顺序）——
param_range = {
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

# —— 你提供的参数生成器（原样集成）——
def param_random_generator_linear(param_range: dict):
    ''' generate a random parameter set for the HEMT model '''
    var_dict = {key: str(np.random.uniform(low=val[0], high=val[1])) for key, val in param_range.items()}
    return var_dict

# —— 数据缓存：每个参数一条 list —— 
all_samples = {p: [] for p in param_range.keys()}
total_groups = 0  # 已接收的样本组数（一组=一次函数返回的整套参数）

def _to_float(x):
    try:
        return float(x)
    except Exception:
        return np.nan

def add_sample(sample_dict: dict):
    """
    接收一组样本（dict），key 为参数名、value 为该参数值（字符串或数值均可）。
    未在 param_range 中的 key 忽略；缺失的参数不报错。
    """
    global total_groups
    wrote = False
    for p in param_range.keys():
        if p in sample_dict:
            val = _to_float(sample_dict[p])
            if not np.isnan(val):
                all_samples[p].append(val)
                wrote = True
    if wrote:
        total_groups += 1
        lbl_status_var.set(f"已接收样本组数：{total_groups}")

def clear_samples():
    """清空所有已收集的数据并刷新图面"""
    global total_groups
    for p in all_samples:
        all_samples[p].clear()
    total_groups = 0
    lbl_status_var.set("已接收样本组数：0")
    plot_samples()

def fmt_num(x: float) -> str:
    """端点标签友好的数值格式"""
    ax_abs = abs(x)
    if (ax_abs != 0 and ax_abs < 1e-3) or ax_abs >= 1e4:
        return f"{x:.2e}"
    return f"{x:.6g}"

def plot_samples():
    """
    绘制“独立数轴”：
    - 每个参数一个子图（独立 x 轴），xlim 即该参数范围；
    - 在 y=0 处画一条数轴（横线），并把样本点打在这条线上；
    - 仅显示左右端点刻度，以凸显“首尾=参数范围首尾”。
    """
    fig.clf()
    n = len(param_range)
    base_h = max(6, 0.6 * n)  # 动态高度：每条数轴约 0.6 英寸
    fig.set_size_inches(10, base_h)

    for i, (param, (low, high)) in enumerate(param_range.items()):
        ax = fig.add_subplot(n, 1, i + 1)

        # 独立数轴
        ax.set_xlim(low, high)
        ax.set_ylim(-0.5, 0.5)
        ax.hlines(0, low, high, linewidth=1)

        # 样本点
        vals = all_samples[param]
        if vals:
            ax.scatter(vals, [0] * len(vals), s=16, zorder=3)

        # 仅端点刻度
        ax.set_xticks([low, high])
        ax.set_xticklabels([fmt_num(low), fmt_num(high)])
        ax.grid(axis='x', linestyle=':', alpha=0.25)

        # 隐藏 y 轴
        ax.set_yticks([])
        for spine in ['left', 'right', 'top']:
            ax.spines[spine].set_visible(False)

        # 左侧标注参数名
        ax.set_ylabel(param, rotation=0, ha='right', va='center', labelpad=25)

        if i == n - 1:
            ax.set_xlabel("Value")

    fig.tight_layout()
    canvas.draw()

def run_10x():
    """点击按钮：用你的生成器循环 10 次，累积并绘图"""
    for _ in range(10):
        sample = param_random_generator(param_range)
        print(sample['VDSCALE'])
        add_sample(sample)
    plot_samples()

# —— 构建 GUI —— 
root = tk.Tk()
root.title("参数独立数轴 · 样本可视化")

toolbar = ttk.Frame(root)
toolbar.pack(fill=tk.X, padx=8, pady=6)

btn_run = ttk.Button(toolbar, text="Run (10x)", command=run_10x)
btn_run.pack(side=tk.LEFT, padx=(0, 6))

btn_plot = ttk.Button(toolbar, text="Refresh", command=plot_samples)
btn_plot.pack(side=tk.LEFT, padx=(0, 6))

btn_clear = ttk.Button(toolbar, text="Reset", command=clear_samples)
btn_clear.pack(side=tk.LEFT)

lbl_status_var = tk.StringVar(value="已接收样本组数：0")
lbl_status = ttk.Label(toolbar, textvariable=lbl_status_var)
lbl_status.pack(side=tk.RIGHT)

fig = plt.Figure(figsize=(10, 6), dpi=100)
canvas = FigureCanvasTkAgg(fig, master=root)
canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True, padx=8, pady=(0, 8))

# —— 可选：对外暴露接口（若你想在外部脚本里自己驱动）——
def get_add_sample_func():
    return add_sample
def get_plot_func():
    return plot_samples
def get_clear_func():
    return clear_samples

# 初始空图
plot_samples()

root.mainloop()
