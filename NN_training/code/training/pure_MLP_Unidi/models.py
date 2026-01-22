import math
from typing import List, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F

# ==========
# MLP blocks (基础模块)
# ==========
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

# ==========
# Encoders
# ==========
class DualCurveMLPEncoder(nn.Module):
    """
    High-Capacity MLP Encoder.
    Processes IV and GM curves separately.
    """
    def __init__(self, iv_shape=(9, 61), gm_shape=(6, 101), feat_dim=256, dropout=0.0):
        super().__init__()
        
        self.iv_in_dim = int(torch.tensor(iv_shape).prod().item())
        self.gm_in_dim = int(torch.tensor(gm_shape).prod().item())
        
        # Branch A: IV MLP
        self.iv_net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(self.iv_in_dim, 2048),
            nn.BatchNorm1d(2048),
            nn.GELU(),
            nn.Dropout(dropout),
            
            nn.Linear(2048, 1024),
            nn.BatchNorm1d(1024),
            nn.GELU(),
            nn.Dropout(dropout)
        )
        
        # Branch B: GM MLP
        self.gm_net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(self.gm_in_dim, 1024),
            nn.BatchNorm1d(1024),
            nn.GELU(),
            nn.Dropout(dropout),
            
            nn.Linear(1024, 512),
            nn.BatchNorm1d(512),
            nn.GELU(),
            nn.Dropout(dropout)
        )
        
        fusion_dim = 1024 + 512
        
        self.proj = nn.Sequential(
            nn.Linear(fusion_dim, 1024),
            nn.BatchNorm1d(1024),
            nn.GELU(),
            nn.Dropout(dropout),
            
            nn.Linear(1024, feat_dim),
            nn.LayerNorm(feat_dim),
            nn.GELU()
        )

    def forward(self, x_iv, x_gm):
        h_iv = self.iv_net(x_iv)
        h_gm = self.gm_net(x_gm)
        combined = torch.cat([h_iv, h_gm], dim=1)
        return self.proj(combined)

# ==========
# Pure MLP Regressor
# ==========
class DualInputMLP(nn.Module):
    """
    Pure MLP Baseline.
    Directly maps (IV, GM) -> Parameters Y.
    """
    def __init__(self, y_dim: int, hidden: List[int], 
                 feat_dim: int = 256, 
                 dropout: float = 0.1,  # 统一的 dropout 参数
                 iv_shape=(9,61), gm_shape=(6,101)):
        super().__init__()
        
        # 1. Feature Extractor (Using MLP Encoder)
        self.x_encoder = DualCurveMLPEncoder(
            iv_shape=iv_shape, 
            gm_shape=gm_shape, 
            feat_dim=feat_dim, 
            dropout=dropout # 应用 dropout
        )

        # 2. Regression Head (Features -> Y)
        self.regressor = _MLPBlock(feat_dim, hidden, y_dim, dropout=dropout) # 应用 dropout

    def forward(self, x_iv: torch.Tensor, x_gm: torch.Tensor, y: Optional[torch.Tensor] = None):
        # 1. Extract Features
        h = self.x_encoder(x_iv, x_gm)
        # 2. Predict Y
        y_pred = self.regressor(h)
        return y_pred