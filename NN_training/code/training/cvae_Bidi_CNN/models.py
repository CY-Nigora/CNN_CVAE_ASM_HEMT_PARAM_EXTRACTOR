# models.py
import math
from typing import List, Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F

# ==========
# MLP blocks (same style as your original)
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
# CNN feature extractor for dual-branch curves
# ==========
class DualCurveCNNEncoder(nn.Module):
    """
    Improved Encoder with Hybrid Pooling to preserve Amplitude and Shape.
    """
    def __init__(self, feat_dim: int = 256, dropout: float = 0.0):
        super().__init__()
        
        # 增加通道数，因为池化后特征图变大了，需要更强的表达能力
        # I-V Branch (7 x 121)
        self.iv_branch = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=(3, 5), padding=(1, 2)),
            nn.BatchNorm2d(16), # 建议加入BN，有助于幅度信息的稳定传递
            nn.GELU(),
            nn.Conv2d(16, 32, kernel_size=(3, 5), padding=(1, 2)),
            nn.BatchNorm2d(32),
            nn.GELU()
            # Removed AdaptiveAvgPool2d(1) here
        )
        
        # Gm Branch (10 x 71)
        self.gm_branch = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=(3, 5), padding=(1, 2)),
            nn.BatchNorm2d(16),
            nn.GELU(),
            nn.Conv2d(16, 32, kernel_size=(3, 5), padding=(1, 2)),
            nn.BatchNorm2d(32),
            nn.GELU()
        )

        # 定义保留的空间网格大小
        # (3, 3) 意味着保留 9 个局部区域的特征
        self.pool_grid = (3, 3) 
        flat_grid_size = self.pool_grid[0] * self.pool_grid[1]
        
        # 投影层输入维度计算：
        # 32 channels * (3*3 spatial) * 2 curves (IV+Gm) * 2 types (Avg+Max)
        # = 32 * 9 * 2 * 2 = 1152
        self.proj_in_dim = 32 * flat_grid_size * 2 * 2 

        self.proj = nn.Sequential(
            nn.Flatten(),
            nn.Linear(self.proj_in_dim, feat_dim),
            nn.LayerNorm(feat_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )

    def forward(self, x_iv: torch.Tensor, x_gm: torch.Tensor) -> torch.Tensor:
        # 1. Convolutions
        f_iv = self.iv_branch(x_iv) # [B, 32, 7, 121]
        f_gm = self.gm_branch(x_gm) # [B, 32, 10, 71]

        # 2. Hybrid Pooling (Avg + Max) to grid (3, 3)
        # Avg captures global shape/trend
        iv_avg = F.adaptive_avg_pool2d(f_iv, self.pool_grid).flatten(1)
        gm_avg = F.adaptive_avg_pool2d(f_gm, self.pool_grid).flatten(1)
        
        # Max captures peak amplitude (saturation current, peak gm)
        iv_max = F.adaptive_max_pool2d(f_iv, self.pool_grid).flatten(1)
        gm_max = F.adaptive_max_pool2d(f_gm, self.pool_grid).flatten(1)

        # 3. Concatenate Everything
        # [B, 32*9] * 4 parts
        concat_feat = torch.cat([iv_avg, iv_max, gm_avg, gm_max], dim=1)
        
        # 4. Project
        h = self.proj(concat_feat)
        return h


class CoordinateHybridEncoder(nn.Module):
    """
    Ultimate Architecture V2: Split Grids
    分别针对 IV 和 Gm 设置不同的池化网格，以适应不同的长宽比。
    """
    def __init__(self, iv_shape=(9, 61), gm_shape=(6, 101), feat_dim=256, dropout=0.0, use_stats=False):
        super().__init__()
        self.iv_shape = iv_shape
        self.gm_shape = gm_shape
        self.use_stats = use_stats
        
        # --- 1. Coordinate Channels Setup ---
        in_ch = 3 
        
        # --- 2. CNN Backbone ---
        self.iv_conv = nn.Sequential(
            nn.Conv2d(in_ch, 32, kernel_size=(3, 5), padding=(1, 2)),
            nn.BatchNorm2d(32),
            nn.GELU(),
            nn.Conv2d(32, 64, kernel_size=(3, 5), padding=(1, 2)),
            nn.BatchNorm2d(64),
            nn.GELU()
        )
        
        self.gm_conv = nn.Sequential(
            nn.Conv2d(in_ch, 32, kernel_size=(3, 5), padding=(1, 2)),
            nn.BatchNorm2d(32),
            nn.GELU(),
            nn.Conv2d(32, 64, kernel_size=(3, 5), padding=(1, 2)),
            nn.BatchNorm2d(64),
            nn.GELU()
        )
        
        # --- 3. Hybrid Pooling Grids (Split Scales) ---
        # [关键修改]: 针对 IV (长) 和 Gm (短) 使用不同的网格
        # IV (9, 61): 高度9适合被3整除，宽度61适合被6整除(约10像素/格)
        self.iv_grid = (3, 6)  # 原先是 (3, 12)
        # Gm (6, 101): 高度6适合被2或3整除，宽度101适合被10整除(约10像素/格)
        self.gm_grid = (2, 10) # 原先是 (3, 6)
        
        iv_flat_size = self.iv_grid[0] * self.iv_grid[1] # 36
        gm_flat_size = self.gm_grid[0] * self.gm_grid[1] # 18
        
        # 计算总维度: 64通道 * (IV网格 + Gm网格) * 2种池化(Max+Avg)
        # Total = 64 * (36 + 18) * 2 = 64 * 54 * 2 = 6912
        self.cnn_out_dim = 64 * (iv_flat_size + gm_flat_size) * 2 
        
        # --- 4. Physics Stats Branch (Optional) ---
        self.stats_dim = 4 * 2 
        if self.use_stats:
            self.stats_proj = nn.Sequential(
                nn.Linear(self.stats_dim, 32),
                nn.GELU()
            )
            total_dim = self.cnn_out_dim + 32
        else:
            self.stats_proj = None
            total_dim = self.cnn_out_dim

        # --- 5. Fusion & Project ---
        self.proj = nn.Sequential(
            nn.Flatten(),
            nn.Linear(total_dim, feat_dim),
            nn.LayerNorm(feat_dim),
            nn.GELU(),
            nn.Dropout(dropout)
        )
        
        self.register_buffer('iv_coords', self._make_grid(iv_shape))
        self.register_buffer('gm_coords', self._make_grid(gm_shape))

    def _make_grid(self, shape):
        H, W = shape
        y_grid = torch.linspace(-1, 1, H).view(1, 1, H, 1).expand(1, 1, H, W)
        x_grid = torch.linspace(-1, 1, W).view(1, 1, 1, W).expand(1, 1, H, W)
        return torch.cat([y_grid, x_grid], dim=1)

    def _add_coords(self, x, coords):
        B = x.shape[0]
        coords_batch = coords.expand(B, -1, -1, -1)
        return torch.cat([x, coords_batch], dim=1)

    def _calc_stats(self, x):
        x_flat = x.flatten(1)
        return torch.stack([
            x_flat.max(dim=1).values,
            x_flat.min(dim=1).values,
            x_flat.mean(dim=1),
            x_flat.std(dim=1)
        ], dim=1)

    def forward(self, x_iv, x_gm):
        # 1. Coordinate Input
        x_iv_coord = self._add_coords(x_iv, self.iv_coords)
        x_gm_coord = self._add_coords(x_gm, self.gm_coords)
        
        # 2. Conv Features
        f_iv = self.iv_conv(x_iv_coord)
        f_gm = self.gm_conv(x_gm_coord)
        
        # 3. Split Hybrid Pooling
        # IV 使用 (3, 12)
        iv_pool = torch.cat([
            F.adaptive_avg_pool2d(f_iv, self.iv_grid).flatten(1),
            F.adaptive_max_pool2d(f_iv, self.iv_grid).flatten(1)
        ], dim=1)
        
        # Gm 使用 (3, 6)
        gm_pool = torch.cat([
            F.adaptive_avg_pool2d(f_gm, self.gm_grid).flatten(1),
            F.adaptive_max_pool2d(f_gm, self.gm_grid).flatten(1)
        ], dim=1)
        
        cnn_feat = torch.cat([iv_pool, gm_pool], dim=1)

        # 4. Optional Stats
        if self.use_stats:
            st_iv = self._calc_stats(x_iv)
            st_gm = self._calc_stats(x_gm)
            stats = torch.cat([st_iv, st_gm], dim=1)
            stats_feat = self.stats_proj(stats)
            combined = torch.cat([cnn_feat, stats_feat], dim=1)
            return self.proj(combined)
        else:
            return self.proj(cnn_feat)

class DualCurveMLPEncoder(nn.Module):
    """
    High-Capacity MLP Encoder.
    Processes IV and GM curves separately with wide fully-connected layers
    to capture global correlations and absolute amplitude values.
    """
    def __init__(self, iv_shape=(9, 61), gm_shape=(6, 101), feat_dim=256, dropout=0.0):
        super().__init__()
        
        # Calculate Flatten Input Dimensions
        self.iv_in_dim = int(torch.tensor(iv_shape).prod().item()) # 7 * 121 = 847
        self.gm_in_dim = int(torch.tensor(gm_shape).prod().item()) # 10 * 71 = 710
        
        # --- Branch A: IV MLP ---
        # 使用宽层 (1024/2048) 来捕捉 847 个点之间的精细关系
        self.iv_net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(self.iv_in_dim, 2048),
            nn.BatchNorm1d(2048), # BN 对 MLP 拟合物理幅度至关重要
            nn.GELU(),
            nn.Dropout(dropout),
            
            nn.Linear(2048, 1024),
            nn.BatchNorm1d(1024),
            nn.GELU(),
            nn.Dropout(dropout)
        )
        
        # --- Branch B: GM MLP ---
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
        
        # --- Fusion & Projection ---
        # Concatenated dim: 1024 (IV) + 512 (GM) = 1536
        fusion_dim = 1024 + 512
        
        self.proj = nn.Sequential(
            nn.Linear(fusion_dim, 1024),
            nn.BatchNorm1d(1024),
            nn.GELU(),
            nn.Dropout(dropout),
            
            nn.Linear(1024, feat_dim),
            nn.LayerNorm(feat_dim), # Final Norm for Latent Space stability
            nn.GELU()
        )

    def forward(self, x_iv, x_gm):
        # x_iv: [B, 1, 7, 121] -> Flatten inside
        # x_gm: [B, 1, 10, 71] -> Flatten inside
        
        h_iv = self.iv_net(x_iv)
        h_gm = self.gm_net(x_gm)
        
        combined = torch.cat([h_iv, h_gm], dim=1)
        return self.proj(combined)


# ==========
# CVAE with dual CNN encoder
# ==========
class DualInputCVAE(nn.Module):
    """
    Keep CVAE structure:
      Encoder:  P(z|h,y)
      Prior:    P(z|h)
      Decoder:  P(y|h,z)
    where h = CNN(x_iv, x_gm)
    """
    def __init__(self, y_dim: int, hidden: List[int], latent_dim: int,
                 feat_dim: int = 256, cnn_dropout: float = 0.0, mlp_dropout: float = 0.1, use_stats_flag: bool = False):
        super().__init__()
        self.y_dim = y_dim
        self.latent_dim = latent_dim
        self.feat_dim = feat_dim

        # -----------------------------------------------------------
        # Option 1: Hybrid CNN 
        # -----------------------------------------------------------
        # self.x_encoder = CoordinateHybridEncoder(
        #     iv_shape=(7,121), 
        #     gm_shape=(10,71), 
        #     feat_dim=feat_dim, 
        #     dropout=cnn_dropout,
        #     use_stats=use_stats_flag  
        # )
        # -----------------------------------------------------------
        # Option 2: Pure MLP (Baseline)
        # -----------------------------------------------------------
        self.x_encoder = DualCurveMLPEncoder(
            iv_shape=(9, 61), 
            gm_shape=(6, 101), 
            feat_dim=feat_dim, 
            dropout=cnn_dropout # Reuse cnn_dropout arg for MLP dropout
        )

        self.encoder   = _MLPBlock(feat_dim + y_dim, hidden, latent_dim * 2, mlp_dropout)
        self.prior_net = _MLPBlock(feat_dim, hidden, latent_dim * 2, mlp_dropout)
        self.decoder   = _MLPBlock(feat_dim + latent_dim, hidden, y_dim, mlp_dropout)

    def encode_x(self, x_iv, x_gm):
        return self.x_encoder(x_iv, x_gm)

    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x_iv: torch.Tensor, x_gm: torch.Tensor, y: Optional[torch.Tensor] = None):
        h = self.encode_x(x_iv, x_gm)

        prior_out = self.prior_net(h)
        mu_prior, logvar_prior = prior_out.chunk(2, dim=-1)

        if y is not None:
            enc_in = torch.cat([h, y], dim=1)
            enc_out = self.encoder(enc_in)
            mu_post, logvar_post = enc_out.chunk(2, dim=-1)

            z = self.reparameterize(mu_post, logvar_post)
            dec_in = torch.cat([h, z], dim=1)
            y_hat = self.decoder(dec_in)
            return y_hat, h, (mu_post, logvar_post), (mu_prior, logvar_prior)
        else:
            z = self.reparameterize(mu_prior, logvar_prior)
            dec_in = torch.cat([h, z], dim=1)
            y_hat = self.decoder(dec_in)
            return y_hat, h, (None, None), (mu_prior, logvar_prior)

    def sample(self, x_iv: torch.Tensor, x_gm: torch.Tensor,
               num_samples: int = 1, sample_mode: str = 'rand'):
        self.eval()
        with torch.no_grad():
            h = self.encode_x(x_iv, x_gm)
            prior_out = self.prior_net(h)
            mu_prior, logvar_prior = prior_out.chunk(2, dim=-1)

            ys = []
            for _ in range(num_samples):
                if sample_mode == 'mean':
                    z = mu_prior
                else:
                    z = self.reparameterize(mu_prior, logvar_prior)
                dec_in = torch.cat([h, z], dim=1)
                ys.append(self.decoder(dec_in).unsqueeze(0))
        return torch.cat(ys, dim=0)
