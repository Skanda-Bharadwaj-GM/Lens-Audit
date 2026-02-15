# [LEAD] Member A: Restormer Architecture
import torch
import torch.nn as nn
import torch.nn.functional as F

class GatedFeedForward(nn.Module):
    """Member A: Gated Feed-Forward Network for feature selection."""
    def __init__(self, dim, expansion_factor=2.66):
        super(GatedFeedForward, self).__init__()
        hidden_dim = int(dim * expansion_factor)
        self.project_in = nn.Conv2d(dim, hidden_dim * 2, kernel_size=1, bias=False)
        self.dwconv = nn.Conv2d(hidden_dim * 2, hidden_dim * 2, kernel_size=3, stride=1, padding=1, groups=hidden_dim * 2, bias=False)
        self.project_out = nn.Conv2d(hidden_dim, dim, kernel_size=1, bias=False)

    def forward(self, x):
        x = self.project_in(x)
        x1, x2 = self.dwconv(x).chunk(2, dim=1)
        x = F.gelu(x1) * x2  # Gating mechanism
        x = self.project_out(x)
        return x

class RestormerBlock(nn.Module):
    """The complete Transformer Block for Image Restoration."""
    def __init__(self, dim, num_heads):
        super(RestormerBlock, self).__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        
        # We use the MDTA block defined in previous steps
        from .restormer_parts import MDTA 
        self.attn = MDTA(dim, num_heads)
        self.ffn = GatedFeedForward(dim)

    def forward(self, x):
        # LayerNorm expects [B, L, C], but images are [B, C, H, W]
        # We permute before/after normalization
        x = x + self.attn(self.norm1(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2))
        x = x + self.ffn(self.norm2(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2))
        return x

class LensAuditNet(nn.Module):
    """The full Network Skeleton."""
    def __init__(self, in_channels=3, out_channels=3, dim=48, num_blocks=4):
        super(LensAuditNet, self).__init__()
        self.intro = nn.Conv2d(in_channels, dim, kernel_size=3, padding=1)
        
        self.body = nn.Sequential(*[
            RestormerBlock(dim, num_heads=4) for _ in range(num_blocks)
        ])
        
        self.outro = nn.Conv2d(dim, out_channels, kernel_size=3, padding=1)

    def forward(self, x):
        identity = x
        x = self.intro(x)
        x = self.body(x)
        x = self.outro(x)
        return x + identity # Residual learning makes deblurring much easier

print("[SUCCESS] Restormer (LensAuditNet) code is finalized.")