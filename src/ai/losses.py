# [LEAD] Member A: Physics-Informed Loss Functions
import torch
import torch.nn as nn

class LensAuditLoss(nn.Module):
    def __init__(self, fft_weight=0.1):
        super(LensAuditLoss, self).__init__()
        self.eps = 1e-3
        self.fft_weight = fft_weight

    def forward(self, restored, target):
        # 1. Charbonnier Loss (Spatial Domain)
        diff = restored - target
        loss_char = torch.mean(torch.sqrt(diff * diff + self.eps * self.eps))

        # 2. FFT Loss (Frequency Domain)
        # Force the AI to match the frequency 'fingerprint'
        res_fft = torch.fft.rfft2(restored)
        tar_fft = torch.fft.rfft2(target)
        loss_fft = torch.mean(torch.abs(res_fft - tar_fft))

        return loss_char + (self.fft_weight * loss_fft)