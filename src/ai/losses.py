import torch
import torch.nn as nn
import torchvision.models as models

class LensAuditLoss(nn.Module):
    def __init__(self, fft_weight=0.1, vgg_weight=0.1):
        super().__init__()
        self.l1 = nn.L1Loss()
        self.fft_weight = fft_weight
        self.vgg_weight = vgg_weight

        # --- THE UPGRADE: VGG16 Feature Extractor ---
        # We load a pre-trained VGG16 model but freeze its weights.
        # We only use the first 16 layers (up to relu3_3) to capture 
        # the perfect balance of structural shapes and fine textures.
        vgg = models.vgg16(weights=models.VGG16_Weights.IMAGENET1K_V1).features
        self.vgg_extractor = nn.Sequential(*list(vgg.children())[:16]).eval()
        
        # Freeze VGG parameters (we don't train VGG, we just use it as a judge)
        for param in self.vgg_extractor.parameters():
            param.requires_grad = False

    def forward(self, restored, sharp):
        # 1. Pixel Loss (L1 - The Foundation)
        l1_loss = self.l1(restored, sharp)

        # 2. Frequency Loss (FFT - The Crispness)
        fft_restored = torch.fft.rfft2(restored, norm="backward")
        fft_sharp = torch.fft.rfft2(sharp, norm="backward")
        fft_loss = self.l1(fft_restored, fft_sharp)

        # 3. Perceptual Loss (VGG - The Texture & Realism)
        # VGG expects images to be normalized to ImageNet standards
        mean = torch.tensor([0.485, 0.456, 0.406]).view(1,3,1,1).to(restored.device)
        std = torch.tensor([0.229, 0.224, 0.225]).view(1,3,1,1).to(restored.device)
        
        restored_vgg = (restored - mean) / std
        sharp_vgg = (sharp - mean) / std

        # Extract features and calculate the difference
        restored_features = self.vgg_extractor(restored_vgg)
        sharp_features = self.vgg_extractor(sharp_vgg)
        vgg_loss = self.l1(restored_features, sharp_features)

        # Total Forensic Loss
        return l1_loss + (self.fft_weight * fft_loss) + (self.vgg_weight * vgg_loss)