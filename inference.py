import torch
import os
import cv2
import numpy as np
from torchvision.utils import save_image
from src.ai.restormer import LensAuditNet
import math

def calculate_psnr(img1, img2):
    mse = torch.mean((img1 - img2) ** 2)
    if mse == 0: return 100
    return 20 * math.log10(1.0 / math.sqrt(mse.item()))

def run_evaluation():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    WEIGHTS_PATH = 'lens_audit_diverse_best.pth'
    TEST_DATA_ROOT = r'C:\Users\Skanda Bharadwaj G M\Downloads\IPCV\data\gopro\GOPRO_Large\test'
    SAVE_DIR = 'reports/final_results'
    os.makedirs(SAVE_DIR, exist_ok=True)

    model = LensAuditNet(dim=48, num_blocks=4).to(device)
    model.load_state_dict(torch.load(WEIGHTS_PATH, map_location=device))
    model.eval()

    print(f"[*] Starting Robust Forensic Audit...")
    
    count = 0
    total_psnr = 0

    with torch.no_grad():
        for root, dirs, files in os.walk(TEST_DATA_ROOT):
            # Only look in 'blur' folders
            if root.endswith('blur'):
                for file in files[:2]: # Sample 2 images per sequence
                    blur_path = os.path.join(root, file)
                    
                    # SMART PATH FIX:
                    # Finds the 'sharp' folder at the same level as the 'blur' folder
                    sharp_path = blur_path.replace('\\blur\\', '\\sharp\\').replace('/blur/', '/sharp/')
                    
                    if not os.path.exists(sharp_path):
                        # Fallback for some GoPro versions that use 'sharp_gamma'
                        sharp_path = blur_path.replace('blur', 'sharp_gamma')

                    # Read images with check
                    blur_img = cv2.imread(blur_path)
                    sharp_img = cv2.imread(sharp_path)

                    if blur_img is None or sharp_img is None:
                        print(f"[!] Skipping {file}: Path mismatch.")
                        continue

                    # Process
                    blur_img = cv2.cvtColor(blur_img, cv2.COLOR_BGR2RGB)
                    sharp_img = cv2.cvtColor(sharp_img, cv2.COLOR_BGR2RGB)
                    
                    blur_t = torch.from_numpy(blur_img).permute(2,0,1).float().divide(255).unsqueeze(0).to(device)
                    sharp_t = torch.from_numpy(sharp_img).permute(2,0,1).float().divide(255).unsqueeze(0).to(device)
                    
                    # Consistent 256 crop for PSNR math
                    blur_t = blur_t[:, :, :256, :256]
                    sharp_t = sharp_t[:, :, :256, :256]

                    restored = model(blur_t)
                    
                    psnr = calculate_psnr(restored, sharp_t)
                    total_psnr += psnr
                    count += 1

                    # Save result
                    comparison = torch.cat([blur_t[0], restored[0], sharp_t[0]], dim=2)
                    save_image(comparison, f'{SAVE_DIR}/test_result_{count}.png')
                    print(f"Sample {count} | PSNR: {psnr:.2f}dB")

    if count > 0:
        print(f"\n[FINAL] Avg PSNR: {total_psnr/count:.2f}dB")
    avg_psnr = total_psnr / count
    print(f"\n[FINAL AUDIT COMPLETE]")
    print(f"Average PSNR on Test Set: {avg_psnr:.2f}dB")

if __name__ == "__main__":
    run_evaluation()