import torch
import cv2
import numpy as np
from src.ai.restormer import LensAuditNet

def run_test(image_path, model_path='lens_audit_latest.pth'):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 1. Load Model
    model = LensAuditNet(dim=48, num_blocks=4).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    # 2. Load and Prep Image
    img = cv2.imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_tensor = torch.from_numpy(img).permute(2, 0, 1).float().unsqueeze(0) / 255.0
    
    # 3. Inference
    with torch.no_grad():
        restored = model(img_tensor.to(device))
    
    # 4. Save Result
    output = restored.squeeze().permute(1, 2, 0).cpu().numpy() * 255
    cv2.imwrite('test_output.png', cv2.cvtColor(output.astype(np.uint8), cv2.COLOR_RGB2BGR))
    print("[SUCCESS] Test image saved as test_output.png")

if __name__ == "__main__":
    # Member B will update this path to a real test image
    # run_test('data/gopro/test/some_scene/blur/000001.png')
    pass