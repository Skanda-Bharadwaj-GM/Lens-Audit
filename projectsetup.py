import os

def create_structure():
    # Define the folder structure
    folders = [
        "data/gopro",
        "src/physics",
        "src/ai",
        "src/auditor",
        "scripts",
        "notebooks",
        "reports/figures"
    ]

    # Define initial empty files with basic headers
    files = {
        "src/physics/blur_layer.py": "# [LEAD] Member A: Forward Physics (Blur Simulation)\n",
        "src/physics/kernel_est.py": "# [LEAD] Member A: Inverse Physics (FFT & PSF Estimation)\n",
        "src/ai/restormer.py": "# [LEAD] Member A: Restormer Architecture\n",
        "src/ai/losses.py": "# [LEAD] Member A: Physics-Informed Loss Functions\n",
        "src/auditor/ocr_engine.py": "# [SUPPORT] Member B: EasyOCR/Tesseract Logic\n",
        "src/auditor/metrics.py": "# [SUPPORT] Member B: PSNR, SSIM, LPIPS calculations\n",
        "requirements.txt": "torch\ntorchvision\nopencv-python\nscikit-image\nscipy\nkornia\nmatplotlib\neasyocr\ngradio\n",
        ".gitignore": "data/\n__pycache__/\n*.pyc\n.ipynb_checkpoints/\n.env\n"
    }

    # Create folders
    for folder in folders:
        os.makedirs(folder, exist_ok=True)
        print(f"Created folder: {folder}")

    # Create files
    for file_path, content in files.items():
        if not os.path.exists(file_path):
            with open(file_path, "w") as f:
                f.write(content)
            print(f"Created file: {file_path}")

    print("\n[SUCCESS] Lens-Audit project structure is ready.")
    print("[NEXT STEP] Move your GoPro dataset into 'data/gopro/'")

if __name__ == "__main__":
    create_structure()