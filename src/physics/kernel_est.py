import cv2
import numpy as np
import matplotlib.pyplot as plt

def estimate_blur_kernel(image_path, visualize=True):
    """
    [LEAD] Member A: Inverse Physics Module.
    Extracts the 'Blur Fingerprint' using the Frequency Domain.
    """
    # Load image in grayscale
    img = cv2.imread(image_path, 0)
    if img is None:
        raise FileNotFoundError(f"Image not found at {image_path}")

    # 1. Compute 2D Fast Fourier Transform
    dft = np.fft.fft2(img)
    dft_shift = np.fft.fftshift(dft)
    
    # 2. Compute Magnitude Spectrum (log scale)
    # The spectrum shows dark lines perpendicular to the motion direction.
    magnitude_spectrum = 20 * np.log(np.abs(dft_shift) + 1)
    
    # 3. Detect Orientation (Simple Gradient Analysis)
    # In a real forensic scenario, we'd use Radon Transform here, 
    # but for now, we visualize the 'streaks'.
    if visualize:
        plt.figure(figsize=(10, 5))
        plt.subplot(121), plt.imshow(img, cmap='gray'), plt.title('Forensic Evidence')
        plt.subplot(122), plt.imshow(magnitude_spectrum, cmap='gray'), plt.title('FFT Blur Fingerprint')
        plt.show()

    return magnitude_spectrum

if __name__ == "__main__":
    # Test on a sample image (Make sure you have a test image in your root)
    # estimate_blur_kernel('test_blur.jpg')
    print("[SUCCESS] Kernel Estimation module initialized.")
