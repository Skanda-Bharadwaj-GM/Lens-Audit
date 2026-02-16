import torch
import torch.optim as optim
from torch.utils.data import DataLoader
import os
from torchvision.utils import save_image

# Importing your Lead Modules
from src.utils.dataloader import GoProDataset
from src.ai.restormer import LensAuditNet
from src.ai.losses import LensAuditLoss

def train():
    # --- 1. Configuration ---
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    BATCH_SIZE = 4      # Adjust to 2 if you get 'Out of Memory'
    LEARNING_RATE = 2e-4
    EPOCHS = 50
    DATA_ROOT = r'C:\Users\Skanda Bharadwaj G M\Downloads\IPCV\data\gopro\GOPRO_Large'  # Update this path to your data location
    
    # Create folder for progress images
    os.makedirs('reports/figures', exist_ok=True)

    # --- 2. Data & Model Initialization ---
    print(f"Initializing Dataset from {DATA_ROOT}...")
    dataset = GoProDataset(root_dir=DATA_ROOT, split='train', size=256)
    train_loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)

    model = LensAuditNet(dim=48, num_blocks=4).to(device)
    if os.path.exists('lens_audit_latest.pth'):
    # map_location ensures it loads correctly even if you switch between CPU/GPU
        model.load_state_dict(torch.load('lens_audit_latest.pth', map_location=device))
        print("[RECOVERY] Successfully loaded Epoch 1 weights. Resuming training...")
    else:
        print("[INFO] No checkpoint found. Starting training from scratch.")
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)
    criterion = LensAuditLoss(fft_weight=0.1).to(device)

    print(f"Starting Training on {device}. Total Batches: {len(train_loader)}")

    # --- 3. Training Loop ---
    for epoch in range(EPOCHS):
        model.train()
        epoch_loss = 0
        
        for i, (blur, sharp) in enumerate(train_loader):
            blur, sharp = blur.to(device), sharp.to(device)

            # Optimization step
            optimizer.zero_grad()
            restored = model(blur)
            loss = criterion(restored, sharp)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

            # Progress Monitoring
            if i % 10 == 0:
                print(f"Epoch [{epoch+1}/{EPOCHS}] Step [{i}/{len(train_loader)}] Loss: {loss.item():.4f}")

            # Save visual progress every 500 steps
            if i % 500 == 0:
                comparison = torch.cat([blur[0:1], restored[0:1], sharp[0:1]], dim=3)
                save_image(comparison, f'reports/figures/progress_ep{epoch+1}_step{i}.png')

        # --- 4. Epoch Summary & Checkpoint ---
        avg_loss = epoch_loss / len(train_loader)
        print(f"--- Epoch {epoch+1} Finished. Avg Loss: {avg_loss:.4f} ---")
        
        # Save the Lead's trained weights
        torch.save(model.state_dict(), 'lens_audit_latest.pth')

if __name__ == "__main__":
    train()