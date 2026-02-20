import torch
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
import os
import random
from torchvision import transforms
from torchvision.utils import save_image

# Lead Modules - Ensure these paths match your local structure
from src.utils.dataloader import GoProDataset
from src.ai.restormer import LensAuditNet
from src.ai.losses import LensAuditLoss

def train():
    # --- 1. Configuration & Memory Management ---
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    BATCH_SIZE = 1            # Process 1 image at a time to save VRAM
    ACCUMULATION_STEPS = 4    # Update weights every 4 images (Virtual Batch = 4)
    INITIAL_LR = 2e-5         # Low learning rate for fine-tuning
    TOTAL_EPOCHS = 10   
    LIMIT_IMAGES = 400        # High diversity subset
    
    DATA_ROOT = r'C:\Users\Skanda Bharadwaj G M\Downloads\IPCV\data\gopro\GOPRO_Large'
    os.makedirs('reports/figures/Final_Refinement', exist_ok=True)

    # --- 2. Dataset & Loader ---
    # size=384 provides the high-frequency context needed for 28dB+ clarity
    full_dataset = GoProDataset(root_dir=DATA_ROOT, split='train', size=384) 
    
    all_indices = list(range(len(full_dataset)))
    random.shuffle(all_indices)
    train_subset = Subset(full_dataset, all_indices[:LIMIT_IMAGES])
    train_loader = DataLoader(train_subset, batch_size=BATCH_SIZE, shuffle=True)

    # --- 3. Model & Loss Logic ---
    model = LensAuditNet(dim=48, num_blocks=4).to(device)
    
    if os.path.exists('lens_audit_diverse_best.pth'):
        model.load_state_dict(torch.load('lens_audit_diverse_best.pth', map_location=device))
        print("[*] Phase 2 Weights Loaded. Starting Sub-Pixel Refinement.")

    optimizer = optim.AdamW(model.parameters(), lr=INITIAL_LR, weight_decay=1e-2)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=TOTAL_EPOCHS)
    
    # High FFT weight (0.4) forces the model to recover sharp edges/text
    criterion = LensAuditLoss(fft_weight=0.4).to(device)

    best_loss = float('inf')

    # --- 4. Training Loop ---
    model.train()
    
    for epoch in range(21, 21 + TOTAL_EPOCHS):
        epoch_loss = 0
        optimizer.zero_grad() # Start each epoch with clean gradients
        
        for i, (blur, sharp) in enumerate(train_loader):
            blur, sharp = blur.to(device), sharp.to(device)

            # Forward Pass
            restored = model(blur)
            loss = criterion(restored, sharp)
            
            # Scale loss for gradient accumulation
            scaled_loss = loss / ACCUMULATION_STEPS 
            scaled_loss.backward()

            epoch_loss += loss.item()

            # Update weights only every 'ACCUMULATION_STEPS'
            if (i + 1) % ACCUMULATION_STEPS == 0:
                optimizer.step()
                optimizer.zero_grad()
                torch.cuda.empty_cache() # Clear VRAM to prevent OOM

            if i % 20 == 0:
                print(f"E[{epoch+1}] Step[{i}/{len(train_loader)}] Loss: {loss.item():.4f} | LR: {optimizer.param_groups[0]['lr']:.2e}")

        avg_epoch_loss = epoch_loss / len(train_loader)
        scheduler.step()
        
        # Save visual progress comparison
        with torch.no_grad():
            comparison = torch.cat([blur[0:1], restored[0:1], sharp[0:1]], dim=3)
            save_image(comparison, f'reports/figures/Final_Refinement/ep{epoch+1}_refinement.png')
        
        if avg_epoch_loss < best_loss:
            best_loss = avg_epoch_loss
            torch.save(model.state_dict(), 'lens_audit_final_polish.pth')
            print(f"[*] New Best Loss: {best_loss:.4f} - Weights Saved.")

if __name__ == "__main__":
    train()