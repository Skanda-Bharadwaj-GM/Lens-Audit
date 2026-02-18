import torch
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
import os
import math
import random
from torchvision import transforms
from torchvision.utils import save_image

# Lead Modules
from src.utils.dataloader import GoProDataset
from src.ai.restormer import LensAuditNet
from src.ai.losses import LensAuditLoss

def train():
    # --- 1. Configuration ---
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    BATCH_SIZE = 4      
    LEARNING_RATE = 5e-5 # Lowered for more stable fine-tuning
    TOTAL_EPOCHS = 15    
    LIMIT_IMAGES = 200   # The model sees 200 different images per run
    
    DATA_ROOT = r'C:\Users\Skanda Bharadwaj G M\Downloads\IPCV\data\gopro\GOPRO_Large'
    os.makedirs('reports/figures/Diverse_Subset', exist_ok=True)

    # --- 2. Advanced Augmentation ---
    # Adding flips helps prevent overfitting on a small subset
    transform = transforms.Compose([
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.5),
    ])

    # Note: Your current GoProDataset applies CenterCrop and ToTensor. 
    # We will wrap the output in these flips for Phase 2.
    full_dataset = GoProDataset(root_dir=DATA_ROOT, split='train', size=256)

    # --- 3. DIVERSITY LOGIC: Random Sampling ---
    # This picks 200 random indices from the whole dataset
    all_indices = list(range(len(full_dataset)))
    random.shuffle(all_indices)
    subset_indices = all_indices[:LIMIT_IMAGES]
    
    train_subset = Subset(full_dataset, subset_indices)
    train_loader = DataLoader(train_subset, batch_size=BATCH_SIZE, shuffle=True,num_workers=2, pin_memory=True)

    model = LensAuditNet(dim=48, num_blocks=4).to(device)
    
    if os.path.exists('lens_audit_best.pth'):
        model.load_state_dict(torch.load('lens_audit_best.pth', map_location=device))
        print(f"[*] Diverse Subset Mode: Training on {LIMIT_IMAGES} random images.")

    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-3) # Added weight decay
    criterion = LensAuditLoss(fft_weight=0.2).to(device)
    
    # Scheduler to slow down if loss plateaus
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)

    best_loss = float('inf')

    # --- 4. Loop ---
    for epoch in range(5, 5 + TOTAL_EPOCHS):
        model.train()
        epoch_loss = 0
        
        for i, (blur, sharp) in enumerate(train_loader):
            blur, sharp = blur.to(device), sharp.to(device)
            
            # Apply Augmentation manually if not in Dataloader
            if random.random() > 0.5:
                blur = torch.flip(blur, [3])
                sharp = torch.flip(sharp, [3])

            optimizer.zero_grad()
            restored = model(blur)
            loss = criterion(restored, sharp)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

            if i % 5 == 0:
                print(f"E[{epoch+1}] Step[{i}/{len(train_loader)}] Loss: {loss.item():.4f}")

        avg_loss = epoch_loss / len(train_loader)
        scheduler.step()
        
        # Save a sample to see progress
        comparison = torch.cat([blur[0:1], restored[0:1], sharp[0:1]], dim=3)
        save_image(comparison, f'reports/figures/Diverse_Subset/ep{epoch+1}.png')
        
        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save(model.state_dict(), 'lens_audit_diverse_best.pth')

if __name__ == "__main__":
    train()