import os
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as T

class GoProDataset(Dataset):
    def __init__(self, root_dir, split='train', size=256):
        self.root_dir = os.path.join(root_dir, split)
        self.pairs = []
        
        # Walk through all scene folders
        for scene in os.listdir(self.root_dir):
            scene_path = os.path.join(self.root_dir, scene)
            blur_dir = os.path.join(scene_path, 'blur')
            sharp_dir = os.path.join(scene_path, 'sharp')
            
            if os.path.exists(blur_dir):
                for img_name in os.listdir(blur_dir):
                    self.pairs.append((
                        os.path.join(blur_dir, img_name),
                        os.path.join(sharp_dir, img_name)
                    ))

        self.transform = T.Compose([
            T.CenterCrop(size),
            T.ToTensor()
        ])

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        blur_path, sharp_path = self.pairs[idx]
        return self.transform(Image.open(blur_path).convert('RGB')), \
               self.transform(Image.open(sharp_path).convert('RGB'))

print(f"[SUCCESS] Data Loader Script Ready.")