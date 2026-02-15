from src.utils.dataloader import GoProDataset
import matplotlib.pyplot as plt

# Update this path to where your data actually is
dataset = GoProDataset(root_dir='data/gopro/GOPRO_Large', split='train')

print(f"Total image pairs found: {len(dataset)}")

# Pull one pair to verify
blur, sharp = dataset[0]
plt.subplot(121), plt.imshow(blur.permute(1, 2, 0)), plt.title("Blur Input")
plt.subplot(122), plt.imshow(sharp.permute(1, 2, 0)), plt.title("Sharp Target")
plt.show()