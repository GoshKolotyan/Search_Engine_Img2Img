import os
import torch
import random
import matplotlib.pyplot as plt
import torchvision.utils as vutils
import torchvision.transforms as transforms

from PIL import Image
from torch.utils.data import Dataset, DataLoader

class SiameseDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root = root_dir
        self.transform = transform
        self.data = {}

        for category in os.listdir(self.root):
            category_path = os.path.join(self.root, category)
            self.data[category] = {}
            for subclass in os.listdir(category_path):
                subclass_path = os.path.join(category_path, subclass)
                img_paths = [os.path.join(subclass_path, img) for img in os.listdir(subclass_path)]
                self.data[category][subclass] = img_paths

        self.categories = list(self.data.keys())

    def __len__(self):
        return sum(len(subclasses) for subclasses in self.data.values()) * 100  # good sampling

    def __getitem__(self):
        category = random.choice(self.categories)
        subclasses = list(self.data[category].keys())

        same_class = random.choice([True, False])

        if same_class or len(subclasses) == 1:
            subclass = random.choice(subclasses)
            img1, img2 = random.sample(self.data[category][subclass], 2)
            label = 1.0
        else:
            subclass1, subclass2 = random.sample(subclasses, 2)
            img1 = random.choice(self.data[category][subclass1])
            img2 = random.choice(self.data[category][subclass2])
            label = 0.0

        img1 = Image.open(img1).convert('RGB')
        img2 = Image.open(img2).convert('RGB')

        if self.transform:
            img1 = self.transform(img1)
            img2 = self.transform(img2)

        return img1, img2, torch.tensor([label], dtype=torch.float32)


if __name__ == "__main__":
    root_dir = 'data'

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

    siamese_dataset = SiameseDataset(root_dir=root_dir, transform=transform)
    dataloader = DataLoader(siamese_dataset, batch_size=16, shuffle=True)

    # Visualization
    img1_batch, img2_batch, labels_batch = next(iter(dataloader))
    img_grid1 = vutils.make_grid(img1_batch, normalize=True, scale_each=True)
    img_grid2 = vutils.make_grid(img2_batch, normalize=True, scale_each=True)

    plt.figure(figsize=(12,6))
    plt.subplot(2,1,1)
    plt.title(f"Batch of Images 1 (Label: {labels_batch.numpy().flatten()})")
    plt.imshow(img_grid1.permute(1, 2, 0))
    plt.axis('off')

    plt.subplot(2,1,2)
    plt.title(f"Batch of Images 2 (Label: {labels_batch.numpy().flatten()})")
    plt.imshow(img_grid2.permute(1, 2, 0))
    plt.axis('off')

    plt.show()