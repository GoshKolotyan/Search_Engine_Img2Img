import os
import torch
import random
import matplotlib.pyplot as plt
import torchvision.utils as vutils
import torchvision.transforms as transforms

from PIL import Image
from torch.utils.data import Dataset, DataLoader

class SiameseDataset(Dataset):
    def __init__(self, root_dir, transform=None, negative_ratio=1.0, seed=42):
        super().__init__()
        self.root = root_dir
        self.transform = transform
        self.data = {}
        self.seed = seed
        self.negative_ratio = negative_ratio


        for category in os.listdir(self.root):
            category_path = os.path.join(self.root, category)
            if not os.path.isdir(category_path):
                continue 
            self.data[category] = {}
            for subclass in os.listdir(category_path):
                subclass_path = os.path.join(category_path, subclass)
                if not os.path.isdir(subclass_path):
                    continue
                img_paths = [
                    os.path.join(subclass_path, img)
                    for img in os.listdir(subclass_path)
                    if os.path.isfile(os.path.join(subclass_path, img))
                ]
                self.data[category][subclass] = img_paths

        self.subclass_keys = []
        for category, subclasses in self.data.items():
            for subclass in subclasses:
                self.subclass_keys.append((category, subclass))

        # A positive pair = two images from the same category & subclass
        self.pos_pairs = []
        for cat_subclass in self.subclass_keys:
            category, subclass = cat_subclass
            img_list = self.data[category][subclass]
            if len(img_list) < 2:
                continue
            for i in range(len(img_list)):
                for j in range(i + 1, len(img_list)):
                    self.pos_pairs.append((img_list[i], img_list[j], 1.0))

        # A negative pair = two images from the same category, different subclass
        self.neg_pairs = []
        random.seed(self.seed)
        all_subclass_keys_by_cat = {}
        for category, subclasses in self.data.items():
            all_subclass_keys_by_cat[category] = list(subclasses.keys())

        # We'll randomly pick negative pairs in the same category
        num_neg_needed = int(len(self.pos_pairs) * self.negative_ratio)
        neg_pairs_temp = []
        cats = list(self.data.keys())

        # Generate random negatives:
        #   1) pick a category
        #   2) pick two distinct subclasses in that category
        #   3) pick one image from each subclass
        #   4) label=0
        while len(neg_pairs_temp) < num_neg_needed:
            category = random.choice(cats)
            subclasses = list(self.data[category].keys())
            if len(subclasses) < 2:
                continue
            subclass1, subclass2 = random.sample(subclasses, 2)
            imgs1 = self.data[category][subclass1]
            imgs2 = self.data[category][subclass2]
            if len(imgs1) == 0 or len(imgs2) == 0:
                continue
            img1 = random.choice(imgs1)
            img2 = random.choice(imgs2)
            neg_pairs_temp.append((img1, img2, 0.0))

        self.neg_pairs = neg_pairs_temp

        # Combining positive and negative pairs into one 
        self.pairs = self.pos_pairs + self.neg_pairs
        random.shuffle(self.pairs)  

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, index):
        img_path1, img_path2, label = self.pairs[index]

        img1 = Image.open(img_path1).convert("RGB")
        img2 = Image.open(img_path2).convert("RGB")

        if self.transform:
            img1 = self.transform(img1)
            img2 = self.transform(img2)

        return img1, img2, torch.tensor([label], dtype=torch.float32)

    def __contains__(self, item):
        return item in self.pairs

    def __repr__(self):
        return f"SiameseDataset(num_pairs={len(self)})"


if __name__ == "__main__":
    root_dir = "data"

    transform = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ]
    )

    siamese_dataset = SiameseDataset(root_dir, transform=transform, negative_ratio=1.0, seed=42)
    print(siamese_dataset)

    dataloader = DataLoader(
        dataset=siamese_dataset, 
        batch_size=3, 
        num_workers=8, 
        shuffle=True, 
        pin_memory=True, #Change false for CPU computing stand for converting from CPU -> GPU 
    )  

    img1_batch, img2_batch, labels_batch = next(iter(dataloader))


    img_grid1 = vutils.make_grid(img1_batch, normalize=True, scale_each=True)
    img_grid2 = vutils.make_grid(img2_batch, normalize=True, scale_each=True)

    plt.figure(figsize=(12, 6))
    plt.subplot(2, 1, 1)
    plt.title(f"Batch of Images 1 (Labels: {labels_batch.squeeze().tolist()})")
    plt.imshow(img_grid1.permute(1, 2, 0))
    plt.axis("off")

    plt.subplot(2, 1, 2)
    plt.title(f"Batch of Images 2 (Labels: {labels_batch.squeeze().tolist()})")
    plt.imshow(img_grid2.permute(1, 2, 0))
    plt.axis("off")

    plt.show()
