import os
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from PIL import Image
from pprint import pprint
from torchvision import transforms, models


class RBenchmarking:
    def __init__(self, 
                 folder_path,
                 original_image_path,
                 model_name):
        self.folder_path = folder_path
        self.original_image_path = original_image_path
        self.model_name = model_name

        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        self.model = getattr(models, self.model_name)(weights="IMAGENET1K_V1")

        self.model = torch.nn.Sequential(*list(self.model.children())[:-1])
        self.model.to(self.device)
        self.model.eval()

        self.preprocess = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ])

        if not os.path.isfile(self.original_image_path):
            raise FileNotFoundError(
                f"Original image not found at: {self.original_image_path}"
            )
        self.original_image = self._load_image(self.original_image_path)

        self.original_features = self._compute_features(self.original_image)

    def _load_image(self, image_path: str) -> Image.Image:
        try:
            img = Image.open(image_path).convert("RGB")
            return img
        except Exception as e:
            print(f"Error loading image at {image_path}: {e}")
            return None

    def _compute_features(self, image: Image.Image) -> torch.Tensor:
        if image is None:
            return None

        tensor = self.preprocess(image).unsqueeze(0).to(self.device)

        with torch.no_grad():
            features = self.model(tensor)

        features = features.view(features.size(0), -1) 
        features = features / torch.norm(features, p=2, dim=1, keepdim=True)
        return features

    def compute_similarity(self, image_features: torch.Tensor) -> float:
        if image_features is None or self.original_features is None:
            return float("nan")

        similarity = F.cosine_similarity(self.original_features, image_features, dim=1)
        return similarity.item()

    def augment_image(self, image: Image.Image) -> dict:

        augmentations = {
            "original": image,
            "flipped_horizontally": image.transpose(Image.FLIP_LEFT_RIGHT),
            "rotated_20": image.rotate(20)
        }
        return augmentations

    def compute_augmented_similarities_for_all_images(self) -> dict:

        similarity_scores = {}

        for filename in os.listdir(self.folder_path):
            full_path = os.path.join(self.folder_path, filename)
            if not os.path.isfile(full_path) or not filename.lower().endswith((".png", ".jpg", ".jpeg", ".bmp")):
                continue 

            img = self._load_image(full_path)
            if img is None:
                continue

            augmented_images = self.augment_image(img)

            for aug_name, aug_image in augmented_images.items():
                features = self._compute_features(aug_image)
                score = self.compute_similarity(features)
                similarity_scores[f"{filename} ({aug_name})"] = score

        return similarity_scores

    def plot(self, sorted_results):

        if not sorted_results:
            print("No data to plot.")
            return

        filenames, scores = zip(*sorted_results)

        plt.figure(figsize=(20, 14)) 
        bars = plt.barh(filenames, scores, color='skyblue', edgecolor='black', label="Similarity Score")

        for bar, score in zip(bars, scores):
            plt.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height()/2,  
                    f'{score:.2f}', ha='left', va='center', fontsize=9, fontweight='bold', color='black')

        plt.title(f"Similarity Scores of Augmented Images {self.model_name}", fontsize=14, fontweight="bold")
        plt.xlabel("Cosine Similarity Score", fontsize=12)
        plt.ylabel("Augmented Images", fontsize=12)

        plt.grid(axis='x', linestyle='--', alpha=0.7) 
        plt.legend()
        plt.savefig(f"./Plots/Dist_{self.model_name}.jpg")
        plt.show()



if __name__ == "__main__":

    model_names = ['swin_t', 'swin_b', 'swin_s','swin_v2_t', 'swin_v2_b', 'swin_v2_s',]
    for model_name in model_names:
        rb = RBenchmarking(
            folder_path="Bathtub_images",
            original_image_path="Bathtub_images/original.png",
            model_name=model_name
        )

        aug_results = rb.compute_augmented_similarities_for_all_images()
        pprint(aug_results)

        sorted_aug_results = sorted(aug_results.items(), key=lambda x: x[1], reverse=False)

        rb.plot(sorted_aug_results)
