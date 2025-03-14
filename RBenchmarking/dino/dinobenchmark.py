import os
import csv
import torch
import logging
import seaborn as sns
import matplotlib.pyplot as plt
import torch.nn.functional as F

from typing import Dict
from PIL import Image
from torchvision import transforms


class DinoBenchmarking:
    def __init__(self, model_name: str, folder_path: str, output_dir: str):
        # Validate inputs
        if not os.path.isdir(folder_path):
            raise ValueError(f"Invalid folder path: {folder_path}")
        if not os.path.isdir(output_dir):
            os.makedirs(output_dir, exist_ok=True)

        # Initialize paths and device
        self.model_name = model_name
        self.output_dir = output_dir
        self.folder_path = folder_path
        self.original_image_path = os.path.join(folder_path, "original.jpg")
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Initialize model with proper configuration
        self.model = torch.hub.load("facebookresearch/dinov2", self.model_name)
        self.model = self.model.to(
            self.device
        ).eval()  # Move to device and set eval mode

        # DINOv2-specific transforms (correct image size and normalization)
        self.transform = transforms.Compose(
            [
                transforms.Resize((518, 518)),  # Official DINOv2 input size
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
        )

        # Load original image with validation
        self.original_image = self._load_image(self.original_image_path)
        if self.original_image is None:
            raise RuntimeError(
                f"Failed to initialize: Could not load original image from {self.original_image_path}"
            )

        self.original_features = self.extract_embedding(self.original_image)

    def _load_image(self, image_path: str) -> Image.Image:
        try:
            img = Image.open(image_path).convert("RGB")
            return img
        except Exception as e:
            logging.error(f"Error loading image at {image_path}: {e}")
            return None

    def extract_embedding(self, image: Image.Image) -> torch.Tensor:
        """Extract normalized embeddings from DINOv2 model."""
        try:
            input_tensor = self.transform(image).unsqueeze(0).to(self.device)
            with torch.no_grad(), torch.autocast(device_type='cuda'):
                features = self.model.backbone.forward_features(input_tensor)
                embeddings = features["x_norm_clstoken"]
            return F.normalize(embeddings, p=2, dim=-1).squeeze(0)
        except Exception as e:
            logging.error(f"Embedding extraction failed: {e}")
            return torch.tensor([])

    def compute_similarity(self, image_features: torch.Tensor) -> float:
        """Compute cosine similarity with proper dimension handling."""
        if image_features is None or self.original_features is None:
            return float("nan")

        if image_features.dim() > 1:
            image_features = image_features.squeeze()

        return F.cosine_similarity(
            self.original_features.unsqueeze(0), image_features.unsqueeze(0), dim=1
        ).item()

    def augment_image(self, image: Image.Image) -> Dict[str, Image.Image]:
        """Generate augmentation variants with size maintenance."""
        try:
            width, height = image.size
            return {
                "original": image,
                "flipped_horizontally": image.transpose(Image.FLIP_LEFT_RIGHT),
                "rotated_20": image.rotate(20, expand=True),  # Maintain full image
                "cropped_left": image.crop((0, 0, width // 2, height)),
                "cropped_right": image.crop((width // 2, 0, width, height)),
                "cropped_top": image.crop((0, 0, width, height // 2)),
                "cropped_bottom": image.crop((0, height // 2, width, height)),
            }
        except Exception as e:
            logging.error(f"Augmentation failed: {e}")
            return {}

    def visualize_all_images(
        self,
        all_augmented_images: Dict[str, Image.Image],
        similarity_scores: Dict[str, int],
    ) -> None:
        num_images = len(all_augmented_images)
        num_augmentations = len(next(iter(all_augmented_images.values())))
        fig, axes = plt.subplots(
            num_images, num_augmentations, figsize=(10, 3 * num_images), dpi=100
        )

        if num_images == 1:
            axes = [axes]

        for row, (filename, augmented_images) in enumerate(
            all_augmented_images.items()
        ):
            for col, (aug_name, aug_image) in enumerate(augmented_images.items()):
                ax = axes[row][col] if num_images > 1 else axes[col]
                ax.imshow(aug_image)
                score_key = f"{filename} ({aug_name})"
                score = similarity_scores.get(score_key, "N/A")
                ax.set_title(f"{aug_name}\nScore: {score:.2f}", fontsize=8, pad=3)
                ax.axis("off")

        plt.suptitle(
            f"model name {self.model_name} {self.original_image_path}",
            fontsize=14,
            fontweight="bold",
        )
        plt.tight_layout(pad=4)
        save_path = (
            f"{self.output_dir}/{self.folder_path.split('/')[-1]}/{self.model_name}"
        )
        if not os.path.exists(save_path):
            os.makedirs(save_path, exist_ok=True)
        plt.savefig(save_path + f"/Compar_all_images_{self.model_name}.jpg")
        plt.close(fig)

        logging.info(f"Saved in {save_path}/" + f"Compare_all_{self.model_name}.jpg")

        # plt.show()

    def visualize_similarity_heatmap(
        self,
        all_augmented_images: Dict[str, Image.Image],
        similarity_scores: Dict[str, int],
    ) -> None:
        filenames = list(all_augmented_images.keys())
        augmentations = list(next(iter(all_augmented_images.values())).keys())

        num_images = len(filenames)
        num_augmentations = len(augmentations)

        heatmap_data = torch.zeros((num_images, num_augmentations))

        for i, filename in enumerate(filenames):
            for j, aug_name in enumerate(augmentations):
                score_key = f"{filename} ({aug_name})"
                heatmap_data[i, j] = similarity_scores.get(score_key, torch.nan)

        fig, ax = plt.subplots(
            figsize=(num_augmentations * 1.5, num_images * 1.5), dpi=100
        )
        sns.heatmap(
            heatmap_data,
            annot=True,
            xticklabels=augmentations,
            yticklabels=filenames,
            cmap="coolwarm",
            linewidths=0.5,
            fmt=".2f",
            ax=ax,
        )

        plt.title(
            f"Similarity Scores Heatmap - {self.model_name}",
            fontsize=14,
            fontweight="bold",
        )
        plt.xlabel("Augmentations")
        plt.ylabel("Original Images")
        plt.yticks(rotation=20)
        plt.xticks(rotation=20)

        save_path = (
            f"{self.output_dir}/{self.folder_path.split('/')[-1]}/{self.model_name}"
        )
        if not os.path.exists(save_path):
            os.makedirs(save_path, exist_ok=True)

        plt.savefig(save_path + f"/Similarity_Heatmap_{self.model_name}.jpg")
        plt.close(fig)

        logging.info(
            f"Saved heatmap in {save_path}/"
            + f"Similarity_Heatmap_{self.model_name}.jpg"
        )

    def _record_to_csv(self, similarity_scores: Dict[str, int], model_name: str):
        save_path = f"{self.output_dir}/{self.folder_path.split('/')[-1]}"
        csv_filename = f"{save_path}/{self.folder_path.split('/')[-1]}_records.csv"

        with open(csv_filename, "a", newline="") as f:  # Use 'a' to append results
            records = csv.writer(
                f, delimiter=",", quotechar='"', quoting=csv.QUOTE_MINIMAL
            )

            if f.tell() == 0:
                records.writerow(["Model Name", "Similarity Score"])

            records.writerow([model_name, similarity_scores])

    def compute_augmented_similarities_for_all_images(self) -> Dict[str, float]:
        similarity_scores = {}
        all_augmented_images = {}

        for filename in os.listdir(self.folder_path):
            full_path = os.path.join(self.folder_path, filename)
            if not os.path.isfile(full_path) or not filename.lower().endswith(
                (".png", ".jpg", ".jpeg", ".bmp")
            ):
                continue

            img = self._load_image(full_path)
            if img is None:
                continue

            augmented_images = self.augment_image(img)
            all_augmented_images[filename] = augmented_images

            for aug_name, aug_image in augmented_images.items():
                features = self.extract_embedding(aug_image)  # Fixed method call
                score = self.compute_similarity(features)
                similarity_scores[f"{filename} ({aug_name})"] = score

        self.visualize_all_images(all_augmented_images, similarity_scores)
        self.visualize_similarity_heatmap(all_augmented_images, similarity_scores)

        return similarity_scores

    def plot(self, sorted_results: Dict[str, int]):

        filenames, scores = zip(*sorted_results)

        plt.figure(figsize=(20, max(14, len(filenames) * 0.5)))
        bars = plt.barh(
            filenames,
            scores,
            color="skyblue",
            edgecolor="black",
            label="Similarity Score",
        )

        for bar, score in zip(bars, scores):
            plt.text(
                bar.get_width() + 0.01,
                bar.get_y() + bar.get_height() / 2,
                f"{score:.2f}",
                ha="left",
                va="center",
                fontsize=9,
                fontweight="bold",
                color="black",
            )

        plt.title(
            f"Similarity Scores of Augmented Images {self.model_name}",
            fontsize=14,
            fontweight="bold",
        )
        plt.xlabel("Cosine Similarity Score", fontsize=12)
        plt.ylabel("Augmented Images", fontsize=12)

        plt.yticks(fontsize=10)
        plt.gca().invert_yaxis()

        plt.grid(axis="x", linestyle="--", alpha=0.7)
        plt.legend()

        plt.tight_layout()
        save_path = (
            f"{self.output_dir}/{self.folder_path.split('/')[-1]}/{self.model_name}"
        )
        if not os.path.exists(save_path):
            os.mkdir(save_path)
        plt.savefig(save_path + f"/Dist_{self.model_name}.jpg")
        plt.close()
        logging.info(f"Saved in {save_path}" + f"/Dist_{self.model_name}.jpg")
        # plt.show()
