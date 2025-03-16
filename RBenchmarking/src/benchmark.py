import os
import csv
import torch
import logging
import torch.nn.functional as F
import matplotlib.pyplot as plt
import seaborn as sns

from PIL import Image
from typing import Dict
from warnings import filterwarnings
from torchvision import transforms, models
from torchvision.models.feature_extraction import create_feature_extractor

filterwarnings("ignore")


class RBenchmarking:
    def __init__(self, folder_path: str, model_name: str, output_dir: str):
        self.output_dir = output_dir
        self.folder_path = folder_path
        self.original_image_path = self.folder_path + "/original.jpg"
        if not os.path.isfile(self.original_image_path):
            raise FileNotFoundError(
                f"Original image not found at: {self.original_image_path}"
            )

        self.original_image = self._load_image(self.original_image_path)
        self.model_name = model_name

        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        # Different handling for ViTs vs. Dino vs. CNNs
        if "vit_" in self.model_name:
            # Extract features from transformer’s last hidden state
            return_nodes = {"encoder.ln": "features"}
            self.model = getattr(models, self.model_name)(weights="IMAGENET1K_V1")
            self.model = create_feature_extractor(self.model, return_nodes=return_nodes).to(self.device).eval()
    
            self.preprocess = transforms.Compose(
                [
                    transforms.Resize((224, 224)),
                    transforms.ToTensor(),
                    transforms.Normalize(
                        mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225],
                    ),
                ]
            )
            self.original_features = self._compute_features_vit(
                image=self.original_image
            )

        elif "dinov2" in self.model_name:
            logging.info(f"Loading model {self.model_name}")
            self.model = torch.hub.load("facebookresearch/dinov2", self.model_name)
            self.model = self.model.to(self.device).eval()
            self.preprocess = transforms.Compose(
                [
                    transforms.Resize((518, 518)),  # Official DINOv2 input size
                    transforms.ToTensor(),
                    transforms.Normalize(
                        mean=[0.485, 0.456, 0.406], 
                        std=[0.229, 0.224, 0.225]
                    ),
                ]
            )
            self.original_features = self._compute_features_dino(
                image=self.original_image
            )

        else:
            # Extract features from the last convolutional layer
            self.model = getattr(models, self.model_name)(weights="IMAGENET1K_V1")
            self.model = torch.nn.Sequential(*list(self.model.children())[:-1])
            self.model = self.model.to(self.device).eval()
            self.preprocess = transforms.Compose(
                [
                    transforms.Resize((224, 224)),
                    transforms.ToTensor(),
                    transforms.Normalize(
                        mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225],
                    ),
                ]
            )
            self.original_features = self._compute_features(
                image=self.original_image
            )

    def _load_image(self, image_path: str) -> Image.Image:
        try:
            img = Image.open(image_path).convert("RGB")
            return img
        except Exception as e:
            logging.error(f"Error loading image at {image_path}: {e}")
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

    def _compute_features_vit(self, image: Image.Image) -> torch.Tensor:
        if image is None:
            return None

        tensor = self.preprocess(image).unsqueeze(0).to(self.device)

        with torch.no_grad():
            # This returns a dictionary
            output_dict = self.model(tensor)

        features = output_dict["features"]

        features = features.view(features.size(0), -1)
        features = features / torch.norm(features, p=2, dim=1, keepdim=True)
        return features

    def _compute_features_dino(self, image: Image.Image) -> torch.Tensor:
        if image is None:
            return None
        input_tensor = self.preprocess(image).unsqueeze(0).to(self.device)
        with torch.no_grad(), torch.autocast(device_type="cuda"):
            features = self.model.backbone.forward_features(input_tensor)
            embeddings = features["x_norm_clstoken"]
        return F.normalize(embeddings, p=2, dim=-1).squeeze(0)

    def compute_similarity(self, image_features: torch.Tensor) -> float:
        if image_features is None or self.original_features is None:
            return float("nan")

        if image_features.dim() > 1:
            image_features = image_features.squeeze()

        return F.cosine_similarity(
            self.original_features.unsqueeze(0), image_features.unsqueeze(0), dim=1
        ).item() if "dinov2" in self.model_name  else  F.cosine_similarity(self.original_features, image_features, dim=1).item()


    def augment_image(self, image: Image.Image) -> Dict:
        width, height = image.size

        img_left_area = (0, 0, width // 2, height)
        img_right_area = (width // 2, 0, width, height)
        img_top_area = (0, 0, width, height // 2)
        img_bottom_area = (0, height // 2, width, height)

        augmentations = {
            "original": image,
            "flipped_horizontally": image.transpose(Image.FLIP_LEFT_RIGHT),
            "rotated_20": image.rotate(20),
            "cropped_from_left": image.crop(img_left_area),
            "cropped_from_right": image.crop(img_right_area),
            "cropped_from_top": image.crop(img_top_area),
            "cropped_from_bottom": image.crop(img_bottom_area),
        }
        return augmentations

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
            f"../{self.output_dir}/{self.folder_path.split('/')[-1]}/{self.model_name}"
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
            f"../{self.output_dir}/{self.folder_path.split('/')[-1]}/{self.model_name}"
        )
        if not os.path.exists(save_path):
            os.makedirs(save_path, exist_ok=True)

        plt.savefig(save_path + f"/Similarity_Heatmap_{self.model_name}.jpg")
        plt.close(fig)

        logging.info(
            f"Saved heatmap in {save_path}/"
            + f"Similarity_Heatmap_{self.model_name}.jpg"
        )

    def _record_to_csv(
        self, similarity_scores: Dict[str, int], model_name: str
    ) -> None:
        save_path = f"../{self.output_dir}/{self.folder_path.split('/')[-1]}"
        csv_filename = f"{save_path}/{self.folder_path.split('/')[-1]}_records.csv"

        with open(csv_filename, "a", newline="") as f:  # Use 'a' to append results
            records = csv.writer(
                f, delimiter=",", quotechar='"', quoting=csv.QUOTE_MINIMAL
            )

            if f.tell() == 0:
                records.writerow(["Model Name", "Similarity Score"])

            records.writerow([model_name, similarity_scores])

    def compute_augmented_similarities_for_all_images(self) -> Dict[str, int]:
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
                if "vit_" in self.model_name:
                    features = self._compute_features_vit(aug_image)
                elif "dinov2" in self.model_name:
                    features = self._compute_features_dino(
                        image=aug_image
                    )
                else:
                    features = self._compute_features(aug_image)
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
            f"../{self.output_dir}/{self.folder_path.split('/')[-1]}/{self.model_name}"
        )
        if not os.path.exists(save_path):
            os.mkdir(save_path)
        plt.savefig(save_path + f"/Dist_{self.model_name}.jpg")
        plt.close()
        logging.info(f"Saved in {save_path}" + f"/Dist_{self.model_name}.jpg")
        # plt.show()
