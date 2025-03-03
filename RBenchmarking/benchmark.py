import os
import csv
import glob
import torch
import numpy as np
import pandas as pd
import torch.nn.functional as F
import matplotlib.pyplot as plt
import logging
from PIL import Image
from torchvision import transforms, models

os.makedirs("plots", exist_ok=True)
os.makedirs("logs", exist_ok=True)

logging.basicConfig(
    format="{asctime} - {levelname} - {message}",
    level=logging.INFO,
    style="{",
    datefmt="%Y-%m-%d %H:%M",
    filename="./logs/errores.log",
)

logging.info("Application started successfully.")


class RBenchmarking:
    def __init__(self, folder_path, model_name):
        self.folder_path = folder_path
        self.original_image_path = self.folder_path + "/original.jpg"
        self.model_name = model_name

        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        self.model = getattr(models, self.model_name)(weights="IMAGENET1K_V1")

        self.model = torch.nn.Sequential(*list(self.model.children())[:-1])
        self.model.to(self.device)
        self.model.eval()

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
            logging.info(f"Error loading image at {image_path}: {e}")
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

    def visualize_all_images(self, all_augmented_images, similarity_scores):
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
        save_path = f"./plots/{self.folder_path.split('/')[-1]}/{self.model_name}"
        if not os.path.exists(save_path):
            os.makedirs(save_path, exist_ok=True)
        plt.savefig(save_path + f"/Compar_all_images_{self.model_name}.jpg")
        plt.close(fig)

        logging.info(f"Saved in {save_path}" + f"Compare_all_{self.model_name}.jpg")

        # plt.show()

    def augment_image(self, image: Image.Image) -> dict:
        width, height = image.size

        img_left_area = (0, 0, width//2, height)
        img_right_area = (width//2, 0, width, height)
        img_top_area = (0, 0, width, height//2) 
        img_bottom_area = (0, height//2, width, height)  

        augmentations = {
            "original": image,
            "flipped_horizontally": image.transpose(Image.FLIP_LEFT_RIGHT),
            "rotated_20": image.rotate(20),
            "cropped_from_left": image.crop(img_left_area),
            "cropped_from_right": image.crop(img_right_area),
            "cropped_from_top": image.crop(img_top_area), 
            "cropped_from_bottom": image.crop(img_bottom_area) 
        }
        return augmentations


    def _record_to_csv(self, similarity_scores, model_name):
        save_path = f"./plots/{self.folder_path.split('/')[-1]}"
        csv_filename = f"{save_path}/{self.folder_path.split('/')[-1]}_records.csv"

        with open(csv_filename, "a", newline="") as f:  # Use 'a' to append results
            records = csv.writer(
                f, delimiter=",", quotechar='"', quoting=csv.QUOTE_MINIMAL
            )

            if f.tell() == 0:
                records.writerow(["Model Name", "Similarity Score"])

            records.writerow([model_name, similarity_scores])

    def compute_augmented_similarities_for_all_images(self) -> dict:
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
                features = self._compute_features(aug_image)
                score = self.compute_similarity(features)
                similarity_scores[f"{filename} ({aug_name})"] = score
        self.visualize_all_images(all_augmented_images, similarity_scores)

        return similarity_scores

    # TODO crop image by sizes

    def plot(self, sorted_results):

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
        save_path = f"./plots/{self.folder_path.split('/')[-1]}/{self.model_name}"
        if not os.path.exists(save_path):
            os.mkdir(save_path)
        plt.savefig(save_path + f"/Dist_{self.model_name}.jpg")
        plt.close()
        logging.info(f"Saved in {save_path}" + f"/Dist_{self.model_name}.jpg")
        # plt.show()


class Analyzer:
    def __init__(self, paths):
        self.paths = paths

    def read_csv(self, path):
        """Reads a CSV file and returns a DataFrame."""
        try:
            logging.info(f"Loading csv from {path}")
            return pd.read_csv(path)
        except Exception as e:
            logging.info(f"Error reading {path}: {e}")
            return None

    def _plot(self):
        dfs = {path: self.read_csv(path) for path in self.paths}
        dfs = {path: df for path, df in dfs.items() if df is not None}

        if not dfs:
            logging.info("No valid CSV files to analyze.")
            return

        plt.figure(figsize=(30, 20), dpi=200)

        num_csvs = len(dfs)
        bar_width = 0.1
        index = np.arange(len(next(iter(dfs.values()))["Model Name"]))

        for i, (file_path, df) in enumerate(dfs.items()):
            if "Model Name" not in df.columns or "Similarity Score" not in df.columns:
                logging.info(f"Skipping {file_path}: Required columns missing.")
                continue

            df = df.sort_values("Model Name")

            model_names = df["Model Name"].values
            scores = df["Similarity Score"].values

            x_positions = index + (i * bar_width)

            bars = plt.bar(
                x_positions,
                scores,
                width=bar_width,
                label=f"File {i+1}: {file_path.split('/')[-1]}",
                alpha=0.7,
            )

            for bar in bars:
                height = bar.get_height()
                plt.text(
                    bar.get_x() + bar.get_width() / 2,
                    height + 0.01,
                    f"{height:.2f}",
                    ha="center",
                    fontsize=10,
                )

        plt.xlabel("Model Name")
        plt.ylabel("Similarity Score")
        plt.title("Model Similarity Scores from Multiple CSV Files")
        plt.xticks(
            index + (num_csvs * bar_width) / 2, model_names, rotation=0, ha="right"
        )
        plt.grid(visible=True)
        plt.legend()
        plt.tight_layout(pad=4)
        plt.savefig(f"plots/barplots.jpg")
        plt.close()
        # plt.show()

    def _avg_plot(self):

        valid_dfs = []
        for path in self.paths:
            df = self.read_csv(path)
            if df is None:
                logging.info(f"Skipping {path}: Could not read file.")
                continue
            if "Model Name" not in df.columns or "Similarity Score" not in df.columns:
                logging.info(f"Skipping {path}: Required columns missing.")
                continue
            valid_dfs.append(df)

        if not valid_dfs:
            logging.info("No valid CSV files to analyze.")
            return

        all_data = pd.concat(valid_dfs, ignore_index=True)

        avg_scores = (
            all_data.groupby("Model Name")["Similarity Score"]
            .mean()
            .reset_index()
            .sort_values("Model Name")
        )
        plt.figure(figsize=(30, 20), dpi=200)
        x_positions = np.arange(len(avg_scores))
        scores = avg_scores["Similarity Score"].values
        model_names = avg_scores["Model Name"].values

        bars = plt.bar(x_positions, scores, alpha=0.7)

        for bar in bars:
            height = bar.get_height()
            plt.text(
                bar.get_x() + bar.get_width() / 2,
                height + 0.01,
                f"{height:.2f}",
                ha="center",
                fontsize=10,
            )

        plt.xlabel("Model Name")
        plt.ylabel("Average Similarity Score")
        plt.title("Average Model Similarity Scores Across CSV Files")
        plt.xticks(x_positions, model_names, rotation=0, ha="right")
        plt.grid(True)
        plt.legend(["Avg Similarity Score"])
        plt.savefig("plots/avg_scores.jpg")
        plt.close()
        # plt.show()

    def __call__(self):
        self._plot()
        self._avg_plot()


if __name__ == "__main__":
    model_names = ["swin_t", "swin_b", "swin_s", "swin_v2_t", "swin_v2_b", "swin_v2_s"]

    images_dir = "Test Images"

    for filename in os.listdir(images_dir):
        folder_path = os.path.join(images_dir, filename)

        for model_name in model_names:
            logging.info(f"Running model {model_name}")

            rb = RBenchmarking(
                folder_path=folder_path,
                model_name=model_name,
            )

            aug_results = rb.compute_augmented_similarities_for_all_images()
            # Must be added here
            sorted_aug_results = sorted(
                aug_results.items(), key=lambda x: x[1], reverse=False
            )

            rb.plot(sorted_results=sorted_aug_results)

            sum_score = [res[1] for res in sorted_aug_results]

            average_score = sum(sum_score) / len(sum_score)
            rb._record_to_csv(similarity_scores=average_score, model_name=model_name)
            # break
        # break

    analyzer = Analyzer(paths="plots")

    csv_dir = "plots"
    csv_files = glob.glob(os.path.join(csv_dir, "**/*.csv"), recursive=True)

    analyzer = Analyzer(paths=csv_files)
    analyzer()
