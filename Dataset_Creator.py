import os
import pickle
from tqdm import tqdm
import torch
from torchvision import transforms, models
from PIL import Image
import numpy as np
from torchvision.transforms.functional import rotate, hflip
from annoy import AnnoyIndex


class ModelDatabaseBuilder:
    def __init__(self, model_name, base_directory):
        """
        Initializes the database builder by setting up the model, transformations, and directories.

        Args:
            model_name (str): Name of the pre-trained model to use for feature extraction.
            base_directory (str): Path to the dataset directory.
        """
        self.model_name = model_name
        self.base_directory = base_directory

        # Load the pre-trained model
        self.model = getattr(models, model_name)(weights="IMAGENET1K_V1")
        self.model.eval()

        # Remove the classification head
        self.model = torch.nn.Sequential(*list(self.model.children())[:-1])

        # Determine the feature dimension dynamically
        dummy_input = torch.zeros(1, 3, 224, 224)  # Example input tensor
        with torch.no_grad():
            feature_vector = self.model(dummy_input)
        self.feature_dim = feature_vector.view(-1).shape[0]

        self.preprocess = transforms.Compose(
            [
                transforms.Resize((224, 224)),  # Resize to match input size
                transforms.ToTensor(),  # Convert image to tensor
                transforms.Normalize(  # Normalize using ImageNet stats
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
        )

        self.database_features = []
        self.database_metadata = []

    def extract_features(self, image):
        """
        Extracts feature vectors from an image object using the pre-trained model.

        Args:
            image (PIL.Image.Image): Image object.

        Returns:
            np.ndarray: Normalized feature vector.
        """
        image_tensor = self.preprocess(image).unsqueeze(0)
        with torch.no_grad():
            features = self.model(image_tensor).squeeze().numpy()
        features /= np.linalg.norm(features)
        return features

    def build_database(self):
        """
        Iterates over the dataset, extracts features (with augmentation), and stores metadata.
        """
        print(f"Building database from directory: {self.base_directory}")
        for category in os.listdir(self.base_directory):
            category_path = os.path.join(self.base_directory, category)
            if os.path.isdir(category_path):
                for filename in tqdm(
                    os.listdir(category_path), desc=f"Processing {category}"
                ):
                    if filename.endswith((".png", ".jpg", ".jpeg")):
                        image_path = os.path.join(category_path, filename)

                        # Load the original image
                        image = Image.open(image_path).convert("RGB")

                        # Extract features for the original image
                        original_features = self.extract_features(image)
                        self.add_to_database(original_features, category, filename, image_path)

                        # Generate and process augmented versions
                        augmented_versions = [
                            ("rotated_15", rotate(image, 15)),
                            ("rotated_-15", rotate(image, -15)),
                            ("flipped", hflip(image)),
                        ]

                        for aug_suffix, aug_image in augmented_versions:
                            aug_filename = f"{filename.split('.')[0]}_{aug_suffix}.{filename.split('.')[-1]}"
                            aug_features = self.extract_features(aug_image)
                            self.add_to_database(aug_features, category, aug_filename, image_path)

    def add_to_database(self, features, category, filename, image_path):
        """
        Adds feature vectors and metadata to the database.

        Args:
            features (np.ndarray): Feature vector.
            category (str): Image category.
            filename (str): Image filename.
            image_path (str): Original image path.
        """
        self.database_features.append(features)
        self.database_metadata.append(
            {
                "filename": filename,
                "category": category,
                "path": image_path,
            }
        )

    def save_database(
        self, metadata_path="features_metadata.pkl", index_path="image_search_index.ann", num_trees=200
    ):
        """
        Saves the features and metadata to a pickle file and builds the Annoy index.

        Args:
            metadata_path (str): Path to save the metadata pickle file.
            index_path (str): Path to save the Annoy index file.
            num_trees (int): Number of trees for the Annoy index.
        """
        with open(metadata_path, "wb") as f:
            pickle.dump((self.database_features, self.database_metadata), f)
        print(f"Features and metadata saved to {metadata_path}")

        print("Building Annoy index...")
        annoy_index = AnnoyIndex(self.feature_dim, "angular")
        for i, features in enumerate(self.database_features):
            annoy_index.add_item(i, features)
        annoy_index.build(num_trees)
        annoy_index.save(index_path)
        print(f"Annoy index saved to {index_path}")


if __name__ == "__main__":
    model_name = "efficientnet_b7"  # Model from torchvision
    base_directory = "DB"

    builder = ModelDatabaseBuilder(model_name=model_name, base_directory=base_directory)

    builder.build_database()

    builder.save_database(
        metadata_path=f"B7/{model_name}_features_metadata.pkl",
        index_path=f"B7/{model_name}_image_search_index.ann",
        num_trees=200,
    )
