import os
import pickle
from tqdm import tqdm
import torch
import timm
from torchvision import transforms
from PIL import Image
import numpy as np
from annoy import AnnoyIndex


class ModelDatabaseBuilder:
    def __init__(
        self,
        model_name="efficientnet_b4",
        base_directory="Image_Base",
        feature_dim=1792,
    ):
        """
        Initializes the database builder by setting up the model, transformations, and directories.

        Args:
            model_name (str): Name of the pre-trained model to use for feature extraction.
            base_directory (str): Path to the dataset directory.
            feature_dim (int): Dimensionality of the feature vectors.
        """
        self.model_name = model_name
        self.base_directory = base_directory
        self.feature_dim = feature_dim

        self.model = timm.create_model(model_name, pretrained=True)
        self.model.eval()
        self.model = torch.nn.Sequential(*list(self.model.children())[:-1])

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

    def extract_features(self, image_path):
        """
        Extracts feature vectors from an image using the pre-trained model.

        Args:
            image_path (str): Path to the image file.

        Returns:
            np.ndarray: Normalized feature vector.
        """
        image = Image.open(image_path).convert("RGB")
        image_tensor = self.preprocess(image).unsqueeze(0)
        with torch.no_grad():
            features = self.model(image_tensor).squeeze().numpy()
        features /= np.linalg.norm(features)
        return features

    def build_database(self):
        """
        Iterates over the dataset, extracts features, and stores metadata.
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

                        features = self.extract_features(image_path)

                        self.database_features.append(features)
                        self.database_metadata.append(
                            {
                                "filename": filename,
                                "category": category,
                                "path": image_path,
                            }
                        )

    def save_database(
        self,
        metadata_path="features_metadata.pkl",
        index_path="image_search_index.ann",
        num_trees=200,
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
    model_name = "efficientnet_b4"
    base_directory = "Image_Base"

    builder = ModelDatabaseBuilder(
        model_name=model_name, base_directory=base_directory, feature_dim=1792
    )

    builder.build_database()

    builder.save_database(
        metadata_path=f"{model_name}_features_metadata.pkl",
        index_path=f"{model_name}_image_search_index.ann",
        num_trees=200,
    )
