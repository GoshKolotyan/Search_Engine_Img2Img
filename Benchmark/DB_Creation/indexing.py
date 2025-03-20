import os
import pickle
import torch
import numpy as np
import torch.nn.functional as F

import logging
from PIL import Image
from tqdm import tqdm
from annoy import AnnoyIndex
from torchvision import transforms, models
from torchvision.transforms.functional import hflip

from configs import (
    NUMBER_OF_TREES_FOR_ANNOY_INDEX,

)


class ModelDatabaseBuilder:
    def __init__(
        self,
        base_dir: str,
        model_name: str,
        num_trees: str = NUMBER_OF_TREES_FOR_ANNOY_INDEX,
    ):
        self.model_name = model_name
        os.mkdir(self.model_name)

        self.metadata_path = self.model_name + "/features_metadata.pkl"
        self.index_path = self.model_name + "/image_search_index.ann"
        # self.num_tree = num_trees
        # self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        # self.base_dir = base_dir
        # if "swin" in self.model_name:
        #     self.model = getattr(models, self.model_name)(weights="IMAGENET1K_V1")
        #     self.model = torch.nn.Sequential(*list(self.model.children())[:-1]).to(device=self.device)
        #     self.model = self.model.to(self.device).eval()

        #     self.preprocess = transforms.Compose(
        #         [
        #             transforms.Resize((224, 224)),
        #             transforms.ToTensor(),
        #             transforms.Normalize(
        #                 mean=[0.485, 0.456, 0.406],
        #                 std=[0.229, 0.224, 0.225],
        #             ),
        #         ]
        #     )
        #     dummy_input = torch.zeros(1, 3, 224, 224)
        #     with torch.no_grad():
        #         feature_vector = self.model(dummy_input)
        #     self.feature_dim = feature_vector.view(-1).shape[0]

        # elif "dino":
        #     self.model = torch.hub.load("facebookresearch/dinov2", self.model_name)
        #     self.model = self.model.to(self.device).eval()
        #     self.preprocess = transforms.Compose(
        #         [
        #             transforms.Resize((518, 518)),  # Official DINOv2 input size
        #             transforms.ToTensor(),
        #             transforms.Normalize(
        #                 mean=[0.485, 0.456, 0.406], 
        #                 std=[0.229, 0.224, 0.225]
        #             ),
        #         ]
        #     )
        #     dummy_input = torch.zeros(1, 3, 518, 518)
        #     with torch.no_grad():
        #         feature_vector = self.model(dummy_input)
        #     self.feature_dim = feature_vector.view(-1).shape[0]



        self.database_features = []
        self.database_metadata = []

        logging.info("Initilazting of DatabaseBuilder")

    def extract_features(self, image: Image.Image) -> torch.Tensor:
        if image is None:
            return None
        
        image_tensor = self.processor(image).unsqueeze(0).to(self.device)
        with torch.no_grad():
            features = self.model(image_tensor)
        # L2 normalize
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
    
    # def build_database_and_save(
    #     self,
    #     save=True,
    # ):
    #     """
    #     Scans 'self.base_dir' for category subfolders, loads images, extracts features (+ flip augmentation),
    #     stores them, and then optionally saves the metadata + builds the Annoy index.
    #     """
    #     for category in os.listdir(self.base_dir):
    #         category_path = os.path.join(self.base_dir, category)
    #         if os.path.isdir(category_path):
    #             for filename in tqdm(
    #                 os.listdir(category_path), desc=f"Processing {category}"
    #             ):
    #                 if filename.lower().endswith((".png", ".jpg", ".jpeg")):
    #                     image_path = os.path.join(category_path, filename)

    #                     image = Image.open(image_path).convert("RGB")
    #                     print(filename)
    #                     # Attempt to parse product_id from the filename (before first underscore)
    #                     # product_id = int(filename.split("_")[0])

    #                     # Extract features for original image
    #                     original_features = self.extract_features(image)
    #                     self.add_to_database(
    #                         original_features,
    #                         category,
    #                         filename,
    #                         image_path,
    #                         # product_id,
    #                     )

    #                     # Example data augmentation: horizontal flip
    #                     augmented_versions = [("flipped", hflip(image))]
    #                     for aug_suffix, aug_image in augmented_versions:
    #                         aug_filename = f"{os.path.splitext(filename)[0]}_{aug_suffix}{os.path.splitext(filename)[1]}"
    #                         aug_features = self.extract_features(aug_image)
    #                         # For flips, we can keep the same product_id
    #                         self.add_to_database(
    #                             aug_features,
    #                             category,
    #                             aug_filename,
    #                             image_path,
    #                             # product_id,
    #                         )

    #     if save:
    #         # Save features + metadata
    #         with open(self.metadata_path, "wb") as f:
    #             pickle.dump((self.database_features, self.database_metadata), f)

    #         logging.info(f"Metadata save to {self.metadata_path}")

    #         # Build Annoy index
    #         logging.info("Building Annoy index")
    #         annoy_index = AnnoyIndex(self.feature_dim, "angular")

    #         for i, features in enumerate(self.database_features):
    #             annoy_index.add_item(i, features)

    #         annoy_index.build(self.num_tree)
    #         annoy_index.save(self.index_path)

    #         logging.info(f"Annoy index saved to {self.index_path}")

    def add_to_database(
        self,
        features: np.ndarray,
        category: str,
        filename: str,
        image_path: str,
        # product_id: int,
    ):
        """
        Stores the feature vector and metadata in local lists for later saving.
        """
        self.database_features.append(features)
        self.database_metadata.append(
            {
                "filename": filename,
                "category": category,
                "image_path": image_path,
                # "product_id": product_id,
            }
        )
