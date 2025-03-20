import cv2
import math
import torch
import pickle
import numpy as np
import matplotlib.pyplot as plt
import torch.nn.functional as F

import os 
from annoy import AnnoyIndex
from torchvision import transforms, models

class SearchEngine:
    def __init__(self, label: str, model_name: str):
        self.label = label
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model_name = model_name
        self.index_path = f"/home/gosh/Desktop/Search_Engine_Img2Img/Benchmark/DB_Creation/{self.model_name}/image_search_index.ann"
        self.metadata_path = f"/home/gosh/Desktop/Search_Engine_Img2Img/Benchmark/DB_Creation/{self.model_name}/features_metadata.pkl"
        
        if "swin" in self.model_name:
            self.model = getattr(models, self.model_name)(weights="IMAGENET1K_V1")
            self.model = torch.nn.Sequential(*list(self.model.children())[:-1])
            self.model = self.model.to(self.device).eval()

            self.feature_dim = self._get_feature_dim_swin()

            self.annoy_index = AnnoyIndex(self.feature_dim, "angular")
            self.annoy_index.load(self.index_path)

            with open(self.metadata_path, "rb") as f:
                self.database_features, self.database_metadata = pickle.load(f)

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
        elif "dino" in self.model_name:
            self.model = torch.hub.load("facebookresearch/dinov2", self.model_name).to(self.device).eval()
            self.feature_dim = self._get_features_dim_dino()

            self.annoy_index = AnnoyIndex(self.feature_dim, "angular")
            self.annoy_index.load(self.index_path)
            with open(self.metadata_path, "rb") as f:
                self.database_features, self.database_metadata = pickle.load(f)
            self.preprocess = transforms.Compose(
                [
                    transforms.Resize((518, 518)),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                ]
            )
            dummy_input = torch.zeros(1, 3, 518, 518).to(self.device)
            with torch.no_grad():
                features = self.model.backbone.forward_features(dummy_input)
                embeddings = features["x_norm_clstoken"]
            self.feature_dim = embeddings.shape[-1]


    def _get_feature_dim_swin(self):
        dummy_input = torch.zeros(1, 3, 224, 224, device=self.device)
        with torch.no_grad():
            feature_vector = self.model(dummy_input)
        return feature_vector.view(-1).shape[0]
    
    def _get_features_dim_dino(self):
        dummy_input = torch.zeros(1, 3, 518, 518).to(self.device)
        with torch.no_grad():
            feature_vector = self.model.backbone.forward_features(dummy_input)
            embeddings = feature_vector["x_norm_clstoken"]
        return embeddings.shape[-1]


    def extract_features_swin(self, image: np.ndarray) -> np.ndarray:
        if image is None:
            return None
        image_tensor = self.preprocess(image).unsqueeze(0).to(self.device)
        with torch.no_grad():
            features = self.model(image_tensor)
        features = features.view(features.size(0), -1)
        features = F.normalize(features, p=2, dim=1)
        return features.cpu().numpy().flatten()
    
    def extract_features_dino(self, image: np.ndarray) ->np.ndarray:
        if image is None:
            return None
        input_tensor = self.preprocess(image).unsqueeze(0).to(self.device)
        with torch.no_grad(), torch.autocast(device_type="cuda"):
            features = self.model.backbone.forward_features(input_tensor)
            embeddings = features["x_norm_clstoken"]
        embeddings = F.normalize(embeddings, p=2, dim=-1)
        return embeddings.cpu().numpy().flatten()


    def query(self, query_features: np.ndarray, top_n: int):
        similar_indices = self.annoy_index.get_nns_by_vector(query_features, top_n * 2)
        filtered_results = []
        seen_paths = set()

        for idx in similar_indices:
            metadata = self.database_metadata[idx]

            if self.label is None or metadata["category"] == self.label:
                # Calculate cosine similarity
                similarity = float(
                    1 - np.linalg.norm(query_features - self.database_features[idx])
                )
                metadata_with_similarity = {
                    "category": metadata["category"],
                    "similarity": similarity,
                    "product_path": metadata['image_path'][3:]
                }

                base_path = (
                    metadata["image_path"].replace("_flipped", "")
                )

                if base_path in seen_paths:
                    continue

                filtered_results.append(metadata_with_similarity)
                seen_paths.add(base_path)

                if len(filtered_results) >= top_n:
                    break

        filtered_results = sorted(
            filtered_results, key=lambda x: x["similarity"], reverse=True
        )
        return filtered_results


    def visualize_results(self, query_image_path: str, results: list):
        query_image = cv2.imread(query_image_path)
        query_image = cv2.cvtColor(query_image, cv2.COLOR_BGR2RGB)

        total_images = len(results) + 1
        rows = math.ceil(total_images / 3)
        cols = min(5, total_images)

        plt.figure(figsize=(5 * cols, 5 * rows))

        # Plot the query image
        plt.subplot(rows, cols, 1)
        plt.imshow(query_image)
        plt.title("Query Image", fontsize=14, fontweight='bold', color='blue')
        plt.axis("off")

        # Plot the result images
        for i, result in enumerate(results):
            result_image = cv2.imread(result["product_path"])
            result_image = cv2.cvtColor(result_image, cv2.COLOR_BGR2RGB)

            plt.subplot(rows, cols, i + 2)
            plt.imshow(result_image)
            
            # Extract the filename from the product path
            product_name = os.path.basename(result["product_path"])
            
            # Display the similarity score and product name as an annotation
            plt.text(0.5, -0.15, f"Similarity: {result['similarity']:.2f}\nProduct: {product_name}", 
                    fontsize=10, ha='center', va='center', transform=plt.gca().transAxes, color='green')
            
            plt.axis("off")

        plt.tight_layout(pad=3.0)
        plt.show()
    def search(self, query_image: np.ndarray, top_n: int , visualize: bool = False):
        if "swin" in self.model_name:
            query_features = self.extract_features_swin(query_image) 
        elif "dino" in self.model_name:
            query_features = self.extract_features_dino(query_image)

        results = self.query(query_features, top_n=top_n)

        # if visualize:
        #     self.visualize_results(query_image) 
            

        return results