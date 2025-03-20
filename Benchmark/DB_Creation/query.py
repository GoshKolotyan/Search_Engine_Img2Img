import cv2
import math
import torch
import pickle
import numpy as np
import matplotlib.pyplot as plt

import os 
from annoy import AnnoyIndex
from torchvision import transforms, models
# from .configs import INDEX_PATH_FOR_SAVING, METADATA_PATH_FOR_SAVING, EMBEDDING_MODEL


class SearchEngine:
    def __init__(self, label: str):
        self.label = label
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.index_path = "/home/gosh/Desktop/Search_Engine_Img2Img/Benchmark/DB_Creation/swin_s/image_search_index.ann"
        self.metadata_path = "/home/gosh/Desktop/Search_Engine_Img2Img/Benchmark/DB_Creation/swin_s/features_metadata.pkl"
        self.model_name = "swin_s"

        self.model = getattr(models, self.model_name)(weights="IMAGENET1K_V1")
        self.model.eval()
        self.model = torch.nn.Sequential(*list(self.model.children())[:-1])
        self.model.to(self.device)

        self.feature_dim = self._get_feature_dim()

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

    def _get_feature_dim(self):
        dummy_input = torch.zeros(1, 3, 224, 224, device=self.device)
        with torch.no_grad():
            feature_vector = self.model(dummy_input)
        return feature_vector.view(-1).shape[0]

    def extract_features(self, image: np.ndarray) -> np.ndarray:
        image_tensor = self.preprocess(image).unsqueeze(0).to(self.device)
        with torch.no_grad():
            features = self.model(image_tensor).squeeze().cpu().numpy()
        features /= np.linalg.norm(features)
        return features

    def query(self, query_features: np.ndarray, top_n: int = 5):
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
    def search(self, query_image: np.ndarray, top_n: int = 5, visualize: bool = False):
        query_features = self.extract_features(query_image)

        results = self.query(query_features, top_n=top_n)

        if visualize:
            self.visualize_results(query_image, results)

        return results


# if __name__ == "__main__":
#     query_image_path = "../Dataset Selection/Images_frankwebb/Sink_URLs/frankwebb_Sink_1.jpg"
#     query_image = Image.open(query_image_path).convert("RGB")
#     print("Image is loaded for query")

#     predictor = SearchEngine(label="Sink_URLs")
#     print('Running for predictor')
#     query_features = predictor.extract_features(query_image)

#     results = predictor.query(query_features, top_n=12)
#     print("\n=== Query Results ===")
#     pprint(results)

#     predictor.visualize_results(query_image_path, results)