import os
import pickle
from tqdm import tqdm
from annoy import AnnoyIndex
import torch
from torchvision import transforms, models
from PIL import Image
import numpy as np
import cv2
import matplotlib.pyplot as plt
import math 
from pprint import pprint


class ImageSearchPredictor:
    def __init__(self, index_path, metadata_path, model_name):
        """
        Initializes the predictor by loading the Annoy index, metadata, and the feature extraction model.

        Args:
            index_path (str): Path to the Annoy index file.
            metadata_path (str): Path to the pickled metadata file.
            model_name (str): Name of the pre-trained model to use for feature extraction.
        """
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.annoy_index = AnnoyIndex(self._get_feature_dim(model_name), "angular")
        self.annoy_index.load(index_path)

        with open(metadata_path, "rb") as f:
            self.database_features, self.database_metadata = pickle.load(f)

        self.model = getattr(models, model_name)(weights="IMAGENET1K_V1")
        self.model.eval()
        self.model.to(self.device)
        self.model = torch.nn.Sequential(*list(self.model.children())[:-1]) 

        self.preprocess = transforms.Compose([
            transforms.Resize((224, 224)), 
            transforms.ToTensor(),         
            transforms.Normalize(           
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])

    def _get_feature_dim(self, model_name):
        """
        Dynamically calculates the feature dimension of the model.

        Args:
            model_name (str): Name of the pre-trained model.

        Returns:
            int: Dimensionality of the feature vector.
        """
        model = getattr(models, model_name)(weights="IMAGENET1K_V1")
        model.eval()
        model = torch.nn.Sequential(*list(model.children())[:-1])
        model.to(self.device)
        dummy_input = torch.zeros(1, 3, 224, 224, device=self.device)
        with torch.no_grad():
            feature_vector = model(dummy_input)
        return feature_vector.view(-1).shape[0]

    def extract_features(self, image_path):
        """
        Extracts feature vectors from an image using the pre-trained model.

        Args:
            image_path (str): Path to the image file.

        Returns:
            np.ndarray: Normalized feature vector.
        """
        image = Image.open(image_path).convert("RGB")
        image_tensor = self.preprocess(image).unsqueeze(0).to(self.device)  # Add batch dimension
        with torch.no_grad():
            features = self.model(image_tensor).squeeze().cpu().numpy()
        # Normalize the features
        features /= np.linalg.norm(features)
        return features

    def query(self, query_features, query_image_path=None, category_filter=None, top_n=5):
        """
        Queries the database for the most similar images based on features, excluding flipped or transparent
        versions unless they have the highest similarity score.

        Args:
            query_features (np.ndarray): Feature vector of the query image.
            query_image_path (str): Path to the query image (optional).
            category_filter (str): Filter results by category (optional).
            top_n (int): Number of top results to return.

        Returns:
            list: List of filtered metadata for the similar images.
        """
        similar_indices = self.annoy_index.get_nns_by_vector(query_features, top_n * 2)
        filtered_results = []
        seen_paths = set()  

        for idx in similar_indices:
            metadata = self.database_metadata[idx]

            if query_image_path and metadata["path"] == query_image_path:
                continue

            if category_filter is None or metadata["category"] == category_filter:
                # Calculate cosine similarity
                similarity = 1 - np.linalg.norm(query_features - self.database_features[idx])
                metadata_with_similarity = metadata.copy()
                metadata_with_similarity["similarity"] = similarity

                base_path = metadata["path"].replace("_flipped", "").replace("_transparent", "")

                if base_path in seen_paths:
                    continue

                filtered_results.append(metadata_with_similarity)
                seen_paths.add(base_path)

                if len(filtered_results) >= top_n:
                    break

        # Sort the final results by similarity score in descending order
        filtered_results = sorted(filtered_results, key=lambda x: x["similarity"], reverse=True)
        return filtered_results

    def debug_categories(self):
        """
        Prints all unique categories in the database and the number of matches for each category.
        """
        category_count = {}
        for metadata in self.database_metadata:
            category = metadata.get("category", "Unknown")
            category_count[category] = category_count.get(category, 0) + 1

        print("\n=== Categories and Match Counts in Database ===")
        for category, count in category_count.items():
            print(f"Category: {category}, Count: {count}")

    def visualize_results(self, query_image_path, results):
        """
        Visualizes the query image and the most similar results in a subplot layout with 3 rows.

        Args:
            query_image_path (str): Path to the query image.
            results (list): List of metadata for the similar images, including paths and similarity scores.
        """
        query_image = cv2.imread(query_image_path)
        query_image = cv2.cvtColor(query_image, cv2.COLOR_BGR2RGB)

        total_images = len(results) + 1  
        rows = 1  
        cols = math.ceil(total_images / rows)  

        plt.figure(figsize=(5 * cols, 5 * rows)) 

        plt.subplot(rows, cols, 1)
        plt.imshow(query_image)
        plt.title("Query Image", fontsize=12)
        plt.axis("off")

        for i, result in enumerate(results):
            result_image = cv2.imread(result["path"])
            result_image = cv2.cvtColor(result_image, cv2.COLOR_BGR2RGB)

            plt.subplot(rows, cols, i + 2)  
            plt.imshow(result_image)
            plt.title(f"{result['category']}\nSim: {result['similarity']:.2f}", fontsize=10)
            plt.axis("off")

        plt.tight_layout()
        plt.show()



if __name__ == "__main__":
    visualize = True
    predictor = ImageSearchPredictor(
        index_path="swin_b/swin_b_image_search_index.ann",
        metadata_path="swin_b/swin_b_features_metadata.pkl",
        model_name="swin_b"  
    )

    predictor.debug_categories()

    query_image_path = "DB/Sinks/download (27).png"  
    query_features = predictor.extract_features(query_image_path)

    results = predictor.query(query_features, category_filter="Sinks", top_n=7)

    category_matches = {}
    for result in results:
        category = result["category"]
        category_matches[category] = category_matches.get(category, 0) + 1

    print("\n=== Query Results ===")
    print(f"Query Image Path: {query_image_path}")
    pprint(results)
    # for category, count in category_matches.items():
    #     print(f"Category: {category}, Matches: {count}")

    # Visualize the query results
    if visualize:
        predictor.visualize_results(query_image_path, results)