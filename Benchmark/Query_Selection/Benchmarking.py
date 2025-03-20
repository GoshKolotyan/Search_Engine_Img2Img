import os
import csv
import sys

os.chdir("../")
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import json
from PIL import Image
from DB_Creation.query import SearchEngine
from DB_Creation.configs import GlobalConfigs


class Benchmark:
    def __init__(self, retrieved: str, model_name: str, search_label: str, search_image_name: str, top_n: int):
        self.model_name = model_name
        self.retrieved = retrieved
        self.search_img_path = f"Dataset Selection/Images_frankwebb/Sink_URLs/frankwebb_{search_image_name}.jpg"
        self.search_label = search_label

        self.search_engine = SearchEngine(label=self.search_label, model_name=self.model_name)
        self.predictions = self.query(self.search_img_path, top_n)

    def precision(self):
        true_positive = len(set(self.predictions) & set(self.retrieved)) if self.retrieved else 0
        predicted_positives = len(self.predictions)
        return true_positive / predicted_positives if predicted_positives > 0 else 0.0

    def query(self, query_image_path: str, top_n: int):
        img = Image.open(query_image_path).convert("RGB")
        query_features = (
            self.search_engine.extract_features_swin(img)
            if "swin" in self.model_name
            else self.search_engine.extract_features_dino(img)
        )
        results = self.search_engine.query(query_features, top_n=top_n)
        return [
            results[i]["product_path"].split("/")[-1].split(".")[0].split("frankwebb_")[-1]
            for i in range(len(results))
        ]

    def record_results_to_csv(self, precision, key,  csv_dir='results'):
        if not os.path.exists(csv_dir):
            os.makedirs(csv_dir)

        csv_file = os.path.join(csv_dir, f'{self.model_name}_results.csv')
        file_exists = os.path.isfile(csv_file)

        with open(csv_file, mode='a', newline='', encoding='utf-8') as file:
            writer = csv.writer(file)
            if not file_exists:
                writer.writerow(['Filename', 'Precision Score'])

            writer.writerow([key, precision])

    # def __call__(self):
        # precision_score = self.precision()
        # print("Precision -->", precision_score)
        # print("\n")
        # print(f"Number of True Positives {len(set(self.predictions) & set(self.retrieved))}")
        # print(f"Length of retrieved dataset: {len(self.retrieved)}")
        # print(f"Prediction length: {len(self.predictions)}")
        # print(2 * "\n")
        # print(f"Correct files: {set(self.predictions) & set(self.retrieved)}")
        # print(f"All predicted files: {set(self.predictions)}")



if __name__ == "__main__":
    benchmark_path = "Query_Selection/relevant_sinks.json"
    with open(benchmark_path, "r") as file:
        retrieved = json.load(file)

    precision_dict = {model: [] for model in GlobalConfigs.MODEL_NAME}

    for key in retrieved:
        search_image_name = list(retrieved[key].keys())[0]
        for model_name in GlobalConfigs.MODEL_NAME:
            benchmark = Benchmark(
                retrieved=list(retrieved[key].values())[0],
                search_label="Sink_URLs",  # Fixed parameter name
                model_name=model_name,
                search_image_name=search_image_name,
                top_n=10,
            )
            precision = benchmark.precision()
            precision_dict[model_name].append(precision)
            benchmark.record_results_to_csv(key=key, precision=precision)

    # Compute per-model averages
    for model_name, precisions in precision_dict.items():
        overall_precision = sum(precisions) / len(precisions) if precisions else 0.0
        print(f"Model {model_name} overall precision: {overall_precision}")