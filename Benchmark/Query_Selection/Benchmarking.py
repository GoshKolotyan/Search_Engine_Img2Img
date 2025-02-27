import os
import sys

os.chdir("../")
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import json
import numpy as np
from PIL import Image
from pprint import pprint
from DB_Creation.query import SearchEngine


class Benchmark:
    def __init__(
        self,
        benchmark_path: str,
        search_img_path: str,
        search_abel_label: str,
        top_n: int,
    ):
        self.search_abel_label = search_abel_label
        self.search_engine = SearchEngine(label=self.search_abel_label)
        self.predictions = self.query(search_img_path, top_n)
        with open(benchmark_path, "r") as file:
            self.retrieved = json.load(file)

        self.retrieved = self.retrieved.get("Deep White sink", {}).get(
            "Sink_1", []
        )  # Available keys are for sink now < Deep White sink, footed White sink, Black Sink> TODO add automatation of this proces

    def precision(self):
        """
        Computes Precision:
        Precision = True Positives / (True Positives + False Positives)
        """
        true_positive = (
            len(set(self.predictions) & set(self.retrieved)) if self.retrieved else 0
        )
        predicted_positives = len(self.predictions)

        return true_positive / predicted_positives if predicted_positives > 0 else 0.0

    def recall(self):
        """
        Computes Recall:
        Recall = True Positives / (True Positives + False Negatives)
        """

        true_positive = len(set(self.predictions) & set(self.retrieved))
        false_negative = len(set(self.retrieved) - set(self.predictions))

        return (
            true_positive / (true_positive + false_negative)
            if (true_positive + false_negative) > 0
            else 0.0
        )

    def query(self, query_image_path: str, top_n: int):
        img = Image.open(query_image_path).convert("RGB")
        query_features = self.search_engine.extract_features(img)
        results = self.search_engine.query(query_features, top_n=top_n)
        return [
            results[i]["product_path"]
            .split("/")[-1]
            .split(".")[0]
            .split("frankwebb_")[-1]
            for i in range(len(results))
        ]

    def __call__(self):
        print("Recall -->", self.recall())
        print("Precision -->", self.precision())
        print(2 * "\n")
        print(
            f"Number of True Positives {len(set(self.predictions) & set(self.retrieved))}"
        )
        print(f"Number of lenght of dataset of our {len(self.retrieved)}")
        print(f"Number of predicition lenght {len(self.predictions)}")
        print(2 * "\n")
        print(
            f"Files that are corrected {(set(self.predictions) & set(self.retrieved))} "
        )
        print(f"File that containes predicition {(self.predictions)}")


if __name__ == "__main__":
    query_image_path = (
        "Dataset Selection/Images_frankwebb/Sink_URLs/frankwebb_Sink_1.jpg"
    )

    cls = Benchmark(
        benchmark_path="Query_Selection/relevant_sinks.json",
        search_img_path=query_image_path,
        search_abel_label="Sink_URLs",
    )
    cls()
