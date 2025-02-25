import json
import numpy as np
from pprint import pprint


class Benchmark:
    def __init__(self, path, predictions):
        self.predictions = predictions
        with open(path, "r") as file:
            self.retrieved = json.load(file)

        self.retrieved = self.retrieved.get("Deep White sink", {}).get("Sink_1", [])

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
        print(true_positive)

        return (
            true_positive / (true_positive + false_negative)
            if (true_positive + false_negative) > 0
            else 0.0
        )

    def __call__(self):
        print("Recall -->", self.recall())
        print("Precision -->", self.precision())
        print(f"Number of True Precision {len(set(self.predictions) & set(self.retrieved))} from {len(self.retrieved)}")
        # print("All True Postives", set(self.predictions) & set(self.retrieved))
        print("Lenght of dataset selections",len(self.retrieved))


if __name__ == "__main__":
    cls = Benchmark(
        path="relevant.json",
        predictions=['Sink_1', 'Sink_210', 'Sink_178', 'Sink_235', 'Sink_534', 'Sink_819', 'Sink_784', 'Sink_256', 
                     'Sink_349', 'Sink_514', 'Sink_187', 'Sink_495', 'Sink_462', 'Sink_833', 'Sink_513', 'Sink_141', 
                     'Sink_267', 'Sink_694', 'Sink_429', 'Sink_872', 'Sink_822', 'Sink_945', 'Sink_287', 'Sink_334', 
                     'Sink_56', 'Sink_860', 'Sink_507', 'Sink_856', 'Sink_940', 'Sink_779', 'Sink_656', 'Sink_530', 
                     'Sink_546', 'Sink_145', 'Sink_331', 'Sink_635', 'Sink_139', 'Sink_807', 'Sink_848', 'Sink_294', 
                     'Sink_979', 'Sink_500', 'Sink_649', 'Sink_566', 'Sink_1041', 'Sink_804'])
    cls()
