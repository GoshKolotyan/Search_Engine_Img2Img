import json
import numpy as np
from pprint import pprint


class Benchmark:
    def __init__(self, path, predictions):
        self.predictions = predictions
        with open(path, "r") as file:
            self.retrieved = json.load(file)

        self.retrieved = self.retrieved.get("Black Sink", {}).get("image_17", []) # Available keys are for sink now < Deep White sink,  footed White sink, Black Sink>

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

    predictions_for_deep_white_sink = [
                    'Sink_1', 'Sink_210', 'Sink_178', 'Sink_235', 'Sink_534', 'Sink_819', 'Sink_784', 'Sink_256', 
                    'Sink_349', 'Sink_514', 'Sink_187', 'Sink_495', 'Sink_462', 'Sink_833', 'Sink_513', 'Sink_141', 
                    'Sink_267', 'Sink_694', 'Sink_429', 'Sink_872', 'Sink_822', 'Sink_945', 'Sink_287', 'Sink_334', 
                    'Sink_56', 'Sink_860', 'Sink_507', 'Sink_856', 'Sink_940', 'Sink_779', 'Sink_656', 'Sink_530', 
                    'Sink_546', 'Sink_145', 'Sink_331', 'Sink_635', 'Sink_139', 'Sink_807', 'Sink_848', 'Sink_294', 
                    'Sink_979', 'Sink_500', 'Sink_649', 'Sink_566', 'Sink_1041', 'Sink_804', 'Sink_579', 'Sink_397', 
                    'Sink_305', 'Sink_768'
                    ]
    predictions_for_footed_white_sink = [
                    'Sink_3', 'Sink_419', 'Sink_899', 'Sink_781', 'Sink_383', 'Sink_911', 'Sink_926', 'Sink_763', 
                    'Sink_152', 'Sink_778', 'Sink_733', 'Sink_748', 'Sink_288', 'Sink_307', 'Sink_401', 'Sink_686', 
                    'Sink_310', 'Sink_76', 'Sink_606', 'Sink_177', 'Sink_345', 'Sink_442', 'Sink_269', 'Sink_974', 
                    'Sink_57', 'Sink_965', 'Sink_955', 'Sink_404', 'Sink_571', 'Sink_422', 'Sink_158', 'Sink_347', 
                    'Sink_838', 'Sink_171', 'Sink_638', 'Sink_702', 'Sink_215', 'Sink_460', 'Sink_991', 'Sink_272', 
                    'Sink_718', 'Sink_587', 'Sink_196', 'Sink_456', 'Sink_234', 'Sink_474', 'Sink_491', 'Sink_385', 
                    'Sink_828', 'Sink_555'
                    ]
    predictions_for_black_sink = [
                    'image_17', 'image_18', 'Sink_381', 'Sink_524', 'Sink_455', 'Sink_941', 'Sink_774', 'Sink_759', 
                    'Sink_257', 'Sink_276', 'Sink_557', 'Sink_589', 'Sink_238', 'Sink_506', 'Sink_541', 'Sink_572', 
                    'Sink_181', 'Sink_382', 'Sink_472', 'Sink_436', 'Sink_162', 'Sink_454', 'Sink_489', 'Sink_399', 
                    'Sink_200', 'Sink_659', 'Sink_643', 'Sink_729', 'Sink_744', 'Sink_671', 'Sink_295', 'Sink_639', 
                    'Sink_655', 'Sink_333', 'Sink_314', 'Sink_97', 'Sink_135', 'Sink_628', 'Sink_78', 'Sink_564', 
                    'Sink_827', 'Sink_580', 'Sink_813', 'Sink_837', 'Sink_808', 'Sink_823', 'Sink_624', 'Sink_437', 
                    'Sink_400', 'Sink_408'
                    ]

    cls = Benchmark(
        path="relevant.json",
        predictions=predictions_for_black_sink)
    cls()
