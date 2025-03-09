
import logging 
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt 

class Analyzer:
    def __init__(self, paths, output_dir):
        self.paths = paths
        self.output_dir = output_dir

    def read_csv(self, path):
        """Reads a CSV file and returns a DataFrame."""
        try:
            logging.info(f"Loading csv from {path}")
            return pd.read_csv(path)
        except Exception as e:
            logging.error(f"Error reading {path}: {e}")
            return None

    def _plot(self):
        dfs = {path: self.read_csv(path) for path in self.paths}
        dfs = {path: df for path, df in dfs.items() if df is not None}

        if not dfs:
            logging.error("No valid CSV files to analyze.")
            return

        plt.figure(figsize=(120, 60), dpi=200)

        num_csvs = len(dfs)
        bar_width = 0.1
        index = np.arange(len(next(iter(dfs.values()))["Model Name"]))

        for i, (file_path, df) in enumerate(dfs.items()):
            if "Model Name" not in df.columns or "Similarity Score" not in df.columns:
                logging.error(f"Skipping {file_path}: Required columns missing.")
                continue

            df = df.sort_values("Model Name")

            model_names = df["Model Name"].values
            scores = df["Similarity Score"].values

            x_positions = index + (i * bar_width)

            bars = plt.bar(
                x_positions,
                scores,
                width=bar_width,
                label=f"File {i+1}: {file_path.split('/')[-1]}",
                alpha=0.7,
            )

            for bar in bars:
                height = bar.get_height()
                plt.text(
                    bar.get_x() + bar.get_width() / 2,
                    height + 0.01,
                    f"{height:.2f}",
                    ha="center",
                    fontsize=10,
                )

        plt.xlabel("Model Name")
        plt.ylabel("Similarity Score")
        plt.title("Model Similarity Scores from Multiple CSV Files")
        plt.xticks(
            index + (num_csvs * bar_width) / 2, model_names, rotation=15, ha="center"
        )
        plt.grid(visible=True)
        plt.legend()
        plt.tight_layout(pad=4)
        plt.savefig(f"../{self.output_dir}/barplots.jpg")
        plt.close()
        # plt.show()

    def _avg_plot(self):

        valid_dfs = []
        for path in self.paths:
            df = self.read_csv(path)
            if df is None:
                logging.error(f"Skipping {path}: Could not read file.")
                continue
            if "Model Name" not in df.columns or "Similarity Score" not in df.columns:
                logging.error(f"Skipping {path}: Required columns missing.")
                continue
            valid_dfs.append(df)

        if not valid_dfs:
            logging.error("No valid CSV files to analyze.")
            return

        all_data = pd.concat(valid_dfs, ignore_index=True)

        avg_scores = (
            all_data.groupby("Model Name")["Similarity Score"]
            .mean()
            .reset_index()
            .sort_values("Model Name")
        )
        plt.figure(figsize=(30, 20), dpi=200)
        x_positions = np.arange(len(avg_scores))
        scores = avg_scores["Similarity Score"].values
        model_names = avg_scores["Model Name"].values

        bars = plt.bar(x_positions, scores, alpha=0.7)

        for bar in bars:
            height = bar.get_height()
            plt.text(
                bar.get_x() + bar.get_width() / 2,
                height + 0.01,
                f"{height:.2f}",
                ha="center",
                fontsize=10,
            )

        plt.xlabel("Model Name")
        plt.ylabel("Average Similarity Score")
        plt.title("Average Model Similarity Scores Across CSV Files")
        plt.xticks(x_positions, model_names, rotation=15, ha="center")
        plt.grid(True)
        plt.legend(["Avg Similarity Score"])
        plt.savefig(f"../{self.output_dir}/avg_scores.jpg")
        plt.close()
        # plt.show()

    def __call__(self):
        self._plot()
        self._avg_plot()
