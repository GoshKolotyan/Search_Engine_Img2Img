import logging
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import List


class Analyzer:
    def __init__(self, paths: List[str], output_dir: str):
        self.paths = paths
        self.output_dir = output_dir

    def read_csv(self, path: str) -> pd.DataFrame:
        try:
            logging.info(f"Loading csv from {path}")
            return pd.read_csv(path)
        except Exception as e:
            logging.error(f"Error reading {path}: {e}")
            return None

    def _plot(self) -> None:
        dfs = {path: self.read_csv(path) for path in self.paths}
        dfs = {path: df for path, df in dfs.items() if df is not None}
        if not dfs:
            logging.error("No valid CSV files to analyze.")
            return

        num_csvs = len(dfs)
        bar_width = 0.8 / num_csvs
        index = np.arange(len(next(iter(dfs.values()))["Model Name"]))
        figsize = (60, 25)
        plt.figure(figsize=figsize, dpi=200)

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
                    fontsize=5,
                )

        plt.xlabel("Model Name")
        plt.ylabel("Similarity Score")
        plt.title("Model Similarity Scores from Multiple CSV Files")
        plt.xticks(
            index + (num_csvs * bar_width) / 2, model_names, rotation=15, ha="center"
        )
        # Horizontal threshold line
        plt.axhline(y=0.7, linewidth=1, color="k")
        plt.grid(visible=True)
        plt.legend()
        plt.savefig(f"{self.output_dir}/Results/barplots.jpg")
        plt.close()

    def _avg_plot(self) -> None:
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
        plt.axhline(y=0.7, linewidth=1, color="k")
        plt.legend(["Avg Similarity Score"])
        plt.savefig(f"{self.output_dir}/Results/avg_scores.jpg")
        plt.close()
        print(avg_scores)
        avg_scores.to_csv(f'{self.output_dir}/Results/avg_scores.csv')
        return avg_scores

    def _min_3_dir_based_model(self) -> pd.DataFrame:

        dfs = {path: self.read_csv(path) for path in self.paths}
        dfs = {path: df for path, df in dfs.items() if df is not None}

        if not dfs:
            logging.error("No valid CSV files to analyze for smallest models.")
            return pd.DataFrame()

        columns = ["Folder_Name"]
        num_models = 19  # Number of all models

        for i in range(1, num_models + 1):
            columns.append(f"Model_name_{i}")
            columns.append(f"Model_name_{i}_score")

        rows = []
        for dir_path, df in dfs.items():
            if "Similarity Score" not in df.columns or "Model Name" not in df.columns:
                logging.error(f"Skipping {dir_path}: Missing required columns.")
                continue

            folder_name = dir_path.split("/")[-2]
            row = [folder_name]

            smallest_df = df.nsmallest(num_models, "Similarity Score")

            for _, row_data in smallest_df.iterrows():
                row.append(row_data["Model Name"])
                row.append(row_data["Similarity Score"])

            while len(row) < len(columns):
                row.append(None)

            rows.append(row)

        if not rows:
            return pd.DataFrame()

        final_df = pd.DataFrame(rows, columns=columns)
        final_df.to_csv(f"{self.output_dir}/Results/smales_values_based_on_dir.csv")
        return final_df

    def get_smallest_dir_per_model(self) -> pd.DataFrame:

        final_df = self._min_3_dir_based_model()
        if final_df.empty:
            logging.error("No data found to determine smallest directories per model.")
            return pd.DataFrame(
                columns=["Model_Name", "Smallest_Folder", "Smallest_Score"]
            )

        model_smallest_dir = {}

        for _, row in final_df.iterrows():
            folder_name = row["Folder_Name"]

            total_pairs = (len(row) - 1) // 2

            for i in range(1, total_pairs + 1):
                model_name = row[f"Model_name_{i}"]
                model_score = row[f"Model_name_{i}_score"]

                if model_name is None:
                    continue

                if model_name not in model_smallest_dir:
                    model_smallest_dir[model_name] = (folder_name, model_score)
                else:

                    _, current_score = model_smallest_dir[model_name]
                    if (model_score is not None) and (model_score < current_score):
                        model_smallest_dir[model_name] = (folder_name, model_score)

        model_smallest_dir_df = (
            pd.DataFrame.from_dict(
                model_smallest_dir,
                orient="index",
                columns=["Smallest_Folder", "Smallest_Score"],
            )
            .reset_index()
            .rename(columns={"index": "Model_Name"})
            .to_csv(f"{self.output_dir}/Results/smales_values_based_on_model.csv")
        )

        print(model_smallest_dir_df)
        return model_smallest_dir_df

    def __call__(self):
        """
        This allows the Analyzer class to be used as a callable object.
        When invoked, it runs:
            - _plot() -> create bar plots per CSV
            - _avg_plot() -> create average bar plot across all CSVs
            - _min_3_dir_based_model() -> internal usage only (returns DF)
            - get_smallest_dir_per_model() -> prints and returns DF of smallest directories per model
        """
        self._plot()
        self._avg_plot()
        self._min_3_dir_based_model()
        self.get_smallest_dir_per_model()
