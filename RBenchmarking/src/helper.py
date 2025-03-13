import pandas as pd
import matplotlib.pyplot as plt


class DifferenceAnalyzer:
    def __init__(self, path_relevant: str, path_not_relevant: str):
        self.path_relevant = path_relevant
        self.path_not_relevant = path_not_relevant
        self.df_before = pd.DataFrame()
        self.df_after = pd.DataFrame()
        self.diff_df = pd.DataFrame()

    def load_json(self, path: str) -> pd.DataFrame:
        """Load JSON file into a DataFrame."""
        try:
            return pd.read_json(path)
        except ValueError as e:
            raise ValueError(f"Error reading JSON from {path}: {e}")

    def compute_difference(self, threshold_percent: float = 10.0) -> pd.DataFrame:
        self.df_before = self.load_json(self.path_relevant)
        self.df_after = self.load_json(self.path_not_relevant)

        required_cols = ['Model Name', 'Similarity Score']
        if not all(col in self.df_before for col in required_cols):
            raise KeyError("Relevant dataframe missing required columns.")
        if not all(col in self.df_after for col in required_cols):
            raise KeyError("Not relevant dataframe missing required columns.")

        diff_df = pd.DataFrame({
            'Model Name': self.df_before['Model Name'],
            'Relevant Score (%)': self.df_before['Similarity Score'] * 100,
            'Not Relevant Score (%)': self.df_after['Similarity Score'] * 100,
        })
        diff_df['Difference Score (%)'] = (
            diff_df['Relevant Score (%)'] - diff_df['Not Relevant Score (%)']
        )

        diff_df[f'{threshold_percent}% of Relevant Score'] = (
            threshold_percent / 100 * diff_df['Relevant Score (%)']
        )

        diff_df['Small Difference'] = (
            diff_df['Difference Score (%)'] > diff_df[f'{threshold_percent}% of Relevant Score']
        )

        self.diff_df = diff_df
        return self.diff_df

    def visualize_difference(self) -> None:
        """Visualize differences."""
        if self.diff_df.empty:
            raise ValueError("Difference DataFrame is empty. Run compute_difference() first.")

        plt.figure(figsize=(10, 6))
        plt.bar(self.diff_df['Model Name'][::-1], self.diff_df['Difference Score (%)'][::-1])
        plt.xlabel('Model Name')
        plt.ylabel('Difference Score (%)')
        plt.xticks(rotation=20)
        plt.title('Differences in Similarity Scores')
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        plt.tight_layout()
        plt.show()

    def __call__(self, visualize: bool = False, threshold_percent: float = 10.0) -> pd.DataFrame:
        diff_df = self.compute_difference(threshold_percent)

        if visualize:
            self.visualize_difference()

        return diff_df


if __name__ == "__main__":
    path_1 = "../OUTPUTS/TOILET_PLOTS/Results/avg_scores.json"
    path_2 = "../OUTPUTS/NOT_RELEVANT_OUT/Results/avg_scores.json"

    analyzer = DifferenceAnalyzer(path_relevant=path_1, path_not_relevant=path_2)
    differences = analyzer(visualize=True, threshold_percent=20)

    print("Top 10 differences:")
    print(differences)
