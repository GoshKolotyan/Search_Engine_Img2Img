import os
import logging
import glob
from pathlib import Path
from analyzer import Analyzer
from benchmark import RBenchmarking
from configs import MODEL_NAMES, IMAGES_DIR, OUTPUT_DIR


def setup_directories():
    Path(f"../{OUTPUT_DIR}").mkdir(parents=True, exist_ok=True)
    Path(f"../{OUTPUT_DIR}/logs").mkdir(parents=True, exist_ok=True)


def setup_logging():
    logging.basicConfig(
        format="{asctime} - {levelname} - {message}",
        level=logging.INFO,
        style="{",
        datefmt="%Y-%m-%d %H:%M",
        filename=f"../{OUTPUT_DIR}/logs/info.log",
    )
    logging.info("Application started successfully.")


def process_image_folder(folder_path: str):
    if not Path(folder_path).exists():
        logging.warning(f"Skipping {folder_path}: Directory not found.")
        return

    for model_name in MODEL_NAMES:
        logging.info(f"Processing model: {model_name} on folder: {folder_path}")

        try:
            rb = RBenchmarking(folder_path=folder_path, model_name=model_name, output_dir=OUTPUT_DIR)
            aug_results = rb.compute_augmented_similarities_for_all_images()

            sorted_aug_results = sorted(aug_results.items(), key=lambda x: x[1])

            rb.plot(sorted_results=sorted_aug_results)

            sum_score = [res[1] for res in sorted_aug_results]
            average_score = sum(sum_score) / len(sum_score) if sum_score else 0.0

            rb._record_to_csv(similarity_scores=average_score, model_name=model_name)

        except Exception as e:
            logging.error(f"Error processing {model_name} on {folder_path}: {e}")


def analyze_results():
    csv_dir = Path(f"../{OUTPUT_DIR}")
    csv_files = glob.glob(str(csv_dir / "**/*.csv"), recursive=True)

    if not csv_files:
        logging.warning("No CSV files found for analysis.")
        return

    analyzer = Analyzer(paths=csv_files, output_dir=OUTPUT_DIR)
    analyzer()


if __name__ == "__main__":
    setup_directories()
    setup_logging()

    image_folders = [os.path.join(IMAGES_DIR, folder) for folder in os.listdir(IMAGES_DIR)]
    
    if not image_folders:
        logging.warning(f"No image folders found in {IMAGES_DIR}. Exiting.")
    else:
        for folder in image_folders:
            process_image_folder(folder)

        analyze_results()
