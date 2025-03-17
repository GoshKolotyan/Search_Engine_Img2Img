import os
import glob
import torch
import logging
from analyzer import Analyzer
from pathlib import Path
from benchmark import RBenchmarking

from configs import MODEL_NAMES, IMAGES_DIR_ONE_VS_ALL, OUTPUT_DIR_ONE_VS_ALL
def setup_directories():
    Path(f"../{OUTPUT_DIR_ONE_VS_ALL}").mkdir(parents=True, exist_ok=True)
    Path(f"../{OUTPUT_DIR_ONE_VS_ALL}/logs").mkdir(parents=True, exist_ok=True)
    Path(f"../{OUTPUT_DIR_ONE_VS_ALL}/Results").mkdir(parents=True, exist_ok=True)

def setup_logging():
    logging.basicConfig(
        format="{asctime} - {levelname} - {message}",
        level=logging.INFO,
        style="{",
        datefmt="%Y-%m-%d %H:%M",
        filename=f"../{OUTPUT_DIR_ONE_VS_ALL}/logs/info.log",
    )
    logging.info("Application started successfully.")



def process_image_folder(folder_path: str):
    if not Path(folder_path).exists():
        logging.warning(f"Skipping {folder_path}: Directory not found.")
        return

    for model_name in MODEL_NAMES:
        logging.info(f"Processing model: {model_name} on folder: {folder_path}")

        try:
            rb = RBenchmarking(
            folder_path=folder_path, model_name=model_name, output_dir=OUTPUT_DIR_ONE_VS_ALL
        )
            aug_results = rb.compute_augmented_similarities_for_all_images()

            # sorted_aug_results = sorted(aug_results.items(), key=lambda x: x[1])
            sorted_aug_results = sorted(aug_results.items(), key=lambda item: item[1], reverse=True)
            filtered_aug_results = [
                                    (key, value) for (key, value) in sorted_aug_results
                                    if "flipped_horizontally" not in key
                                ]
            names = [item[0] for item in filtered_aug_results[:10]]
            print(f"Mode {model_name} output names is {[name for name in names]}")
            similarity_score = sum("t135" in name for name in names)
            print(similarity_score)
            rb._record_to_csv(model_name=model_name, similarity_scores=f"{similarity_score}/11")


            torch.cuda.empty_cache()

        except Exception as e:
            logging.error(f"Error processing {model_name} on {folder_path}: {e}")

def analyze_results():
    csv_dir = Path(f"../{OUTPUT_DIR_ONE_VS_ALL}")
    csv_files = glob.glob(str(csv_dir / "**/*.csv"), recursive=True)

    if not csv_files:
        logging.warning("No CSV files found for analysis.")
        return

    analyzer = Analyzer(paths=csv_files, output_dir=OUTPUT_DIR_ONE_VS_ALL)
    analyzer()
if __name__ == "__main__":
    print("Starting")

    image_folders = [
        os.path.join(IMAGES_DIR_ONE_VS_ALL, folder)
        for folder in os.listdir(IMAGES_DIR_ONE_VS_ALL)
    ]

    if not image_folders:
        logging.warning(
            f"No image folders found in {IMAGES_DIR_ONE_VS_ALL}. Exiting."
        )
    else:
        for folder in image_folders:
            process_image_folder(folder)
        # analyze_results()


#TODO keep 11 the best

