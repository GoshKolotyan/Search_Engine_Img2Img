import logging
import os
from configs import OUTPUT_BASE_FOLDER, MODEL_NAMES
from indexing import ModelDatabaseBuilder
from typing import List

# Create logs folder if it doesn't exist
log_folder = 'logs'
os.makedirs(log_folder, exist_ok=True)

# Configure logging to save to file
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(os.path.join(log_folder, 'indexing.log')),
        logging.StreamHandler()
    ]
)

def main(output_base_folder: str, model_names: List[str]) -> None:
    """
    Read Data -> Load Model -> Segment Element -> Keep in DB in our selected method.
    """
    logging.info(f"Starting indexing for models: {', '.join(model_names)}")

    for name in model_names:
        logging.info(f"Initializing ModelDatabaseBuilder for model: {name}")
        index_creator = ModelDatabaseBuilder(output_base_folder, model_name=name)
        logging.info(f"Building and saving database for model: {name}")
        index_creator.build_database_and_save()
        logging.info(f"Completed indexing for model: {name}")

    logging.info("All models have been successfully processed.")

if __name__ == "__main__":
    main(output_base_folder=OUTPUT_BASE_FOLDER, model_names=MODEL_NAMES)