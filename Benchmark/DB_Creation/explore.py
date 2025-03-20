import os
import logging
import psycopg2
from typing import Dict, List, Optional
from configs import MAX_SIZE
from model_loader import ImageSegmenter

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("exploration_database.log"),  # for Log to file
        logging.StreamHandler(),  # for log to console
    ],
)


class DatabaseConnection:
    def __init__(
        self, db_name: str, db_user: str, db_password: str, db_host: str, db_port: str
    ):
        self.db_name = db_name
        self.db_user = db_user
        self.db_password = db_password
        self.db_host = db_host
        self.db_port = db_port
        self.connection: Optional[psycopg2.extensions.connection] = None
        self.cursor: Optional[psycopg2.extensions.cursor] = None

    def __enter__(self):
        self.connect()
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()

    def connect(self):
        try:
            self.connection = psycopg2.connect(
                dbname=self.db_name,
                user=self.db_user,
                password=self.db_password,
                host=self.db_host,
                port=self.db_port,
            )
            self.cursor = self.connection.cursor()
            logging.info("Successfully connected to the database")
        except psycopg2.Error as e:
            logging.error(f"Database connection error: {e}")

    def execute_query(self, query):
        if not self.cursor:
            logging.error("Query execution failed: No active database connection.")
            return None
        try:
            self.cursor.execute(query)
            return self.cursor.fetchall()
        except psycopg2.Error as e:
            logging.error(f"Query execution error: {e}")
            return None

    def close(self):
        if self.cursor:
            self.cursor.close()
            logging.info("Cursor closed")
        if self.connection:
            self.connection.close()
            logging.info("Database connection closed")


class ImageCategorizer:
    """
    Simply fetches product data from the DB, determines final categories,
    and sends each original image path to the ImageSegmenter (no copying).
    """

    def __init__(self, db_config: Dict[str, str], output_folder: str):
        self.db = DatabaseConnection(**db_config)
        self.final_cats = {
            "bathtub": "tub",
            "shower": "tub",
            "sink": "sink",
            "furniture": "cabinet",
            "toilet": "toilet",
        }
        self.output_folder = output_folder
        logging.info(
            "ImageCategorizer initialized with output folder: %s", self.output_folder
        )

    def fetch_category_mapping(self):

        query = "SELECT id, name FROM categories;"
        logging.info("Executing category mapping for query: %s", query)

        results = self.db.execute_query(query)

        if results is None:
            logging.error("Failed to fetch categroy mapping. None results")
            return None

        logging.info("Successfully fetched categories from the database.")

        return {str(row[0]): row[1].lower() for row in results}

    def fetch_product_data(self):
        query = """
            SELECT
                category_id,
                STRING_AGG(id::text, ', ' ORDER BY id) AS product_ids,
                COUNT(*) AS product_count,
                STRING_AGG(image_path, ', ' ORDER BY id) AS image_paths
            FROM products
            GROUP BY category_id
            ORDER BY product_count DESC;
        """

        logging.info("Execute product data query")

        results = self.db.execute_query(query)

        if results is None:
            logging.error("Failed to fetch product data. None results")

        logging.info("Successfully fetched %d product categories from the database.")
        return results

    def process_results(self):
        """
        Builds a list of (category, product_ids, image_paths) where 'category'
        is the final mapped category, e.g. "tub", "sink", etc.
        """
        self.db.connect()
        category_map = self.fetch_category_mapping()
        results = self.fetch_product_data()
        self.db.close()

        processed_results = []

        for row in results:
            category_id, product_ids, product_count, image_paths = row
            category_id = str(category_id)
            original_category = category_map.get(
                category_id, f"Unknown ({category_id})"
            )
            merged_category = self.final_cats.get(original_category, None)

            if original_category == "accessory":
                # If category is "accessory", only keep paths with "mirror" TODO need to be changed mirror as category
                image_list = image_paths.split(", ")
                product_ids_list = product_ids.split(", ")
                filtered = [
                    (pid, img)
                    for pid, img in zip(product_ids_list, image_list)
                    if "mirror" in img.lower()
                ]
                if filtered:
                    merged_category = "mirror"
                    product_ids = ", ".join(pid for pid, _ in filtered)
                    image_paths = ", ".join(img for _, img in filtered)
                else:
                    continue  # skip if no mirror found

            # skip if no final category
            if not merged_category:
                continue

            processed_results.append((merged_category, product_ids, image_paths))
        return processed_results

    def run_segmentation(self) -> None:
        """
        1) Get data from DB.
        2) For each image path, pass it to the segmenter for final PNG saving.
        """
        processed_results = self.process_results()
        missing_images = []

        # Instantiate segmenter once
        segmenter = ImageSegmenter(
            output_base_folder=self.output_folder, max_size=MAX_SIZE
        )
        logging.info("Successfully initialized  ImageSegmenter")

        for merged_category, product_ids, image_paths in processed_results:
            prod_ids_list = product_ids.split(", ")
            img_list = image_paths.split(", ")

            for prod_id, image_path in zip(prod_ids_list, img_list):
                # Normalize path
                image_path = image_path.replace("\\", "/")
                # optional: remove double slashes
                image_path = image_path.replace("//", "/")
                # turn into absolute path
                image_path = os.path.abspath(image_path)

                if os.path.exists(image_path):
                    # We do NOT copy. We simply pass the original path to the segmenter.
                    image_np = segmenter.process_image(
                        input_image_path=image_path,
                        product_id=prod_id,
                        output_subfolder=merged_category,
                    )
                    if image_np is not None:
                        logging.info(
                            f"Image processed, Category: {merged_category}, shape: {image_np.shape}"
                        )
                    else:
                        logging.info(f"Failed to process image {image_path}")
                else:
                    missing_images.append(image_path)

        if missing_images:
            logging.info(f"Missing images: {[image for image in missing_images]}")

        logging.info(f"Segmentation completed saved to {self.output_folder}")
