
from configs import OUTPUT_BASE_FOLDER, MODEL_NAMES

from indexing import ModelDatabaseBuilder
from typing import List

def main(output_base_folder: str, Model_Names: List[str]) -> None:
    """
    Read Data -> Load Model -> Segment Element -> Keep in DB in our selected method.
    """
    print([name for name in Model_Names])
    for name  in Model_Names:
        index_creator = ModelDatabaseBuilder(output_base_folder, model_name=name)


    # index_creator.build_database_and_save()
    return None


if __name__ == "__main__":
    main(output_base_folder=OUTPUT_BASE_FOLDER, Model_Names= MODEL_NAMES)
