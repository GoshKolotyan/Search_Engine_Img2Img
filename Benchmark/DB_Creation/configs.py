EMBEDDING_MODEL = "swin_s"
NUMBER_OF_TREES_FOR_ANNOY_INDEX = 200

METADATA_PATH_FOR_SAVING = "swin_s/features_metadata.pkl"
INDEX_PATH_FOR_SAVING = "swin_s/image_search_index.ann"

MAX_SIZE = 1024

DB_CONFIGS = {
    "db_name": "wcproject_v0",
    "db_user": "postgres",
    "db_password": "tensor31",
    "db_host": "localhost",
    "db_port": "5432",
}

OUTPUT_BASE_FOLDER = "../Dataset Selection/Images_frankwebb"

MODEL_NAMES = [
    "swin_b",
    "swin_s",
    "swin_t",
    "swin_v2_b",
    "swin_v2_s",
    "swin_v2_t",
    "dinov2_vits14_reg_lc",
    "dinov2_vitb14_reg_lc",
    "dinov2_vitl14_reg_lc",
    "dinov2_vitg14_reg_lc",
]
