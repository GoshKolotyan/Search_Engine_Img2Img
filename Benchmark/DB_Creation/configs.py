from typing import List, Dict
from dataclasses import dataclass, field

NUMBER_OF_TREES_FOR_ANNOY_INDEX = 200


MAX_SIZE = 1024


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


# @dataclass(frozen=False)
class GlobalConfigs:
    MODEL_NAME: List[str] = [
        "swin_b", "swin_s", "swin_t", "swin_v2_b", "swin_v2_s", "swin_v2_t",
        "dinov2_vits14_reg_lc", "dinov2_vitb14_reg_lc", "dinov2_vitl14_reg_lc", "dinov2_vitg14_reg_lc"
    ]
    OUTPUT_BASE_FOLDER: str = "../Dataset Selection/Images_frankwebb"
    MAX_SIZE: int = 1024
    NUMBER_OF_TREES_FOR_ANNOY_INDEX: int = 200

    def __repr__(self):
        return (f"GlobalConfigs(MODEL_NAME={self.MODEL_NAME}, "
                f"OUTPUT_BASE_FOLDER='{self.OUTPUT_BASE_FOLDER}', "
                f"MAX_SIZE={self.MAX_SIZE}, "
                f"NUMBER_OF_TREES_FOR_ANNOY_INDEX={self.NUMBER_OF_TREES_FOR_ANNOY_INDEX})")

global_config = GlobalConfigs()

@dataclass(frozen=True)
class DBConfigs:
    NAME: str = "wcproject_v0"
    USER: str = "postgres"
    PASSWORD: str = "tensor31"
    HOST: str = "localhost"
    PORT: str = "5432"

    def __repr__(self):
        return f"DBConfigs(NAME={self.NAME}, USER={self.USER}, PASSWORD={self.PASSWORD} HOST={self.HOST}, PORT={self.PORT})"

db_config = DBConfigs()

