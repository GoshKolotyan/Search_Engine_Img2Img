from dataclasses import dataclass

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


@dataclass(frozen=True)
class DBConfigs:
    NAME: str = "wcproject_v0"
    USER: str = "postgres"
    PASSWORD: str = "tensor31"
    HOST: str = "localhost"
    PORT: str = "5432"

    def __repr__(self):
        return f"DBConfigs(NAME={self.NAME}, USER={self.USER}, PASSWORD={self.PASSWORD} HOST={self.HOST}, PORT={self.PORT})"

# Creating an instance
db_config = DBConfigs()


print(db_config)  

