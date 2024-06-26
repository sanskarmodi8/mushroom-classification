from MushroomClassification.entity.config_entity import DataIngestionConfig
from MushroomClassification.constants import PARAMS_FILE_PATH, CONFIG_FILE_PATH
from MushroomClassification.utils.common import read_yaml, create_directories
from pathlib import Path

class ConfigurationManager:
    def __init__(self):
        self.params = read_yaml(PARAMS_FILE_PATH)
        self.config = read_yaml(CONFIG_FILE_PATH)
        
    def get_data_ingestion_config(self) -> DataIngestionConfig:
        config = self.config.data_ingestion
        create_directories([Path(config.root_dir), Path(config.unzip_dir)])
        return DataIngestionConfig(root_dir=config.root_dir,
                                source_url=config.source_url,
                                file_path=config.file_path,
                                unzip_dir=config.unzip_dir)
    