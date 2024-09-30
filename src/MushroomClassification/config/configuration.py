from pathlib import Path

from MushroomClassification.constants import CONFIG_FILE_PATH, PARAMS_FILE_PATH
from MushroomClassification.entity.config_entity import (
    DataIngestionConfig, DataTransformationConfig, ModelEvaluationConfig,
    ModelTrainingConfig)
from MushroomClassification.utils.common import create_directories, read_yaml


class ConfigurationManager:
    def __init__(self):
        self.params = read_yaml(PARAMS_FILE_PATH)
        self.config = read_yaml(CONFIG_FILE_PATH)

    def get_data_ingestion_config(self) -> DataIngestionConfig:
        config = self.config.data_ingestion
        create_directories([Path(config.root_dir), Path(config.unzip_dir)])
        return DataIngestionConfig(
            root_dir=config.root_dir,
            source_url=config.source_url,
            file_path=config.file_path,
            unzip_dir=config.unzip_dir,
        )

    def get_data_transformation_config(self) -> DataTransformationConfig:
        config = self.config.data_transformation
        create_directories([Path(config.root_dir)])
        return DataTransformationConfig(
            root_dir=config.root_dir,
            data_path=config.data_path,
            transformed_data=config.transformed_data,
        )

    def get_model_training_config(self) -> ModelTrainingConfig:
        config = self.config.model_training
        create_directories([Path(config.root_dir)])
        return ModelTrainingConfig(
            root_dir=config.root_dir,
            model=config.model,
            transformed_data=config.transformed_data,
            model_params=self.params.model_params,
            test_data=config.test_data,
            mlflow_tracking_uri=config.mlflow_tracking_uri,
        )

    def get_model_evaluation_config(self) -> ModelEvaluationConfig:
        config = self.config.model_evaluation
        create_directories([Path(config.root_dir)])
        return ModelEvaluationConfig(
            root_dir=config.root_dir,
            model=config.model,
            test_data=config.test_data,
            scores=config.scores,
            mlflow_tracking_uri=config.mlflow_tracking_uri,
        )
