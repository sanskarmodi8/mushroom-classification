from pathlib import Path
from dataclasses import dataclass
from box import ConfigBox

@dataclass(frozen=True)
class DataIngestionConfig:
    root_dir: Path
    source_url: str
    file_path: Path
    unzip_dir: Path
    
@dataclass(frozen=True)
class DataTransformationConfig:
    root_dir: Path
    data_path: Path
    transformed_data_dir: Path
    
@dataclass(frozen=True)
class ModelTrainingConfig:
    root_dir: Path
    model: Path
    transformed_data: Path
    model_params: ConfigBox
    test_data: Path
    mlflow_tracking_uri: str
    
@dataclass(frozen=True)
class ModelEvaluationConfig:
    root_dir: Path
    model: Path
    test_data: Path
    scores: Path
    mlflow_tracking_uri: str