from src.MushroomClassification.pipeline.stage_01_data_ingestion import DataIngestionPipeline
from src.MushroomClassification.pipeline.stage_02_data_transformation import DataTransformationPipeline
from src.MushroomClassification.pipeline.stage_03_model_training import ModelTrainingPipeline
from src.MushroomClassification.pipeline.stage_04_model_evaluation import ModelEvaluationPipeline
from src.MushroomClassification import logger

STAGE_NAME = "DATA INGESTION STAGE"
try:
    logger.info(f"\n\n>>>>>> {STAGE_NAME} started <<<<<<\n\n")
    obj = DataIngestionPipeline()
    obj.main()
    logger.info(f"\n\n>>>>>> stage {STAGE_NAME} completed <<<<<<\n\n")
except Exception as e:
    logger.exception(e)
    raise e

STAGE_NAME = "DATA TRANSFORMATION STAGE"
try:
    logger.info(f"\n\n>>>>>> {STAGE_NAME} started <<<<<<\n\n")
    obj = DataTransformationPipeline()
    obj.main()
    logger.info(f"\n\n>>>>>> stage {STAGE_NAME} completed <<<<<<\n\n")
except Exception as e:
    logger.exception(e)
    raise e

STAGE_NAME = "MODEL TRAINING STAGE"
try:
    logger.info(f"\n\n>>>>>> {STAGE_NAME} started <<<<<<\n\n")
    obj = ModelTrainingPipeline()
    obj.main()
    logger.info(f"\n\n>>>>>> stage {STAGE_NAME} completed <<<<<<\n\n")
except Exception as e:
    logger.exception(e)
    raise e

STAGE_NAME = "MODEL EVALUATION STAGE"
try:
    logger.info(f"\n\n>>>>>> {STAGE_NAME} started <<<<<<\n\n")
    obj = ModelEvaluationPipeline()
    obj.main()
    logger.info(f"\n\n>>>>>> stage {STAGE_NAME} completed <<<<<<\n\n")
except Exception as e:
    logger.exception(e)
    raise e