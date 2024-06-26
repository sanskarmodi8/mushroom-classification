from src.MushroomClassification.pipeline.stage_01_data_ingestion import DataIngestionPipeline
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