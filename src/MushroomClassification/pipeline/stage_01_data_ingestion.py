from zenml.steps import step

from MushroomClassification import logger
from MushroomClassification.components.data_ingestion import DataIngestion
from MushroomClassification.config.configuration import ConfigurationManager

STAGE_NAME = "Data Ingestion Stage"


@step
def data_ingestion_step() -> bool:
    """
    This ZenML step handles the data ingestion process, which involves downloading
    and extracting the dataset.
    """
    try:

        logger.info(f"\n\n>>>>> {STAGE_NAME} started. <<<<<\n\n")

        config_manager = ConfigurationManager()
        config = config_manager.get_data_ingestion_config()
        data_ingestion = DataIngestion(config)
        data_ingestion = data_ingestion.create_data_ingestion()

        # Start the data ingestion process
        logger.info("Downloading data...")
        data_ingestion.download_file()

        logger.info("Extracting data...")
        data_ingestion.extract_zip_file()

        logger.info(f"\n\n>>>>> {STAGE_NAME} completed. <<<<<\n\n")
        return True
    except Exception as e:
        logger.exception(f"Error during data ingestion: {e}")
        raise e
