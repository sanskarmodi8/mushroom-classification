from MushroomClassification.components.data_ingestion import DataIngestion
from MushroomClassification import logger
from MushroomClassification.config.configuration import ConfigurationManager

class DataIngestionPipeline:
    def __init__(self):
        self.config_manager = ConfigurationManager()
        self.config = self.config_manager.get_data_ingestion_config()
        self.data_ingestion = DataIngestion(self.config)
        
    def main(self):
        self.data_ingestion.download_file()
        self.data_ingestion.extract_zip_file()
        
if __name__ == '__main__':
    STAGE_NAME = "DATA INGESTION STAGE"
    try:
        logger.info(f"\n\n>>>>>> {STAGE_NAME} started <<<<<<\n\n")
        obj = DataIngestionPipeline()
        obj.main()
        logger.info(f"\n\n>>>>>> stage {STAGE_NAME} completed <<<<<<\n\n")
    except Exception as e:
        logger.exception(e)
        raise e