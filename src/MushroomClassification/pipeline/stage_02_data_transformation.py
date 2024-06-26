from MushroomClassification.config.configuration import ConfigurationManager
from MushroomClassification.components.data_transformation import DataTransformation
from MushroomClassification import logger

class DataTransformationPipeline:
    def __init__(self):
        self.config = ConfigurationManager().get_data_transformation_config()
        self.data_transformation = DataTransformation(self.config)
        
    def main(self):
        self.data_transformation.transform_data()
        
        
if __name__ == "__main__":
    STAGE_NAME = "DATA TRANSFORMATION STAGE"
    try:
        logger.info(f"\n\n>>>>>> {STAGE_NAME} started <<<<<<\n\n")
        obj = DataTransformationPipeline()
        obj.main()
        logger.info(f"\n\n>>>>>> stage {STAGE_NAME} completed <<<<<<\n\n")
    except Exception as e:
        logger.exception(e)
        raise e
        