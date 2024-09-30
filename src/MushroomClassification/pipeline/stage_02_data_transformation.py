from zenml.steps import step

from MushroomClassification import logger
from MushroomClassification.components.data_transformation import \
    DataTransformation
from MushroomClassification.config.configuration import ConfigurationManager

STAGE_NAME = "Data Transformation Stage"


@step
def data_transformation_step(success: bool) -> bool:
    """
    This ZenML step handles the data transformation process which involves performing basic EDA operations, and preprocessing the data.
    """
    try:

        logger.info(f"\n\n>>>>> {STAGE_NAME} started. <<<<<\n\n")
        config_manager = ConfigurationManager()
        config = config_manager.get_data_transformation_config()
        data_transformation = DataTransformation(config)

        # Start the data transformation process
        logger.info("Transforming the data...")
        data_transformation.transform_data()

        logger.info(f"\n\n>>>>> {STAGE_NAME} completed. <<<<<\n\n")

        return True

    except Exception as e:
        logger.exception(f"Error during transformation: {e}")
        raise e
