from zenml.steps import step

from MushroomClassification import logger
from MushroomClassification.components.model_training import ModelTraining
from MushroomClassification.config.configuration import ConfigurationManager

STAGE_NAME = "Model Training Stage"


@step
def model_training_step(success: bool) -> bool:
    """
    This ZenML step handles the model training process.
    """
    try:

        logger.info(f"\n\n>>>>> {STAGE_NAME} started. <<<<<\n\n")
        config_manager = ConfigurationManager()
        config = config_manager.get_model_training_config()
        training = ModelTraining(config, enable_mlflow_logging=False)

        # Start the data transformation process
        logger.info("Doing data splits and Training model...")
        training.data_splits()
        training.train_model()

        logger.info(f"\n\n>>>>> {STAGE_NAME} completed. <<<<<\n\n")

        return True
    except Exception as e:
        logger.exception(f"Error during training: {e}")
        raise e
