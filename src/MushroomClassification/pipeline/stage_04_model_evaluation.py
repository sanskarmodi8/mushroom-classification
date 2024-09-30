from zenml.steps import step

from MushroomClassification import logger
from MushroomClassification.components.model_evaluation import ModelEvaluation
from MushroomClassification.config.configuration import ConfigurationManager

STAGE_NAME = "Model Evaluation Stage"


@step
def model_evaluation_step(success: bool) -> bool:
    """
    This ZenML step handles the model evaluation process.
    """
    try:

        logger.info(f"\n\n>>>>> {STAGE_NAME} started. <<<<<\n\n")
        config_manager = ConfigurationManager()
        config = config_manager.get_model_evaluation_config()
        evaluate = ModelEvaluation(config, enable_mlflow_logging=False)

        # Start the data transformation process
        logger.info("Evaluating model...")
        evaluate.evaluate()

        logger.info(f"\n\n>>>>> {STAGE_NAME} completed. <<<<<\n\n")

        return True
    except Exception as e:
        logger.exception(f"Error during evaluation: {e}")
        raise e
