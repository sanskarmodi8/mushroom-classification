from MushroomClassification import logger
from MushroomClassification.components.model_training import ModelTraining
from MushroomClassification.config.configuration import ConfigurationManager

class ModelTrainingPipeline:
    def __init__(self):
        self.config = ConfigurationManager().get_model_training_config()
        self.model_training = ModelTraining(self.config)
        
    def main(self):
        self.model_training.data_splits()
        self.model_training.train_model_with_logging_into_mlflow()
        # self.model_training.train_model_without_logging_into_mlflow()
        
if __name__=="__main__":
    STAGE_NAME = "MODEL TRAINING STAGE"
    try:
        logger.info(f"\n\n>>>>>> {STAGE_NAME} started <<<<<<\n\n")
        obj = ModelTrainingPipeline()
        obj.main()
        logger.info(f"\n\n>>>>>> stage {STAGE_NAME} completed <<<<<<\n\n")
    except Exception as e:
        logger.exception(e)
        raise e