from MushroomClassification import logger
from MushroomClassification.components.model_evaluation import ModelEvaluation
from MushroomClassification.config.configuration import ConfigurationManager

class ModelEvaluationPipeline:
    def __init__(self):
        self.config = ConfigurationManager().get_model_evaluation_config()
        self.model_evaluation = ModelEvaluation(self.config)
        
    def main(self):
        self.model_evaluation.evaluate_with_logging_into_mlflow()
        # self.model_evaluation.evaluate_without_logging_into_mlflow()
        
if __name__=="__main__":
    STAGE_NAME = "MODEL EVALUATION STAGE"
    try:
        logger.info(f"\n\n>>>>>> {STAGE_NAME} started <<<<<<\n\n")
        obj = ModelEvaluationPipeline()
        obj.main()
        logger.info(f"\n\n>>>>>> stage {STAGE_NAME} completed <<<<<<\n\n")
    except Exception as e:
        logger.exception(e)
        raise e