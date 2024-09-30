from dotenv import load_dotenv

from src.MushroomClassification.pipeline.pipeline import \
    classification_pipeline
from src.MushroomClassification.pipeline.stage_01_data_ingestion import \
    data_ingestion_step
from src.MushroomClassification.pipeline.stage_02_data_transformation import \
    data_transformation_step
from src.MushroomClassification.pipeline.stage_03_model_training import \
    model_training_step
from src.MushroomClassification.pipeline.stage_04_model_evaluation import \
    model_evaluation_step

# load the env variables for the mlflow tracking
load_dotenv()

if __name__ == "__main__":
    # run the pipeline with all the steps
    classification_pipeline(
        data_ingestion_step(),
        data_transformation_step(),
        model_training_step(),
        model_evaluation_step(),
    ).run()
