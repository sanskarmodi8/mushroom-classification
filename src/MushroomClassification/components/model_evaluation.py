from MushroomClassification import logger
from MushroomClassification.entity.config_entity import ModelEvaluationConfig
import pandas as pd
from MushroomClassification.utils.common import save_json, load_bin
from pathlib import Path
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
import mlflow
from urllib.parse import urlparse

class ModelEvaluation:
    def __init__(self, config: ModelEvaluationConfig):
        self.config = config
        
        # load the test data and the model
        self.df = pd.read_csv(self.config.test_data)
        self.model = load_bin(Path(self.config.model))
        logger.info("Loaded test data and model")
        
    def evaluate_without_logging_into_mlflow(self):
        
        # make predictions
        X_test, y_test = self.df.drop("class", axis=1), self.df["class"]
        y_pred_test = self.model.predict(X_test)
        
        # Calculate test set metrics
        test_metrics = {
            "accuracy": accuracy_score(y_test, y_pred_test),
            "precision": precision_score(y_test, y_pred_test),
            "recall": recall_score(y_test, y_pred_test),
            "f1_score": f1_score(y_test, y_pred_test)
        }

        # Save metrics to a JSON file
        save_json(data=test_metrics, path=Path(self.config.scores))
        logger.info(f"Model Evaluated. Test dataset scores: {test_metrics}")

    def evaluate_with_logging_into_mlflow(self):
        
        # configure mlflow for logging and tracking
        mlflow.set_tracking_uri(self.config.mlflow_tracking_uri)
        with mlflow.start_run():
            
            # make predictions
            X_test, y_test = self.df.drop("class", axis=1), self.df["class"]
            
            # Drop the 'Unnamed: 0' column if it exists
            if 'Unnamed: 0' in X_test.columns:
                X_test = X_test.drop(columns=['Unnamed: 0'])
            
            y_pred_test = self.model.predict(X_test)
            
            # Calculate test set metrics
            test_metrics = {
                "accuracy": accuracy_score(y_test, y_pred_test),
                "precision": precision_score(y_test, y_pred_test),
                "recall": recall_score(y_test, y_pred_test),
                "f1_score": f1_score(y_test, y_pred_test)
            }

            # Log metrics to mlflow
            mlflow.log_metrics(test_metrics)
            
            # Save metrics to a JSON file
            save_json(data=test_metrics, path=Path(self.config.scores))
            logger.info(f"Model Evaluated. Test dataset scores: {test_metrics}")