from abc import ABC, abstractmethod
from pathlib import Path

import mlflow
import pandas as pd
from dotenv import load_dotenv
from sklearn.metrics import (accuracy_score, f1_score, precision_score,
                             recall_score)

from MushroomClassification import logger
from MushroomClassification.entity.config_entity import ModelEvaluationConfig
from MushroomClassification.utils.common import load_bin, save_json


class ModelEvaluationStrategy(ABC):
    @abstractmethod
    def evaluate(self, model, X_test, y_test, config):
        """
        Abstract method to evaluate the model using the given test data and configuration.

        Parameters
        ----------
        model : object
            Trained model.
        X_test : DataFrame
            Features of the test dataset.
        y_test : Series
            Target of the test dataset.
        config : ModelEvaluationConfig
            Configuration for the Model Evaluation stage.
        """
        pass


class EvaluationWithoutMLflowLogging(ModelEvaluationStrategy):
    def evaluate(self, model, X_test, y_test, config):
        """
        Evaluate the model using the given test data and configuration.

        Parameters
        ----------
        model : object
            Trained model.
        X_test : DataFrame
            Features of the test dataset.
        y_test : Series
            Target of the test dataset.
        config : ModelEvaluationConfig
            Configuration for the Model Evaluation stage.

        Notes
        -----
        This method does not use MLflow for logging and tracking.
        """
        y_pred_test = model.predict(X_test)

        # Calculate test set metrics
        test_metrics = {
            "accuracy": accuracy_score(y_test, y_pred_test),
            "precision": precision_score(y_test, y_pred_test),
            "recall": recall_score(y_test, y_pred_test),
            "f1_score": f1_score(y_test, y_pred_test),
        }

        # Save metrics to a JSON file
        save_json(data=test_metrics, path=Path(config.scores))
        logger.info(f"Model Evaluated. Test dataset scores: {test_metrics}")


class EvaluationWithMLflowLogging(ModelEvaluationStrategy):
    def evaluate(self, model, X_test, y_test, config):
        """
        Evaluate the model using the given test data and configuration.

        Parameters
        ----------
        model : object
            Trained model.
        X_test : DataFrame
            Features of the test dataset.
        y_test : Series
            Target of the test dataset.
        config : ModelEvaluationConfig
            Configuration for the Model Evaluation stage.

        Notes
        -----
        This method uses MLflow for logging and tracking.
        """
        mlflow.set_tracking_uri(config.mlflow_tracking_uri)
        with mlflow.start_run():
            y_pred_test = model.predict(X_test)

            # Calculate test set metrics
            test_metrics = {
                "accuracy": accuracy_score(y_test, y_pred_test),
                "precision": precision_score(y_test, y_pred_test),
                "recall": recall_score(y_test, y_pred_test),
                "f1_score": f1_score(y_test, y_pred_test),
            }

            # Log metrics to mlflow
            mlflow.log_metrics(test_metrics)

            # Save metrics to a JSON file
            save_json(data=test_metrics, path=Path(config.scores))
            logger.info(f"Model Evaluated. Test dataset scores: {test_metrics}")


class ModelEvaluation:
    def __init__(self, config: ModelEvaluationConfig, enable_mlflow_logging: bool):
        """
        Initializes the ModelEvaluation object with the given configuration and
        strategy.

        Parameters
        ----------
        config : ModelEvaluationConfig
            Configuration for the Model Evaluation stage.
        enable_mlflow_logging : bool
            Whether to use MLflow for logging and tracking the model.

        Attributes
        ----------
        config : ModelEvaluationConfig
            Configuration for the Model Evaluation stage.
        evaluation_strategy : EvaluationStrategy
            Evaluation strategy to use for the model evaluation.
        df : DataFrame
            The test dataset.
        model : object
            The trained model.
        """
        self.config = config
        self.evaluation_strategy = (
            EvaluationWithMLflowLogging()
            if enable_mlflow_logging
            else EvaluationWithoutMLflowLogging()
        )

        # Load the test data and the model
        self.df = pd.read_csv(self.config.test_data)
        self.model = load_bin(Path(self.config.model))
        logger.info("Loaded test data and model")

    def evaluate(self):
        """
        Evaluate the model using the given test data and configuration.

        """
        X_test, y_test = self.df.drop("class", axis=1), self.df["class"]

        # Delegate evaluation to the strategy
        self.evaluation_strategy.evaluate(self.model, X_test, y_test, self.config)
