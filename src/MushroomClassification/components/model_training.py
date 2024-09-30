from abc import ABC, abstractmethod
from pathlib import Path

import mlflow
import mlflow.sklearn
import pandas as pd
from dotenv import load_dotenv
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier

from MushroomClassification import logger
from MushroomClassification.entity.config_entity import ModelTrainingConfig
from MushroomClassification.utils.common import save_bin

load_dotenv()


class TrainingStrategy(ABC):
    @abstractmethod
    def train_model(self, X_train, y_train, config):
        """
        Abstract method to train the model.

        Parameters
        ----------
        X_train : DataFrame
            Features of the training dataset.
        y_train : Series
            Target of the training dataset.
        config : ConfigBox
            Configuration for the Model Training stage.

        Returns
        -------
        model : object
            Trained model.
        """
        pass


class TrainWithMLflowStrategy(TrainingStrategy):
    def train_model(self, X_train, y_train, config):
        # Configure MLflow for logging and tracking
        """
        Train the model using the provided training data and configuration.

        Parameters
        ----------
        X_train : DataFrame
            Features of the training dataset.
        y_train : Series
            Target of the training dataset.
        config : ConfigBox
            Configuration for the Model Training stage.

        Returns
        -------
        model : object
            Trained model.

        Notes
        -----
        This method uses MLflow to log the model parameters and save the model.
        """
        mlflow.set_tracking_uri(config.mlflow_tracking_uri)

        with mlflow.start_run():
            model = DecisionTreeClassifier(**config.model_params)
            model.fit(X_train, y_train)

            # Log model parameters
            mlflow.log_params(config.model_params)

            # Save the model using MLflow
            mlflow.sklearn.log_model(model, "model")
            save_bin(model, Path(config.model))

            logger.info(f"Model trained and saved with MLflow at - {config.model}")


class TrainWithoutMLflowStrategy(TrainingStrategy):
    def train_model(self, X_train, y_train, config):
        """
        Train the model using the provided training data and configuration.

        Parameters
        ----------
        X_train : DataFrame
            Features of the training dataset.
        y_train : Series
            Target of the training dataset.
        config : ConfigBox
            Configuration for the Model Training stage.

        Notes
        -----
        This method does not use MLflow for logging and tracking.
        """
        model = DecisionTreeClassifier(**config.model_params)
        model.fit(X_train, y_train)
        save_bin(model, Path(config.model))

        logger.info(f"Model trained and saved at - {config.model}")


class ModelTraining:
    def __init__(self, config: ModelTrainingConfig, enable_mlflow_logging: bool):
        """
        Initializes the ModelTraining object with the given configuration and
        strategy.

        Parameters
        ----------
        config : ModelTrainingConfig
            Configuration for the Model Training stage.
        enable_mlflow_logging : bool
            Whether to use MLflow for logging and tracking the model.
        """
        self.config = config
        self.strategy = (
            TrainWithMLflowStrategy()
            if enable_mlflow_logging
            else TrainWithoutMLflowStrategy()
        )

        # load transformed data
        self.df = pd.read_csv(self.config.transformed_data)
        logger.info("Loaded transformed data")

    def data_splits(self):

        df_transformed = self.df
        # Split data into train and test datasets
        train_df, test_df = train_test_split(
            df_transformed,
            test_size=0.2,
            random_state=42,
            stratify=df_transformed["class"],
        )

        # split train df to X and y
        self.X_train, self.y_train = train_df.drop("class", axis=1), train_df["class"]
        logger.info("Splitted the loaded data into X_train, y_train, and test_df")

        test_df.to_csv(self.config.test_data, index=False)
        logger.info(f"Saved the test dataset to - {self.config.test_data}")

    def train_model(self):
        # Use the strategy to train the model
        self.strategy.train_model(self.X_train, self.y_train, self.config)
