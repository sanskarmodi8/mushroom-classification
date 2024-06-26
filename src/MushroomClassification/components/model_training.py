from MushroomClassification import logger
from MushroomClassification.entity.config_entity import ModelTrainingConfig
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from MushroomClassification.utils.common import save_bin, save_json
from pathlib import Path

class ModelTraining:
    def __init__(self, config:ModelTrainingConfig):
        self.config = config
        
        # load transformed data 
        self.df = pd.read_csv(self.config.transformed_data)
        logger.info("Loaded transformed data")
        
    def data_splits(self):
        df_transformed= self.df
        # Split data into train and test datasets
        train_df, test_df = train_test_split(df_transformed, test_size=0.2, random_state=42, stratify=df_transformed['class'])

        # split train df to X and y
        self.X_train, self.y_train = train_df.drop("class",axis=1), train_df["class"]
        logger.info("Splitted the loaded data into X_train, y_train, and, test_df")
        
        test_df.to_csv(self.config.test_data, index=False)
        logger.info(f"Saved the test dataset to - {self.config.test_data}")
        
    def train_model(self):
        model= DecisionTreeClassifier(**self.config.model_params)
        model.fit(self.X_train, self.y_train)
        save_bin(model, Path(self.config.model))
        
        logger.info(f"Model trained and saved at - {self.config.model}")
