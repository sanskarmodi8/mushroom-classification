from pathlib import Path

import mlflow
import pandas as pd
from sklearn.preprocessing import OneHotEncoder

from MushroomClassification.utils.common import load_bin


class Prediction:
    def __init__(self, input_data):
        self.input_data = input_data

    def transform(self):
        """
        Transforms the input data according to the model's requirements.

        1. Converts the input data to a pandas DataFrame.
        2. Replaces hyphens with underscores in column names.
        3. One-hot encodes the categorical data using sklearn.preprocessing.OneHotEncoder.
        4. Converts the encoded data to a pandas DataFrame.
        5. Adds missing columns from expected_features_by_model with values of 0.
        6. Filters the DataFrame to keep only columns present in expected_features_by_model.

        :return: transformed DataFrame
        """

        expected_features_by_model = [
            "bruises_f",
            "bruises_t",
            "odor_f",
            "odor_n",
            "gill-spacing_w",
            "gill-size_b",
            "gill-size_n",
            "gill-color_b",
            "stalk-surface-above-ring_k",
            "stalk-surface-above-ring_s",
            "stalk-surface-below-ring_k",
            "stalk-surface-below-ring_s",
            "ring-type_l",
            "ring-type_p",
            "spore-print-color_h",
            "spore-print-color_k",
            "spore-print-color_n",
            "spore-print-color_w",
            "population_v",
        ]

        # convert to dataframe
        self.df = pd.DataFrame([self.input_data])

        # Replace hyphens with underscores in column names
        self.df.columns = self.df.columns.str.replace("-", "_")

        cols = [col for col in self.df.columns]

        # one hot encode
        encoder = OneHotEncoder(sparse_output=False)
        data_encoded = encoder.fit_transform(self.df)

        # Convert encoded data to DataFrame
        df_encoded = pd.DataFrame(
            data_encoded, columns=encoder.get_feature_names_out(cols)
        )
        self.df = df_encoded.astype("int")

        # Add missing columns from expected_features_by_model with values of 0
        missing_columns = list(set(expected_features_by_model) - set(self.df.columns))

        for col in missing_columns:
            self.df[col] = 0

        # Filter DataFrame to keep only columns present in expected_features_by_model
        self.df = self.df[expected_features_by_model]

    def classify(self):
        """
        Classify the given input data using the model from MLflow.

        If the model from MLflow cannot be loaded, the local model is used as a fallback.

        Returns:
            A list of the predicted class labels.
        """
        model = None

        # transform the data
        self.transform()

        # load model from mlflow
        logged_model = "runs:/f545379e93534628a9516c942aee14da/model"
        try:
            # Attempt to load the model from MLflow
            model = mlflow.pyfunc.load_model(logged_model)
            print("MLflow model loaded successfully")
        except Exception as e:
            print(f"Error loading MLflow model: {e}")

            if model == None:
                # Fallback to loading the local model
                model = load_bin(
                    Path("artifacts/model_training/decision_tree_model.joblib")
                )
                print("Local model loaded successfully")

        # return the result
        return model.predict(self.df).tolist()
