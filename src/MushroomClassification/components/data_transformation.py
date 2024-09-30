from abc import ABC, abstractmethod

import pandas as pd
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder

from MushroomClassification import logger
from MushroomClassification.entity.config_entity import \
    DataTransformationConfig


# Define Abstract Base Class for EDA Strategy
class EDAAnalysisStrategy(ABC):
    @abstractmethod
    def execute(self, df):
        """
        Execute EDA operations on given DataFrame.
        """
        pass


# Define Abstract Base Class for Feature Engineering Strategy
class FeatureEngineeringStrategy(ABC):
    @abstractmethod
    def preprocess(self, X, y):
        """
        Preprocess the given DataFrame X and target vector y.
        """
        pass

    @abstractmethod
    def select_features(self, X, y, k=19):
        """
        Select the top k features from given DataFrame X and target vector y.
        """
        pass

    @abstractmethod
    def remove_outliers(self, X):
        """
        Remove outliers from given DataFrame X.
        """
        pass


# Concrete Strategy for EDA
class EDAConcreteStrategy(EDAAnalysisStrategy):
    def execute(self, df):
        # Perform basic EDA operations, e.g., describe, info, value counts
        """
        Perform basic EDA operations, e.g., describe, info, value counts.
        """
        logger.info("Performing EDA:")
        logger.info(df.describe())
        logger.info(df.info())
        for col in df.select_dtypes(include=["object"]).columns:
            logger.info(f"{col} value counts: \n{df[col].value_counts()}")


# Concrete Strategy for Feature Engineering
class FeatureEngineeringConcreteStrategy(FeatureEngineeringStrategy):
    def preprocess(self, X, y):
        """
        Preprocess the given DataFrame X and target vector y.

        This function performs the following steps:

        1. Encode the target column using a LabelEncoder.
        2. Remove all constant features (i.e., columns with only one unique value), excluding the target column.
        3. One-hot encode all other categorical columns using a OneHotEncoder.
        4. Convert the encoded data to a DataFrame.
        5. Add the target column to the DataFrame.
        6. Convert the DataFrame to integers.

        :param X: Input DataFrame
        :param y: Target vector
        :return: Preprocessed DataFrame
        """
        y = y.map({"e": 0, "p": 1})

        # Remove constant features (excluding target)
        constant_cols = [
            col for col in X.columns if X[col].nunique() == 1 and col != "class"
        ]
        X = X.drop(columns=constant_cols)

        # One-hot encode all other categorical columns
        cat_cols = [col for col in X.columns if X[col].dtype == "object"]
        encoder = OneHotEncoder(sparse_output=False)  # Ensure dense output
        data_encoded = encoder.fit_transform(X[cat_cols])

        # Convert encoded data to DataFrame
        df_encoded = pd.DataFrame(
            data_encoded, columns=encoder.get_feature_names_out(cat_cols)
        )

        # Add target column to the DataFrame
        df_encoded["class"] = y.values

        # Convert to integers
        df_encoded = df_encoded.astype("int")

        logger.info(
            "Removed all constant features and encoded all categorical columns."
        )
        return df_encoded

    def select_features(self, X, y, k=19):
        """
        Selects the top k features from the given DataFrame X and target vector y.

        This function uses SelectKBest with the f_classif scoring function to select the top k features.

        :param X: Input DataFrame
        :param y: Target vector
        :param k: Number of features to select (default is 19)
        :return: DataFrame with the top k features and the target column
        """
        selector = SelectKBest(score_func=f_classif, k=k)

        # Drop target column for feature selection
        X_features = X.drop(columns=["class"])

        # Fit selector and transform features
        X_new = selector.fit_transform(X_features, y)

        # Extract selected feature names
        selected_feature_names = X_features.columns[selector.get_support()]

        # Create DataFrame with selected features and target
        final_df = pd.DataFrame(X_new, columns=selected_feature_names)
        final_df["class"] = y.values

        logger.info("Applied feature selection using SelectKBest.")
        return final_df

    def remove_outliers(self, X):
        """
        Removes outliers from the given DataFrame X using the IQR method for numerical columns.

        The IQR (Interquartile Range) method calculates the first quartile (Q1) and the third quartile (Q3)
        of the numerical data, then removes all points more than 1.5 times the interquartile range (IQR)
        away from the quartiles.

        :param X: Input DataFrame
        :return: DataFrame with outliers removed from numerical columns only
        """
        # Select only numerical columns
        num_cols = X.select_dtypes(include=["number"])

        if num_cols.empty:
            # If no numerical columns, return X as is
            return X

        # Calculate Q1, Q3, and IQR for numerical columns
        Q1 = num_cols.quantile(0.25)
        Q3 = num_cols.quantile(0.75)
        IQR = Q3 - Q1

        # Define bounds for outlier removal
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR

        # Filter out outliers only for numerical columns
        num_cols_filtered = num_cols[
            ~((num_cols < lower_bound) | (num_cols > upper_bound)).any(axis=1)
        ]

        # Combine with non-numerical columns (which are unaffected by outlier removal)
        return X.loc[num_cols_filtered.index]


# Context for Data Transformation
class DataTransformation:
    def __init__(self, data_transformation_config: DataTransformationConfig):
        """
        Initializes the DataTransformation object.

        :param data_transformation_config: DataTransformationConfig object with
            configuration for data transformation.
        """
        self.config = data_transformation_config

        # Load the data
        self.df = pd.read_csv(self.config.data_path)
        logger.info("Data CSV loaded")

        # Define strategies
        self.eda_strategy = EDAConcreteStrategy()
        self.feature_engineering_strategy = FeatureEngineeringConcreteStrategy()

    def transform_data(self):
        """
        Transforms the data using the following steps:
        1. Perform exploratory data analysis
        2. Separate features and target variable
        3. Preprocess the data
        4. Remove outliers
        5. Apply feature selection
        6. Save the transformed data to a CSV file.

        :return: None
        """
        df = self.df

        # Perform EDA
        self.eda_strategy.execute(df)

        # Separate features and target variable
        X = df.drop(columns=["class"])
        y = df["class"]

        # Remove outliers for numeric features
        X = self.feature_engineering_strategy.remove_outliers(X)

        # Preprocess the data
        preprocessed_df = self.feature_engineering_strategy.preprocess(X, y)

        # Apply feature selection
        df_transformed = self.feature_engineering_strategy.select_features(
            preprocessed_df, preprocessed_df["class"], k=19
        )

        # Save the transformed data
        df_transformed.to_csv(self.config.transformed_data, index=False)
        logger.info(f"Saved transformed data to the directory - {self.config.root_dir}")
