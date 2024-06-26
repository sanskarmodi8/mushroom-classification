from MushroomClassification.entity.config_entity import DataTransformationConfig
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.pipeline import Pipeline
import pandas as pd
from MushroomClassification import logger
import pickle

class DataTransformation:
    def __init__(self, data_transformation_config: DataTransformationConfig):
        self.config= data_transformation_config
        
        #load the data
        self.df = pd.read_csv(self.config.data_path)
        logger.info("Data csv loaded")
        
    def preprocess(self, X, y):
        # Encode target column
        y = y.map({"e": 0, "p": 1})

        # Remove constant features (excluding target)
        constant_cols = [col for col in X.columns if X[col].nunique() == 1 and col != "class"]
        X = X.drop(columns=constant_cols)

        # Onehot encode all other categorical columns
        cat_cols = [col for col in X.columns if X[col].dtype == "object"]
        encoder = OneHotEncoder(sparse_output=False)  # Ensure dense output
        data_encoded = encoder.fit_transform(X[cat_cols])

        # Convert encoded data to DataFrame
        df_encoded = pd.DataFrame(data_encoded, columns=encoder.get_feature_names_out(cat_cols))

        # Add target column to the DataFrame
        df_encoded["class"] = y.values

        # Convert to integers
        df_encoded = df_encoded.astype("int")
        
        logger.info("removed all constant features and encoded all categorical columns")

        return df_encoded
        
    
    def select_features(self, X, y, k=19):
        selector = SelectKBest(score_func=f_classif, k=k)
        
        # Drop target column for feature selection
        X_features = X.drop(columns=["class"])
        
        # Fit selector and transform features
        X_new = selector.fit_transform(X_features, X["class"])
        
        # Extract selected feature names
        selected_feature_names = X_features.columns[selector.get_support()]
        
        # Create DataFrame with selected features and target
        final_df = pd.DataFrame(X_new, columns=selected_feature_names)
        final_df["class"] = X["class"].values
        
        logger.info("Applied feature selection using SelectKBest")
        
        return final_df
        
    def transform_data(self):
        df = self.df
        preprocess_transformer = FunctionTransformer(lambda X: self.preprocess(X.drop(columns=["class"]), X["class"]))
        feature_select_transformer = FunctionTransformer(lambda X: self.select_features(X, X["class"], k=19))

        pipe = Pipeline([
            ("preprocess", preprocess_transformer),
            ("feature_selection", feature_select_transformer),
        ])
        
        df_transformed = pipe.fit_transform(df)
        df_transformed.to_csv(f"{self.config.transformed_data_dir}/df_transformed.csv", index=False)
        logger.info(f"Saved transformed data to the directory - {self.config.transformed_data_dir}")
    
