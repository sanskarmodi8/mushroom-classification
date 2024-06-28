from sklearn.preprocessing import OneHotEncoder
import pandas as pd
import mlflow

class Prediction:
    def __init__(self, input_data):
        self.input_data = input_data
    
    def transform(self):
        
        # expected cols by model
        expected_features_by_model = [
            "bruises_f", "bruises_t", "odor_f", "odor_n", "gill-spacing_w", "gill-size_b", "gill-size_n", "gill-color_b",
            "stalk-surface-above-ring_k", "stalk-surface-above-ring_s", "stalk-surface-below-ring_k", "stalk-surface-below-ring_s",
            "ring-type_l", "ring-type_p", "spore-print-color_h", "spore-print-color_k", "spore-print-color_n",
            "spore-print-color_w", "population_v"
        ]
                
        # convert to dataframe
        self.df = pd.DataFrame([self.input_data])
        
        # Replace hyphens with underscores in column names
        self.df.columns = self.df.columns.str.replace('-', '_')
        
        cols = [col for col in self.df.columns]
        
        # one hot encode
        encoder = OneHotEncoder(sparse_output=False)
        data_encoded = encoder.fit_transform(self.df)
        
        # Convert encoded data to DataFrame
        df_encoded = pd.DataFrame(data_encoded, columns=encoder.get_feature_names_out(cols))
        self.df = df_encoded.astype("int")
        
        # Add missing columns from expected_features_by_model with values of 0
        missing_columns = list(set(expected_features_by_model) - set(self.df.columns))
        
        for col in missing_columns:
            self.df[col] = 0
        
        # Filter DataFrame to keep only columns present in expected_features_by_model
        self.df = self.df[expected_features_by_model]
        
    def classify(self):
        
        # transform the data 
        self.transform()
        
        # load model from mlflow 
        logged_model = 'runs:/6ffbead0cec3476c8b6f1df39657a511/model'
        model = mlflow.pyfunc.load_model(logged_model)
        
        # return the result 
        return model.predict(self.df).tolist()