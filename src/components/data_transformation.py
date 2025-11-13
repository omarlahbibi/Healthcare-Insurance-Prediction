import sys
import os

import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from dataclasses import dataclass

from src.exception import CustomException
from src.logger import logging
from src.utils import save_object


@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path = os.path.join('artifacts', 'preprocessor.pkl')

class DataTransformation:
    def __init__(self):
        self.data_transformation_config = DataTransformationConfig()

    def get_data_transformer_object(self):
        """
        This function is responsible for data transformation
        """

        try:
            numerical_columns = ['Age', 'Vintage', 'Annual_Premium']
            categorical_columns = ['Gender', 'Vehicle_Age', 'Vehicle_Damage']

            num_pipeline = Pipeline(
                steps=[("scaler", StandardScaler())]
            )

            cat_pipeline = Pipeline(
                steps=[("onehot", OneHotEncoder())]
            )

            logging.info(f"Numerical columns: {numerical_columns}")
            
            logging.info(f"Categorical columns: {categorical_columns}")

            preprocessor = ColumnTransformer(
                transformers=[
                    ("num_pip", num_pipeline, numerical_columns),
                    ("cat_pip", cat_pipeline, categorical_columns)]
                )
            
            return preprocessor
        except Exception as e:
            raise CustomException(e, sys)
        

    def initiate_data_transformation(self, train_path, test_path):

        try:
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)

            logging.info("Train test datasets are loaded")

            logging.info("Obtaining preprocessor object")

            preprocessing_obj = self.get_data_transformer_object()
            
            target_column = "Response"

            input_feature_train_df = train_df.drop(columns=[target_column], axis=1)
            target_feature_train_df = train_df[target_column]

            input_feature_test_df = test_df.drop(columns=[target_column], axis=1)
            target_feature_test_df = test_df[target_column]

            logging.info(f"Applying preprocessing OBJ on training and testing dataframes")

            input_feature_train_arr = preprocessing_obj.fit_transform(input_feature_train_df)
            input_feature_test_arr = preprocessing_obj.transform(input_feature_test_df)

            train_arr = np.c_[input_feature_train_arr, np.array(target_feature_train_df)]
            test_arr = np.c_[input_feature_test_arr, np.array(target_feature_test_df)]

            logging.info(f"Saved preprocessing OBJ")

            save_object(file_path=self.data_transformation_config.preprocessor_obj_file_path,
                        obj=preprocessing_obj
                        )

            return(train_arr, test_arr, self.data_transformation_config.preprocessor_obj_file_path)

        except Exception as e:
            raise CustomException(e, sys)