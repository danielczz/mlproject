import sys
from dataclasses import dataclass

import numpy as np 
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder,StandardScaler

from src.exception import CustomException
from src.logger import logging
import os

from src.utils import save_object

@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path = os.path.join('artifacts', "proprocessor.pkl")

class DataTransformation:
    def __init__(self):
        self.data_transformation_config = DataTransformationConfig()

    def get_data_transformer_object(self, raw_path, target_column_name):
        '''
        This function is responsible for data transformation.
        It will return a prepocessor object with modified columns (Normalized and Encoded as needed).
        '''
        try:
            raw_df = pd.read_csv(raw_path)
            raw_df = raw_df.drop(columns = target_column_name, axis=1)
            
            numerical_columns = [feature for feature in raw_df.columns if raw_df[feature].dtype != "object"]
            categorical_columns = [feature for feature in raw_df.columns if raw_df[feature].dtype == "object"]
            
            logging.info(f"Categorical columns detected: {categorical_columns}")
            logging.info(f"Numerical columns detected: {numerical_columns}")

            num_pipeline = Pipeline(
                steps=[
                ("imputer",SimpleImputer(strategy = "median")),
                ("scaler",StandardScaler(with_mean = False))
                ]
            )

            cat_pipeline = Pipeline(
                steps=[
                ("imputer",SimpleImputer(strategy = "most_frequent")),
                ("one_hot_encoder",OneHotEncoder()),
                ("scaler",StandardScaler(with_mean = False))
                ]
            )

            preprocessor = ColumnTransformer(
                [
                ("num_pipeline", num_pipeline, numerical_columns),
                ("cat_pipelines", cat_pipeline, categorical_columns)
                ]
            )

            return preprocessor
        
        except Exception as e:
            raise CustomException(e,sys)
        
    def initiate_data_transformation(self, train_path, test_path, raw_path):

        try:
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)

            logging.info("Read train and test data completed")
            logging.info("Start building preprocessing object")
            
            target_column_name = "math_score"
            
            preprocessing_obj = self.get_data_transformer_object(raw_path, target_column_name)

            input_feature_train_df = train_df.drop(columns = [target_column_name], axis = 1)
            target_feature_train_df = train_df[target_column_name]

            input_feature_test_df = test_df.drop(columns = [target_column_name], axis=1)
            target_feature_test_df = test_df[target_column_name]

            logging.info(
                f"Task to apply preprocessor object on train and test DataFrame started"
            )

            '''This part will apply data transformation for train and test data according to the steps defined in transformation function'''
            input_feature_train_arr = preprocessing_obj.fit_transform(input_feature_train_df)
            input_feature_test_arr = preprocessing_obj.transform(input_feature_test_df)

            train_arr = np.c_[input_feature_train_arr, np.array(target_feature_train_df)]
            test_arr  = np.c_[input_feature_test_arr, np.array(target_feature_test_df)]

            logging.info(
                f"Task to apply preprocessor object on train and test DataFrame completed"
            )
            
            save_object(
                file_path = self.data_transformation_config.preprocessor_obj_file_path,
                obj = preprocessing_obj
            )

            return (
                train_arr,
                test_arr,
                self.data_transformation_config.preprocessor_obj_file_path,
            )
            
        except Exception as e:
            raise CustomException(e,sys)