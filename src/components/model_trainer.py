# Evaluation and Deployment of ML Model
import os
import sys
from dataclasses import dataclass

from sklearn.ensemble import (
    AdaBoostRegressor,
    GradientBoostingRegressor,
    RandomForestRegressor,
)
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor

from src.exception import CustomException
from src.logger import logging

from src.utils import save_object,evaluate_models

@dataclass
class ModelTrainerConfig:
    # This time we need to identify the path where model will be stored. In this case, artifacts folder.
    trained_model_file_path = os.path.join("artifacts", "model.pkl")

class ModelTrainer:
    def __init__(self):
        # Everytime we create a class, we need to initialize the class with proper paramaters to use.
        self.model_trainer_config = ModelTrainerConfig()

    def initiate_model_trainer(self, train_array, test_array):
        try:
            logging.info("Split of train and test arrays started")
            X_train, y_train, X_test, y_test=(
                train_array[:,:-1],
                train_array[:,-1],
                test_array[:,:-1],
                test_array[:,-1]
            )
            logging.info("Split of train and test arrays completed")
                       
            models = {
                "Random Forest": RandomForestRegressor(),
                "Decision Tree": DecisionTreeRegressor(),
                "Gradient Boosting": GradientBoostingRegressor(),
                "Linear Regression": LinearRegression(),
                "K-Neighbors Classifier": KNeighborsRegressor(),
                "XGBClassifier": XGBRegressor(),
                "AdaBoost Classifier": AdaBoostRegressor(),
            }

            model_report: dict  = evaluate_models(X_train = X_train, y_train = y_train, X_test = X_test, y_test = y_test, models = models)
            
            ## To get best model score and model name from dict
            best_model_score    = max(sorted(model_report.values()))
            best_model_name     = list(model_report.keys())[list(model_report.values()).index(best_model_score)]
            best_model          = models[best_model_name]

            if best_model_score < 0.6:
                raise CustomException("No best model found. Performance is under 60%")
            
            logging.info(f"Best model found in training and test data set")
            logging.info(best_model)
            
            logging.info(f"Saving model on artifacts started")
            save_object(
                file_path   = self.model_trainer_config.trained_model_file_path,
                obj         = best_model
            )
            logging.info(f"Saving model on artifacts completed")
 
            predicted       = best_model.predict(X_test)

            r2_square       = r2_score(y_test, predicted)
            
            return r2_square, best_model

        except Exception as e:
            raise CustomException(e,sys)