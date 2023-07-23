# Loading and import of libraries and dependencies
import os
import sys
from src.exception import CustomException
from src.logger import logging
import pandas as pd

from sklearn.model_selection import train_test_split
from dataclasses import dataclass

from src.components.data_transformation import DataTransformation
from src.components.data_transformation import DataTransformationConfig

# Creating dataclass
'''
A data class is a special type of class that is designed to store data
'''
@dataclass
class DataIngestionConfig:
    '''Identifying initial path for train, test and raw data'''
    train_data_path: str = os.path.join('artifacts',"train.csv")
    test_data_path: str = os.path.join('artifacts',"test.csv")
    raw_data_path: str = os.path.join('artifacts',"data.csv")

class DataIngestion:
    def __init__(self):
        '''Initializing ingestion parameters including path'''
        self.ingestion_config = DataIngestionConfig()

    def initiate_data_ingestion(self):
        logging.info("Data ingestion task started")
        try:
            df = pd.read_csv('notebook/data/stud.csv')
            logging.info('Dataset loading to DataFrame completed')
            '''Identifying file directories to store data'''
            os.makedirs(os.path.dirname(self.ingestion_config.train_data_path),exist_ok = True)

            df.to_csv(self.ingestion_config.raw_data_path,index = False,header = True)

            logging.info("Data split task started")
            
            train_set,test_set = train_test_split(df, test_size = 0.2, random_state = 42)
            '''Train and Test splitted datasets stored on directory'''
            train_set.to_csv(self.ingestion_config.train_data_path, index=False, header=True)
            test_set.to_csv(self.ingestion_config.test_data_path, index=False, header=True)
            
            logging.info("Data split task completed")
            logging.info("Data ingestion task completed")

            return(
                self.ingestion_config.train_data_path,
                self.ingestion_config.test_data_path,
                self.ingestion_config.raw_data_path
            )
            
        except Exception as e:
            raise CustomException(e, sys)
        
if __name__=="__main__":
    '''Initializing Data Ingestion object to kick-off tasks.
    It will get as a result the path of train, test and raw data'''
    obj = DataIngestion()
    train_data, test_data, raw_data = obj.initiate_data_ingestion()
    
    '''Initializing Data Transformation to normalize and encode data as needed. 
    It will get as result train array, test array and a file on artifacts folder'''
    data_transformation = DataTransformation()
    train_arr,test_arr,_ = data_transformation.initiate_data_transformation(train_data, test_data, raw_data)
