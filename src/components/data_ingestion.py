import os
import sys
from src.exceptions import CustomException
from src.logger import logging
import pandas as pd
from sklearn.model_selection import train_test_split
from src.components.data_transformation import DataTransformation, DataTransformationConfig
from dataclasses import dataclass

@dataclass
class DataIngestionConf:
    train_data_path = os.path.join('artifacts','train.csv')
    test_data_path = os.path.join('artifacts','test.csv')
    raw_data_path = os.path.join('artifacts','raw.csv')

class DataIngestion:
    def __init__(self):
        self.ingestion_conf = DataIngestionConf()
    
    def initiate_data(self):
        logging.info("We have started the data ingestion part now...")
        try:
            df = pd.read_csv("C:/Users/Yashkumar Dubey/Documents/Desktop1/youtube/ML CICD Pipe/notebooks/data/data.csv")
            logging.info("Reaing the Dataset as Dataframe...")
            os.makedirs(os.path.dirname(self.ingestion_conf.train_data_path),exist_ok=True)
            df.to_csv(self.ingestion_conf.raw_data_path,index=False,header=True)
            logging.info("Train test split is initiated...")
            train_set,test_set = train_test_split(df,test_size=0.2,random_state=52)
            train_set.to_csv(self.ingestion_conf.train_data_path,index=False,header=True)
            test_set.to_csv(self.ingestion_conf.test_data_path,index=False,header=True)
            logging.info("Ingestion is completed")
            return(
                self.ingestion_conf.train_data_path,
                self.ingestion_conf.test_data_path

            )
        except Exception as e:
            raise CustomException(e,sys)
        

if __name__=="__main__":
    obj = DataIngestion()
    train_data,test_data = obj.initiate_data()
    transform = DataTransformation()
    transform.initiate_data_transformation(train_data,test_data)


