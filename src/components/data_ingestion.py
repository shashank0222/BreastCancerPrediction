import os
import sys
from src.logger import logging
from src.exception import CustomException

import pandas as pd
from sklearn.model_selection import train_test_split

from dataclasses import dataclass

# initialize the data ingestion configuration
@dataclass
class DataIngestionConfig:
    train_data_path :str = os.path.join('artifacts' , 'train.csv')
    test_data_path :str = os.path.join('artifacts' , 'test.csv')
    raw_data_path :str = os.path.join('artifacts' , 'raw.csv')

# create the data ingestion class
class DataIngestion:
    def __init__(self):
        self.ingestion_config = DataIngestionConfig()

    
    def initiate_data_ingestion(self):

        logging.info("Data Ingestion method starts")

        try:
            df = pd.read_csv(os.path.join('notebooks/data' , 'gemstone.csv'))

            logging.info('Dataset read as pd DataFrame')

            # making the  artifacts directory
            os.makedirs(os.path.join(self.ingestion_config.raw_data_path) , exist_ok=True)

            # saving the raw dataset into artifacts
            df.to_csv(self.ingestion_config.raw_data_path , index=False)

            logging.info('Raw data is created')

            train_set , test_set = train_test_split(df , test_size=0.20 , random_state=42)


            # saving the train dataset into artifacts 
            train_set.to_csv(self.ingestion_config.train_data_path , index = False , header =True)

            # saving the test dataset into artifacts
            test_set.to_csv(self.ingestion_config.test_data_path , index = False , header =True)

            logging.info('Ingestion of data is completed')

            return (
                self.ingestion_config.train_data_path,
                self.ingestion_config.test_data_path
            )



        except Exception as e:
            logging.info('Exception occured at Data Ingestion Stage')
            raise CustomException(e,sys)
        
    