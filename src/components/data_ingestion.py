import os
import sys
from src.logger import logging
from src.exception import CustomException

import pandas as pd
from sklearn.model_selection import train_test_split

from dataclasses import dataclass

from pymongo.mongo_client import MongoClient

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
            uri = "mongodb+srv://shashank:shashank@cluster0.ylucjb4.mongodb.net/?retryWrites=true&w=majority"

            # Create a new client and connect to the server

            client = MongoClient(uri)            

            db = client['MLProjects']
            coll = db['breastCancer']

            cursor = coll.find({})

            data = list(cursor)
            df = pd.DataFrame(data)

            # drop the id column

            df = df.drop('_id' , axis=1)

            # converting the datatype into int

            df['target'] = df['target'].astype(int)



            logging.info('Dataset read as pd DataFrame')

            # making the  artifacts directory
            os.makedirs(os.path.dirname(self.ingestion_config.raw_data_path) , exist_ok=True)

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
        
    