import os
import sys
from src.logger import logging
from src.exception import CustomException

import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer# Handling missing values
from sklearn.preprocessing import StandardScaler# feature scaling
from src.utils import save_object


from imblearn.combine import SMOTETomek




from dataclasses import dataclass

@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path = os.path.join('artifacts' , 'preprocessor.pkl')

class DataTransformation:
    def __init__(self):
        self.data_transformation_config = DataTransformationConfig()


    def get_data_transformer_object(self):
        try:
            # define the steps for the preprocessor pipeline
            imputer_step = ('imputer' , SimpleImputer(strategy='median'))
            scaler_step = ('scaler' , StandardScaler())

            preprocessor = Pipeline(
                steps = [
                    # nan_replacement_step,
                    imputer_step,
                    scaler_step
                ]
            )

            return preprocessor

        except Exception as e:
            raise CustomException(e,sys)


    def initiate_data_transformation(self , train_data_path , test_data_path):

        try:
            logging.info("Data transformation initiated")

            # reading the train and test data
            train_df = pd.read_csv(train_data_path)
            test_df = pd.read_csv(test_data_path)

            logging.info("Read test and train data completed")


            preprocessor = self.get_data_transformer_object()

            # training dataframe
            input_feature_train_df = train_df.drop('target' , axis=1)
            target_feature_train_df = train_df['target']

            # testing dataframe
            input_feature_test_df = test_df.drop('target' , axis=1)
            target_feature_test_df = test_df['target']

            transformed_input_train_feature = preprocessor.fit_transform(input_feature_train_df)
            transformed_input_test_feature = preprocessor.transform(input_feature_test_df)

            smt = SMOTETomek(sampling_strategy = "minority")

            input_feature_train_final , target_feature_train_final = smt.fit_resample(
                transformed_input_train_feature , target_feature_train_df
            )

            input_feature_test_final , target_feature_test_final = smt.fit_resample(
                transformed_input_test_feature , target_feature_test_df
            )

            train_arr = np.c_[input_feature_train_final , np.array(target_feature_train_final)]
            test_arr = np.c_[input_feature_test_final , np.array(target_feature_test_final)]

            save_object(
                file_path = self.data_transformation_config.preprocessor_obj_file_path , 
                obj = preprocessor
            )

            return (
                train_arr ,
                test_arr ,
                self.data_transformation_config.preprocessor_obj_file_path
            )
        

        except Exception as e:
            logging.info("Exception occured in data transformation")
            raise CustomException(e,sys)
