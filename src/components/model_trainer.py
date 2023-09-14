import pandas as pd
import numpy as np
from src.logger import logging
from src.exception import CustomException

from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression

from src.utils import save_object
from src.utils import evaluate_model

import os
import sys

from dataclasses import dataclass

@dataclass
class ModelTrainerConfig:
    trained_model_file_path = os.path.join('artifacts' , 'model.pkl')

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()

    def initiate_model_training(self , train_array , test_array):

        try:
            logging.info('Splitting Dependent and Independent variables from train and test data')
            X_train , y_train , X_test , y_test = (
                train_array[: , :-1],
                train_array[: , -1],
                test_array[: , :-1],
                test_array[: , -1]
            )

            #  Automate model training process
            models = {
                'Random Forest' : RandomForestClassifier(),
                'Decision Tree' : DecisionTreeClassifier(),
                'svc' : SVC(),
                'logistic regression' : LogisticRegression()
            }

            model_report : dict = evaluate_model(X_train , y_train , X_test , y_test , models)

            print("\n----------------------------------------------------------------------------------------------")
            logging.info(f'Model Report : {model_report}')

            #  To get the best model score
            best_model_score = max(sorted(model_report.values()))

            best_model_name = list(model_report.keys())[list(model_report.values()).index(best_model_score)]

            best_model = models[best_model_name]

            print(f"Best Model Found , Name : {best_model_name} , accuracy-score : {best_model_score}")
            print("\n--------------------------------------------------------------------------------------")
            logging.info(f"Best Model Found , Name : {best_model_name} , accuracy-score : {best_model_score} ")


            # saving the model

            save_object(
                file_path = self.model_trainer_config.trained_model_file_path,
                obj = best_model
            )

            



        except Exception as e:
            raise CustomException(e,sys)

