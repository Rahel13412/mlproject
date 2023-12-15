import os 
import sys 
from dataclasses import dataclass
import dill
from catboost import CatBoostRegressor
from sklearn.ensemble import (
    AdaBoostRegressor,
    GradientBoostingRegressor,
    RandomForestRegressor
)
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor  
from sklearn.tree import DecisionTreeRegressor
from  xgboost import XGBRegressor 
from sklearn.metrics import r2_score
from src.exception import CustomException
from src.logger import logging 


def save_object(file_path, obj):
    try:
        dir_path = os.path.dirname(file_path)

        # Check if the directory already exists
        if not os.path.exists(dir_path):
            os.makedirs(dir_path)

        with open(file_path, 'wb') as file_obj:
            dill.dump(obj, file_obj)
    except Exception as e:
        raise CustomException(e, sys)
    


def evaluate_model(X_train,y_train,X_test,y_test,models):
    try:
        
        report = {}

        for i in range(len(list(models))):
            model = list(models.values())[i]

            model.fit(X_train,y_train)

            y_train_pred = model.predict(X_train)
            y_test_pred = model.predict(X_test)

            train_model_score = round(r2_score(y_train,y_train_pred),2)
            test_model_score = round(r2_score(y_test,y_test_pred),2)

            report[list(models.keys())[i]] = test_model_score

        return report
    
    except Exception as e:
        raise   CustomException(e, sys)





@dataclass
class ModelTrainerConfig:
    trainer_model_file_path = os.path.join('artifacts','model.pkl')


class ModelTrainer():
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()

    def initiate_model_training(self,train_array,test_array):
        try:
            logging.info('split training and test input data.')
            X_train,y_train,X_test,y_test = (train_array[:,:-1],
                                             train_array[:,-1],
                                             test_array[:,:-1],
                                             test_array[:,-1])
            
            models ={
                "Random Forest":RandomForestRegressor(),
                "Decesion Tree": DecisionTreeRegressor(),
                "Gradient Boosting" : GradientBoostingRegressor(),
                "Linear Regression" : LinearRegression(),
                "K-Neighbors Regressor" : KNeighborsRegressor(),
                "XGB Regressor": XGBRegressor(),
                "CatBoosting Regressor": CatBoostRegressor(verbose=False),
                "AdaBoost Regressor" : AdaBoostRegressor(),
                }
            
            
            model_report: dict=evaluate_model(X_train= X_train,y_train=y_train,
                                              X_test= X_test,y_test=y_test,models=models)
            

            # to get best model score from dict 
            best_model_score = max(sorted(model_report.values()))

            # to get best model name from dict 
            best_model_name = list(model_report.keys())[
                list(model_report.values()).index(best_model_score)
            ]

            best_model = models[best_model_name]


            if best_model_score <0.6:
                raise CustomException('No best model found ')
            
            logging.info('Best model found on both training and testing dataset.')

            save_object(
                file_path=self.model_trainer_config.trainer_model_file_path,
                obj=best_model

            )

            predicted = best_model.predict(X_test)

            score = r2_score(y_test,predicted)
            return score
        



        except Exception as e:
            raise CustomException(e,sys)
        

