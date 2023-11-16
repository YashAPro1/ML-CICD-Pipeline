import os
import sys
from dataclasses import dataclass

from catboost import CatBoostRegressor
from sklearn.ensemble import (
    AdaBoostRegressor,
    GradientBoostingRegressor,
    RandomForestRegressor,
)
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
# from xgboost import XGBRegressor

from src.exceptions import CustomException
from src.logger import logging

from src.utils import save_model,evaluate_models

@dataclass
class ModelTrainerConf:
    trained_model_file_path=os.path.join("artifacts","model.pkl")

class Trained_Model:
    def __init__(self) -> None:
        self.model_trainer_conf = ModelTrainerConf()

    def initialise_training(self,train_data,test_data):
        try:
            logging.info("Creating and Spliting the data into my train and test data")
            X_train,y_train,X_test,y_test = (
                train_data[:,:-1],
                train_data[:,-1],
                test_data[:,:-1],
                test_data[:,-1]
            )
            logging.info("Creating Dictionary of models")
            models = {
                "Decision Tree": DecisionTreeRegressor(),
                "AdaBoost Regressor": AdaBoostRegressor(),
                "Gradient Boosting": GradientBoostingRegressor(),
                "Random Forest": RandomForestRegressor(),
                # "XGBRegressor": XGBRegressor(),
                "Linear Regression": LinearRegression(),
                "CatBoosting Regressor": CatBoostRegressor(verbose=False),
                }
            
            logging.info("Hypertuning the models")

            params = {
                "Decision Tree": {
                        'criterion':['squared_error', 'friedman_mse', 'absolute_error', 'poisson'],
                        # 'splitter':['best','random'],
                        # 'max_features':['sqrt','log2'],
                    },
                    "AdaBoost Regressor":{
                        'learning_rate':[.1,.01,0.5,.001],
                        # 'loss':['linear','square','exponential'],
                        'n_estimators': [8,16,32,64,128,256]
                    },
                    "Gradient Boosting":{
                        # 'loss':['squared_error', 'huber', 'absolute_error', 'quantile'],
                        'learning_rate':[.1,.01,.05,.001],
                        'subsample':[0.6,0.7,0.75,0.8,0.85,0.9],
                        'n_estimators': [8,16,32,64,128,256]
                    },
                    "Random Forest":{
                        'n_estimators': [8,16,32,64,128,256]
                    },
                    # "XGBRegressor":{
                    #     'learning_rate':[.1,.01,.05,.001],
                    #     'n_estimators': [8,16,32,64,128,256]
                    # },
                    "Linear Regression":{},
                    "CatBoosting Regressor":{
                        'depth': [6,8,10],
                        'learning_rate': [0.01, 0.05, 0.1],
                        'iterations': [30, 50, 100]
                    },
                    
            }

            report:dict=evaluate_models(X_train=X_train,y_train=y_train,X_test=X_test,y_test=y_test,
                                                models=models,params=params)
            
            best_model_value = max(sorted(report.values()))
            best_model_name = list(report.keys())[list(report.values()).index(best_model_value)]
            our_model = models[best_model_name]

            if best_model_value<0.6:
                raise CustomException("No best model found")
            logging.info(f"Best found model on both training and testing dataset")

            save_model(
                file_path=self.model_trainer_conf.trained_model_file_path,
                obj=our_model
            )

            predicted=our_model.predict(X_test)

            r2_square = r2_score(y_test, predicted)
            return r2_square
        except Exception as e:
            raise CustomException(e,sys)
        