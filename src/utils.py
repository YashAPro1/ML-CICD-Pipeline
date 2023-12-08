import os
import sys # for exceptional handling
import pandas as ps
import numpy as np
import dill
from src.exceptions import CustomException
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import r2_score
import pickle

def save_model(file_path,obj):
    try:
        directory_path = os.path.dirname(file_path)
        os.makedirs(directory_path,exist_ok =True)
        with open(file_path,"wb") as file_obj:
            dill.dump(obj,file_obj)
    except Exception as e:
        raise CustomException(e,sys)
    

def evaluate_models(X_train,y_train,X_test,y_test,models,params):
    try:
        report = {}
        for i in range(len(list(models))):
            model = list(models.values())[i]
            param = params[list(models.keys())[i]]
            grid_cv = GridSearchCV(model,param,cv = 3)
            grid_cv.fit(X_train,y_train)
            model.set_params(**grid_cv.best_params_)
            model.fit(X_train,y_train)
            y_train_pred = model.predict(X_train)

            y_test_pred = model.predict(X_test)

            train_model_score = r2_score(y_train, y_train_pred)

            test_model_score = r2_score(y_test, y_test_pred)

            report[list(models.keys())[i]] = test_model_score
            return report
    except Exception as e:
        raise CustomException(e,sys)


def load_object(file_path):
    try:
        with open(file_path, "rb") as file_obj:
            return pickle.load(file_obj)

    except Exception as e:
        raise CustomException(e, sys)