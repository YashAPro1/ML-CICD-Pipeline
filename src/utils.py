import os
import sys # for exceptional handling
import pandas as ps
import numpy as np
import dill
from src.exceptions import CustomException

def save_model(file_path,obj):
    try:
        directory_path = os.path.dirname(file_path)
        os.makedirs(directory_path,exist_ok =True)
        with open(file_path,"wb") as file_obj:
            dill.dump(obj,file_obj)
    except Exception as e:
        raise CustomException(e,sys)
    