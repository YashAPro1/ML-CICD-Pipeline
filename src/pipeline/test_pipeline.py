import sys
import os
import numpy as np
import pandas as pd
from src.exceptions import CustomException
from src.utils import load_object



class PredictPipeline:
    def __init__(self) -> None:
        pass
    def predict(self,features):
        try:
            model_path = os.path.join("artifacts","model.pkl")
            preprocessor_path = os.path.join("artifacts","proprocessor.pkl")
            model = load_object(file_path=model_path)
            preprocessor = load_object(file_path=preprocessor_path)
            data_scaled = preprocessor.transform(features)
            preds = model.predict(data_scaled)
            return preds

        except Exception as e:
            raise CustomException(e,sys)
        
class CustomData:
    def __init__(self,ram,weight,touchscreen,ips,ppi,hdd,ssd,company,typename,cpubrand,gpubrand,os) -> None:
        self.ram = ram
        self.weight = weight
        self.touchscreen = touchscreen
        self.ips = ips
        self.ppi = ppi
        self.hdd = hdd
        self.ssd = ssd
        self.company = company
        self.typename = typename
        self.cpubrand = cpubrand
        self.gpubrand = gpubrand
        self.os = os


    def get_data_as_dataframe(self):
        try:
            input_dict = {
                "Company":[self.company],
                "TypeName":[self.typename],
                "Ram":[self.ram],
                "Weight":[self.weight],
                "TouchScreen":[self.touchscreen],
                "Ips":[self.ips],
                "Ppi":[self.ppi],
                "Cpu_brand":[self.cpubrand],
                "HDD":[self.hdd],
                "SSD":[self.ssd],
                "Gpu_brand":[self.gpubrand],
                "Os":[self.os]

            }

            return pd.DataFrame(input_dict)
        except Exception as e:
            raise CustomException(e,sys)