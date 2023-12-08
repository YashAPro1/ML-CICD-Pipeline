from flask import Flask,request,render_template
import numpy as np
import pandas as pd
from src.pipeline.test_pipeline import PredictPipeline,CustomData
from src.exceptions import CustomException
import sys
application=Flask(__name__)

app=application

## Route for a home page

@app.route('/')
def index():
    return render_template('index.html') 

@app.route('/home')
def home():
    return render_template('home.html') 

@app.route('/predict',methods=['GET','POST'])
def predictdata():
    if request.method == 'GET':
        return render_template("index.html")
    else:
        try:
            print(request.form)
            data = CustomData(
                ram = int(request.form.get("ram")),
                weight = float(request.form.get("weight")),
                touchscreen = int(request.form.get("touchscreen")),
                ips =int(request.form.get("ips")) ,
                ppi = float(request.form.get("ppi")),
                hdd = int(request.form.get("hdd")),
                ssd = int(request.form.get("ssd")),
                company = request.form.get("company"),
                typename =  request.form.get("typename"),
                cpubrand =  request.form.get("cpubrand"),
                gpubrand =  request.form.get("gpu_brand"),
                os =  request.form.get("os"),
            )
            pred_df = data.get_data_as_dataframe()
            print(pred_df)
            pred_pipe = PredictPipeline()
            results = pred_pipe.predict(pred_df)
            print(results)
            return render_template("home.html",results = results[0])
        except Exception as e:
            raise CustomException(e,sys)


if __name__=="__main__":
    print("heee")
    app.debug = True
    app.run(host="127.0.0.1",port=5000)   
    print("Hi")