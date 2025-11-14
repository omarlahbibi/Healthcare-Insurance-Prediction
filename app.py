from flask import Flask, request, render_template
import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler
from src.pipeline.predict_pipeline import CustomData, PredictPipeline

application = Flask(__name__)

app = application

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predictdata', methods=['GET', 'POST'])
def predict_datapoint():
    if request.method == 'GET':
        return render_template('home.html')
    else:
        data = CustomData(
            id=request.form.get('id'),
            Gender=request.form.get('Gender'),
            Age=request.form.get('Age'),
            Driving_Licence=request.form.get('Driving_Licence'),
            Region_Code=float(request.form.get('Region_Code')),
            Previously_Insured=request.form.get('Previously_Insured'),
            Vehicle_Age=request.form.get('Vehicle_Age'),
            Vehicle_Damage=request.form.get('Vehicle_Damage'),
            Annual_Premium=float(request.form.get('Annual_Premium')),
            Policy_Sales_Channel=float(request.form.get('Policy_Sales_Channel')),
            Vintage=request.form.get('Vintage')
        )
        pred_df = data.get_data_as_dataframe()
        print(pred_df)

        predict_pipeline = PredictPipeline()
        res = predict_pipeline.predict(pred_df)
        return render_template('home.html', results=res[0])
    

if __name__=="__main__":
    app.run(host="0.0.0.0", debug=True)