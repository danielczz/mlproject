from flask import Flask, request, render_template

import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler
from src.pipeline.predict_pipeline import CustomData, PredictPipeline
from src.logger import logging

# Creation of a Flask application
application = Flask(__name__)

app = application

# Creation for route for a "home" page and "predict_data" page
@app.route('/')
def index():
        return render_template('index.html')

@app.route('/predictdata',methods=['GET', 'POST'])
def predict_datapoint():
    if request.method == 'GET':
        return render_template('home.html')
    else:
        data = CustomData(
            gender                      = request.form.get('gender'),
            race_ethnicity              = request.form.get('ethnicity'),
            parental_level_of_education = request.form.get('parental_level_of_education'),
            lunch                       = request.form.get('lunch'),
            test_preparation_course     = request.form.get('test_preparation_course'),
            reading_score               = float(request.form.get('writing_score')),
            writing_score               = float(request.form.get('reading_score'))
        )
        
        # Creating DataFrame using Custom Data coming from predictdata website
        logging.info("Custom Data gathering from predictdata website is completed")
        
        features_df = data.get_data_as_data_frame()
        
        logging.info("Creation of DataFrame using Custom Data from predictdata website is completed")

        logging.info("Prediction using Custom Data started")

        predict_pipeline = PredictPipeline()

        # Using the DataFrame created with CustomData coming from website, is sent to the predict function and return a score.
        results = predict_pipeline.predict(features_df)

        logging.info("Prediction using Custom Data is completed")
        
        # This "POST" the result on the webpage using results as variable.
        return render_template('home.html', results = results[0])

if __name__=="__main__":
    app.run(host="0.0.0.0")  