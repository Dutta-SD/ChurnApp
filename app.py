'''
Front End for Flask Churn prediction app.
Author : Sandip Dutta
'''
from numpy.core.fromnumeric import ndim
import config
from flask import Flask, request
import numpy as np
import xgboost
import flasgger
from flasgger import Swagger


# App and Globals
app = Flask (__name__)
Swagger(app)

# Load the model, xgboost model
pred_model = xgboost.Booster()
pred_model.load_model(config.XGB_MODEL_PATH)

# Build the ile
@app.route("/")
def home():
    '''
    Home Screen of the API
    '''
    return 'Welcome to Predicting Customer Churn'


@app.route('/predict', methods = ['GET'])
@flasgger.swag_from('prediction.yml')
def predict_churn():
    '''
    predicts Customer Churn
    '''
    cred_score = request.args.get('Credit_Score')

    values_vector = np.array([300, 90, 1, 1, 1, 1, 1, 1, 1, 1, 1]).reshape(-1, 1)
    values_vector = xgboost.DMatrix(values_vector)
    
    print(pred_model)
    pred = pred_model.predict(values_vector)
    return f"Credit Score is {round(pred.mean(), 2)}"


if __name__ == '__main__':
    app.run(debug = True)