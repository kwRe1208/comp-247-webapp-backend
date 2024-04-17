from flask import Flask
from flask import request
from flask import jsonify
from flask import render_template
import numpy as np
import os
import joblib
import pandas as pd
#disable warnings
import warnings
warnings.filterwarnings("ignore")

app = Flask(__name__)

_model_path = "models"
_model_extesnion = "pkl"
_model_dict ={
    'rf': 'rf_best_model',
    'svm': 'SVM_best_model',
    'nb': 'naivebayes_trained_model',
    'vote': 'voting_classifier_model',
    'log': 'best_logistic_regression_model'
}

#reorder the _test_data with below sequencce:
#['ROAD_CLASS' 'DISTRICT' 'LATITUDE' 'LONGITUDE' 'ACCLOC' 'TRAFFCTL'
# 'VISIBILITY' 'LIGHT' 'RDSFCOND' 'ACCLASS' 'INVTYPE' 'INVAGE' 'INJURY'
# 'PEDESTRIAN' 'CYCLIST' 'AUTOMOBILE' 'MOTORCYCLE' 'TRUCK' 'TRSN_CITY_VEH'
# 'EMERG_VEH' 'PASSENGER' 'SPEEDING' 'AG_DRIV' 'REDLIGHT' 'ALCOHOL'
# 'DISABILITY' 'HOOD_158']

_test_data = {
    "ROAD_CLASS": "Expressway",
    "DISTRICT": "Scarborough",
    "LATITUDE": 43.7839925902031,
    "LONGITUDE": -79.23168518007326,
    "ACCLOC": "At Intersection",
    "TRAFFCTL": "Pedestrian Crossover",
    "VISIBILITY": "Clear",
    "LIGHT": "Daylight",
    "RDSFCOND": "Dry",
    "INVTYPE": "Driver",
    "INVAGE": "35 to 39",
    "INJURY": "Fatal",
    "PEDESTRIAN": "Yes",
    "CYCLIST": "No",
    "AUTOMOBILE": "Yes",
    "MOTORCYCLE": "No",
    "TRUCK": "No",
    "TRSN_CITY_VEH": "No",
    "EMERG_VEH": "No",
    "PASSENGER": "No",
    "SPEEDING": "Yes",
    "AG_DRIV": "Yes",
    "REDLIGHT": "Yes",
    "ALCOHOL": "No",
    "DISABILITY": "No",
    "HOOD_158": "133"
}

_selected_cat_columns = ['LIGHT', 'INVAGE', 'RDSFCOND', 'DISTRICT', 
                         'ROAD_CLASS', 'TRAFFCTL', 'ACCLOC', 'VISIBILITY', 
                         'INVTYPE', 'PEDESTRIAN', 'CYCLIST', 'AUTOMOBILE', 
                         'MOTORCYCLE', 'TRUCK', 'TRSN_CITY_VEH', 'EMERG_VEH', 
                         'PASSENGER', 'SPEEDING', 'AG_DRIV', 'REDLIGHT', 'ALCOHOL', 
                         'DISABILITY', 'INJURY']

@app.route("/model/<model_name>", methods=["POST"])
def process_model(model_name):
    #get data from request body
    data = request.get_json()

    # make json data to dataframe
    _data_df = pd.DataFrame(data, index=[0])

    #load the pipline preprocessor
    preprocessor = joblib.load(r'C:\Users\kevin\OneDrive\桌面\Git\comp-247-webapp-backend\models\preprocessor.pkl')
    #Call pipline preprocessor to transform the data
    features = preprocessor.transform(_data_df)

    # Get the JSON data from the request
    # data = request.get_json()

    # transform model name to the correct model name
    model_name = _model_dict[model_name]
    # print(model_name)
    # Process the data using the specified model
    result = prediction(features, model_name)

    # Return the result as JSON
    
    return jsonify({
        'Model': model_name,
        'Result': labeling_result(result[0]),
    })

def prediction(data, model_name):
    # Load the model from the file
    try:
        #model_path = os.path.join(os.path.dirname(__file__), os.path.join(_model_path, f"{model_name}.{_model_extesnion}"))
        model_path = r'C:\Users\kevin\OneDrive\桌面\Git\comp-247-webapp-backend\models\{model_name}.{ext}'.format(model_name=model_name, ext=_model_extesnion)
        model = joblib.load(model_path)
        result = model.predict(data)
    except Exception as e:
        raise e
    # Make a prediction using the model
    
    return result

def labeling_result(result):
    if int(result) == 0:
        return "Non-Fatal"
    else:    
        return "Fatal"

@app.route("/")
def home():
    return render_template("index.html")


if __name__ == "__main__":
    app.run(debug=True, port=5000)