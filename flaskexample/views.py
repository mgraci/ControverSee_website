from flask import render_template
from flaskexample import app

import pandas as pd
from flask import request

from  flaskexample.advanced_model_a import print_prediction_string
from  flaskexample.advanced_model_a import main

import urllib.parse

'''
import base64
'''
@app.route('/')
@app.route('/index')
@app.route('/input', methods = ['GET', 'POST'])
def input():
    return render_template("input.html")

@app.route('/example', methods = ['GET', 'POST'])
def example():
    return render_template("example.html")


@app.route('/output', methods = ['GET', 'POST'])
def output():
    words = request.values.get('text_entry')

    prediction = print_prediction_string(words)
    string_pred, important_words_df  = main(words)
    query_results= important_words_df
    births = []
    for i in range(0,query_results.shape[0]):
        births.append(dict(index=query_results.iloc[i]['Word'], attendant=query_results.iloc[i]['Predictive_Value']))
    
    return render_template("output.html", the_result = string_pred, births = births) #df = important_words_df )


