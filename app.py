# -*- coding: utf-8 -*-
"""This is Heroku Deployment Lecture"""
from os import getcwd
import pickle

from flask import Flask, request, render_template

path = getcwd()

def load_pickle_model(model_name):
    model = None
    with open(f'Models/{model_name}.pkl', 'rb') as pickle_file:
        model = pickle.load(pickle_file)
    return model

VALID_MODELS = ('SVM', 'SGD', 'DecisionTree')
MODELS = list(map(load_pickle_model, VALID_MODELS))

def get_predictions(age, sex, fbs, req_model):
    values = [[float(i) for i in (age, sex, fbs)]]

    index = -1
    try:
        index = VALID_MODELS.index(req_model)
    except ValueError:
        return 'Cannot Predict'

    model = MODELS[index]
    return model.predict(values)[0]

app = Flask(__name__)

@app.route('/')
def homepage():
    return render_template('home.html')

@app.route('/', methods=['POST', 'GET'])
def my_form_post():
    if request.method != 'POST':
        return render_template('home.html')
    age = request.form['age']
    sex = request.form['sex']
    fbs = request.form['fbs']
    req_model = request.form['req_model']

    target = get_predictions(age, sex, fbs, req_model)

    if isinstance(target, str):
        heart_disease = target
    elif target == 1:
        heart_disease = 'Customer has heart disease.'
    else:
        heart_disease = 'Customer does not have heart disease.'

    return render_template('home.html', target=target, heart_disease=heart_disease)

if __name__ == "__main__":
    app.run(debug=True)
