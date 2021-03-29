"""Train and save the models"""
from pickle import dump
from os import path
from typing import Tuple

from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import cross_val_score
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

from numpy import ndarray
from pandas import read_csv

def save_model(model, model_name: str):
    """Saves sklearn model"""
    pickle_path = path.join('Models', f'{model_name}.pkl')
    with open(pickle_path, 'wb') as pickle_file:
        dump(model, pickle_file)

def fit_and_save_model(cls, model_name: str, X, y):
    model = cls()
    model.fit(X, y)
    save_model(model, model_name)

def get_values() -> Tuple[ndarray, ndarray]:
    df = read_csv('./heart.csv')
    X = df[['age', 'sex', 'fbs']].values
    y = df['target'].values
    return X, y

def main():
    X, y = get_values()
    models = [
        (SVC, 'SVM'),
        (SGDClassifier, 'SGD'),
        (DecisionTreeClassifier, 'DecisionTree')
    ]
    for model, model_name in models:
        print(f'Training model {model_name}')
        fit_and_save_model(model, model_name, X, y)

if __name__ == "__main__":
    main()
