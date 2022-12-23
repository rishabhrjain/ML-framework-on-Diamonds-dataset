
import os
import joblib

import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import OrdinalEncoder
from sklearn.metrics import classification_report, accuracy_score
from sklearn.ensemble import RandomForestClassifier

from data_processor import DataProcessor
from model import Model

DATA_PATH = './data/diamonds.csv'
DATA_DIR = './data'
RANDOM_STATE = 40

if __name__ == "__main__":

    print("Reading data")
    df = pd.read_csv(DATA_PATH)

    num_cols = ['carat', 'depth', 'table', 'price', 'x', 'y', 'z']
    binary_cols = []
    categorical_cols = ['color', 'clarity']

    target_col = ['cut']
    print("processing data")
    data_processor = DataProcessor(num_cols, binary_cols, categorical_cols, target_col, mode = 'train')

    train_test_split_args = {'test_size': 0.2, 'shuffle': True, 'random_state': RANDOM_STATE}
    train, test = data_processor.split_data(df, **train_test_split_args)

    print("processing training data and fitting encoders ..")
    X_train, y_train = data_processor.process_data(train)

    print("Processing test data ..")
    data_processor.mode = 'test'
    X_test, y_test = data_processor.process_data(test)

    print("Training model ..")
    clf = Model()
    clf.model = RandomForestClassifier()

    param_grid = {'n_estimators': [70, 100], 'max_depth': [8, 10]}

    gc_rf = clf.grid_search_fit(X_train, y_train.values.ravel(), param_grid)

    print("Model performance on test data:\n")
    y_pred = gc_rf.predict(X_test)

    X_test['target_pred'] = y_pred
    X_test.to_csv(DATA_DIR + '/predictions.csv', index=False)

    print(classification_report(y_test, y_pred))




