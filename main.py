
import os
import joblib

import logging
logging.basicConfig(level = logging.DEBUG)

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

    logging.info("Reading data")
    df = pd.read_csv(DATA_PATH)

    train_test_split_args = {'test_size': 0.2, 'shuffle': True, 'random_state': RANDOM_STATE}
    train, test = data_processor.split_data(df, **train_test_split_args)

    num_cols = ['carat', 'depth', 'table', 'price', 'x', 'y', 'z']
    binary_cols = []
    categorical_cols = ['color', 'clarity']
    target_col = ['cut']


    logging.info("Initiaize the data processor and set to training mode")
    data_processor = DataProcessor(num_cols, binary_cols, categorical_cols, target_col, mode = 'train')

    logging.info("processing training data and fitting encoders ..")
    X_train, y_train = data_processor.process_data(train)

    logging.info("Processing test data by setting to test mode ..")
    data_processor.mode = 'test'
    X_test, y_test = data_processor.process_data(test)

    logging.info("Training model ..")
    clf = Model()
    clf.model = RandomForestClassifier()

    param_grid = {'n_estimators': [70, 100], 'max_depth': [8, 10]}

    gc_rf = clf.grid_search_fit(X_train, y_train.values.ravel(), param_grid)

    logging.info("Model performance on test data:\n")
    y_pred = gc_rf.predict(X_test)

    X_test['target_pred'] = y_pred
    X_test.to_csv(DATA_DIR + '/predictions.csv', index=False)

    logging.info(classification_report(y_test, y_pred))




