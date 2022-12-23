from sklearn.model_selection import GridSearchCV
from utilities import save_pkl, load_pkl
import logging
logging.basicConfig(level = logging.DEBUG)


import pandas as pd

class Model():
    """
    Class to help with model training. 
    """
    def __init__(self, model=None):
        
        self.model = model
        self.grid_search_model = None
        self.MODEL_DIR = './model'
        self.DATA_DIR = './data'
    
    def fit(self, X, y):
        """
        Function that fits the model to training data and saves as .pkl file

        Args:
            X: Input features for the model.
            y: Target label to train on.

        """
        self.model.fit(X, y_)
        save_pkl(self.model, dir = self.MODEL_DIR, filename = 'model.pkl')
    
    
    def grid_search_fit(self, X, y, param_grid):
        """
        Function for model tuning using sklearn GridSearchCV. 

        Args:
            X: Input features for the model.
            y: Target label to train on.
            param_grid: Params for GridSearchCV

        Returns:
            The GRidSearchCV object

        """
        
        gc = GridSearchCV(self.model, param_grid=param_grid)
        gc.fit(X, y)
        self.grid_search_model = gc 
        save_pkl(self.model, dir = self.MODEL_DIR, filename = 'grid_search_model.pkl')
        
        return gc

