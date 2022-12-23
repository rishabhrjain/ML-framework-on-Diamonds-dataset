import os
import joblib


def save_pkl(obj, dir, filename):
        if not os.path.exists(dir):
            os.makedirs(dir, exist_ok=True)
        
        joblib.dump(obj, filename=dir + f'/{filename}')
        
def load_pkl( dir,  filename):
    filepath = dir + f'/{filename}'
    
    
    if not os.path.isfile(filepath):
        print(f'File not found - {filepath}')
        raise FileNotFoundError
        
    obj = joblib.load(filepath)
    return obj