import os
import sklearn.linear_model as linear_model
import pandas as pd
import numpy as np
import models
import pickle

class Linear(models.Model):
    def __init__(self):
        self._configure("demo")
        
        self.lin_reg = linear_model.LinearRegression()
        
    
    def train(self, *args):
        df = self.df
        
        train_df = df[df['subset'] == 'train']
        
        train_imgs, values = self.load_data(train_df)
                
        train_imgs = np.array(train_imgs).reshape(len(train_imgs), (self.dim ** 2)*3)
        
        self.lin_reg.fit(train_imgs, values)
        
    def predict(self, df):
        imgs, _ = self.load_data(df)
        imgs = np.array(imgs).reshape(len(imgs), (self.dim ** 2)*3)
        
        pred_scores = self.lin_reg.predict(imgs)
        
        return pred_scores
            
        
    def load(self, version='default'):
        output_path = self._weight_path(version)
        if not os.path.exists(output_path):
            return
        
        with open(output_path + '/weights', 'rb') as f:
            self.lin_reg = pickle.load(f)

        
    def _save(self, output_path):
        weights = pickle.dumps(self.lin_reg)
        
        with open(output_path + '/weights', 'wb') as f:
            f.write(weights)
        
        
        
        