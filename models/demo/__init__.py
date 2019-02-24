import yaml
import os
import sklearn.linear_model as linear_model
import pandas as pd
import numpy as np
import models
import pickle

class LinearModel(models.Model):
    def __init__(self):
        stream = open("models/demo/config.yml", "r")
        config = yaml.load(stream)
        
        self.dim = int(config['dim'])
        self.img_class = config['img_class']
        self.dir_name = "demo"
        self.score_reg = linear_model.LinearRegression()
        self.std_reg = linear_model.LinearRegression()
        
    
    def train(self):
        df = pd.read_csv('processed_data/{}/df.csv'.format(self.img_class))
        
        train_df = df[df['subset'] == 'train']
        
        train_imgs, train_scores, train_std = self.load_data(train_df)
                
        train_imgs = np.array(train_imgs).reshape(len(train_imgs), (self.dim ** 2)*3)
        
        self.score_reg.fit(train_imgs, train_scores)
        self.std_reg.fit(train_imgs, train_scores)
        
    def predict(self, df):
        imgs, _ , _ = self.load_data(df)
        imgs = np.array(imgs).reshape(len(imgs), (self.dim ** 2)*3)
        
        pred_scores = self.score_reg.predict(imgs)
        pred_stds = self.std_reg.predict(imgs)
        
        return pred_scores, pred_stds
            
        
    def load(self, version='default'):
        output_path = self.model_data_path(version)
        if not os.path.exists(output_path):
            return
        
        with open(output_path + '/score_regression', 'rb') as f:
            self.score_reg = pickle.load(f)
            
        with open(output_path + '/std_regression', 'rb') as f:
            self.std_reg = pickle.load(f)

        
    def _save(self, output_path):
        score_s = pickle.dumps(self.score_reg)
        std_s = pickle.dumps(self.std_reg)
        
        with open(output_path + '/score_regression', 'wb') as f:
            f.write(score_s)
            
        with open(output_path + '/std_regression', 'wb') as f:
            f.write(std_s)
        
        
        
        
        

