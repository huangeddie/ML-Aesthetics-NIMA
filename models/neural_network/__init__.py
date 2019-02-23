import yaml
import os
import sklearn.linear_model as linear_model
import pandas as pd
import numpy as np
import models
import pickle

import keras

class NeuralNetwork(models.Model):
    def __init__(self):
        stream = open("models/neural_network/config.yml", "r")
        config = yaml.load(stream)
        
        self.dim = int(config['dim'])
        self.img_class = config['img_class']
        self.dir_name = "neural_network"
        
        self.neural_network = keras.models.Sequential([
            keras.layers.Dense(self.dim, input_shape=((self.dim ** 2)*3,)),
            keras.layers.Activation('relu'),
            keras.layers.Dense(2),
        ])
        
    
    def train(self):
        df = pd.read_csv('processed_data/{}/df.csv'.format(self.img_class))
        

        train_df = df[df['subset'] == 'train']
        
        train_imgs, train_scores, train_std = self.load_data(train_df)
        
        train_imgs = np.array(train_imgs).reshape(len(train_imgs), (self.dim ** 2)*3)
        
        self.neural_network.compile(optimizer='adam',
              loss='mse',
              metrics=['mse'])
        
        values = np.array([train_scores, train_std]).T
        
        self.neural_network.fit(train_imgs, values, epochs=10, batch_size=32)
        
        
    def predict(self, imgs):
        imgs = np.array(imgs).reshape(len(imgs), (self.dim ** 2)*3)
        
        pred_values = self.neural_network.predict(imgs)
        
        return pred_values[:, 0], pred_values[:, 1]
            
        
    def load(self, version='default'):
        output_path = self.model_data_path(version)
        if not os.path.exists(output_path):
            return
        
        self.neural_network = keras.models.load_model(output_path + '/neural_network.h5')


        
    def _save(self, output_path):
        self.neural_network.save(output_path + '/neural_network.h5')
        
        
        
        
        
        


