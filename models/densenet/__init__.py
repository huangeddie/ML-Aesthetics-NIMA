import yaml
import os
import sklearn.linear_model as linear_model
import pandas as pd
import numpy as np
import models
import pickle

import keras

from keras.applications.densenet import DenseNet121, preprocess_input
from keras.preprocessing import image
from keras_preprocessing.image import ImageDataGenerator
from keras.models import Model
from keras.layers import Dense, GlobalAveragePooling2D
from keras import backend as K
from keras.layers import Input

import pdb

class DenseNet(models.Model):
    def __init__(self):
        stream = open("models/densenet/config.yml", "r")
        config = yaml.load(stream)
        
        self.dim = int(config['dim'])
        self.img_class = config['img_class']
        self.dir_name = "densenet"
        
        # create the base pre-trained model
        input_tensor = Input(shape=(self.dim, self.dim, 3))
        
        base_model = DenseNet121(input_tensor=input_tensor, weights='imagenet', include_top=False)
        base_model.layers.pop()
        # add a global spatial average pooling layer
        x = base_model.layers[-1].output
        x = GlobalAveragePooling2D()(x)
        # let's add a fully-connected layer
        x = Dense(10, activation='relu')(x)
        # and a logistic layer -- let's say we have 200 classes
        predictions = Dense(2)(x)

        # this is the model we will train
        self.model = Model(inputs=base_model.input, outputs=predictions)
    
    def train(self, epochs=1):
        
        df = pd.read_csv('processed_data/{}/df.csv'.format(self.img_class))
        
        train_df = df[df['subset'] == 'train']
        
        train_df = self._clean_df(train_df)
        
        datagen=ImageDataGenerator(rescale=1./255)
        img_folder_path = "processed_data/image_pool/{0}_{0}/".format(self.dim)
        train_generator=datagen.flow_from_dataframe(dataframe=train_df, directory=img_folder_path, 
                                                    x_col="file_name", y_col=['score', 'std'],
                                                    class_mode="other", target_size=(self.dim, self.dim), batch_size=32)
        
        STEP_SIZE_TRAIN=train_generator.n//train_generator.batch_size
        
        self.model.compile(optimizer='adam', loss='mse')
        self.model.fit_generator(generator=train_generator,
                            steps_per_epoch=STEP_SIZE_TRAIN,
                            epochs=epochs)
        
        
    def predict(self, df):        
        datagen=ImageDataGenerator(rescale=1./255)
        img_folder_path = "processed_data/image_pool/{0}_{0}/".format(self.dim)

        a_generator=datagen.flow_from_dataframe(dataframe=df, directory=img_folder_path, 
                                                    x_col="file_name", y_col=['score', 'std'],
                                                    class_mode="other", target_size=(self.dim, self.dim), batch_size=1)

        pred_values = self.model.predict_generator(generator=a_generator, steps=len(df))
        
        return pred_values[:, 0], pred_values[:, 1]
            
        
    def load(self, version='default'):
            
        output_path = self.model_data_path(version)
        
        if not os.path.exists(output_path):
            return
        
        self.model = keras.models.load_model(output_path + '/model.h5')


    def _save(self, output_path):
        self.model.save(output_path + '/model.h5')