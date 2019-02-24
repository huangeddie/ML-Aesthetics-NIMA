import yaml
import os
import sklearn.mixture as mixture
import pandas as pd
import numpy as np
import models
import pickle
import keras

class VQModel(models.Model):
    def __init__(self):
        self.dir_name = "vector_quantization"
        
        stream = open("models/{}/config.yml".format(self.dir_name), "r")
        config = yaml.load(stream)
        
        self.dim = int(config['dim'])
        self.img_class = config['img_class']
        self.num_of_cluster = 64
        self.cluster_gm = mixture.GaussianMixture(n_components=self.num_of_cluster)
        self.neural_network = keras.models.Sequential([
            keras.layers.Dense(int(self.num_of_cluster * 1.5), input_shape=(self.num_of_cluster,)),
            keras.layers.Dense(int(self.num_of_cluster * 0.5)),
            keras.layers.Activation('relu'),
            keras.layers.Dense(2),
        ])
        
    
    def train(self):
        df = pd.read_csv('processed_data/{}/df.csv'.format(self.img_class))
        
        train_df = df[df['subset'] == 'train']
        
        train_imgs, train_scores, train_std = self.load_data(train_df)
        
        # slice the images into 16 * 16 pixel chuncks
        self.vq_dim = 16
        train_imgs = np.array(train_imgs)
        quantized_chuncks = self._quantize_imgs(train_imgs, self.vq_dim)
        
        # construct feature vectors
        self.cluster_gm.fit(quantized_chuncks)
        chunck_labels = self.cluster_gm.predict(quantized_chuncks)
        feature_vectors = self._build_feature_vectors(chunck_labels, len(train_imgs))
            
        # train the nn on the feature vectors
        values = np.array([train_scores, train_std]).T
        self.neural_network.fit(feature_vectors, values, epochs=10, batch_size=32)
        
        
    def _quantize_imgs(self, images, vq_dim):
        """
        Slice each image into vq_dim * vq_dim number of chuncks. 
        """
        quantized_chuncks = []
        
        for img in images:
            img_slices = np.split(img, vq_dim, axis=1)
            
            for slice_ in img_slices:
                chuncks = np.split(slice_, vq_dim, axis=0)
                chuncks = list(map(lambda c : c.flatten(), chuncks))
                quantized_chuncks.extend(chuncks)
                
        return quantized_chuncks
    
    
    def _build_feature_vectors(self, labels,  num_of_img):
        """
        Construct histogram for each image as feature vector
        """
        feature_vectors = []
        num_chunck_per_img = self.vq_dim**2
        
        for i in range(num_of_img):
            data_labels = labels[i * num_chunck_per_img:(i+1)*num_chunck_per_img]
            feature_vectors.append(np.histogram(data_labels, 
                                          bins=np.arange(self.num_of_cluster + 1), 
                                          density=True)[0])
        return feature_vectors
        
        
    def predict(self, df):
        imgs, _ , _ = self.load_data(df)
        
        quantized_chuncks = self._quantize_imgs(imgs, self.vq_dim)
        chunck_labels = self.cluster_gm.predict(quantized_chuncks)
        feature_vectors = self._build_feature_vectors(chunck_labels, len(imgs))
        
        pred_values = self.neural_network.predict(feature_vectors)
        
        return pred_values[:, 0], pred_values[:, 1]
            
        
    def load(self, version=None):
        if version is None:
            version = self.latest_version()
        
        if version is None:
            return
            
        output_path = self.model_data_path(version)
        
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
        
        
        
        
        

