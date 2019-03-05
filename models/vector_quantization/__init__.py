import yaml
import os
from sklearn.cluster import MiniBatchKMeans
from sklearn.ensemble import RandomForestRegressor
import pandas as pd
import numpy as np
import models
import pickle
import keras
import pdb
from tqdm import tqdm

class VQModel(models.Model):
    def __init__(self):
        self.dir_name = "vector_quantization"
        
        stream = open("models/{}/config.yml".format(self.dir_name), "r")
        config = yaml.load(stream)
        
        self.dim = int(config['dim'])
        self.img_class = config['img_class']
        self.num_of_cluster = config['num_of_cluster']
        self.n = config['num_chunk_per_side']
        self.cluster_km = MiniBatchKMeans(n_clusters=self.num_of_cluster)
        self.regr = RandomForestRegressor(n_estimators=100)
#         self.neural_network = keras.models.Sequential([
#             keras.layers.Dense(int(self.num_of_cluster * 1.5), input_shape=(self.num_of_cluster,)),
#             keras.layers.Activation('relu'),
#             keras.layers.Dense(int(self.num_of_cluster * 0.5)),
#             keras.layers.Activation('relu'),
#             keras.layers.Dense(2),
#         ])
        
    
    def train(self, epochs=10):
        print("VQModel Training:")
        print("Loading meta data ...")
        df = pd.read_csv('processed_data/{}/df.csv'.format(self.img_class))
        
        train_df = df[df['subset'] == 'train']
        print("Loading images")
        
        train_imgs, train_scores, train_std = self.load_data(train_df)
        train_imgs = np.array(train_imgs)
        
        # slice the images into n ** 2 chunks
        quantized_chunks = self._quantize_imgs(train_imgs)
        
        patch_size = 500
        
        for i in tqdm(range(int(np.ceil(train_df.shape[0] / patch_size))), desc="Training KMeans"):
            self.cluster_km.partial_fit(quantized_chunks[i*patch_size : (i+1)*patch_size])
        
        print("Predicting")
        
        # construct feature vectors
        chunk_labels = []
        for i in tqdm(range(int(np.ceil(train_df.shape[0] / patch_size))), 
                      desc="Predicting train clusters"):
            chunk_labels.extend(self.cluster_km.predict(quantized_chunks
                                                        [i*patch_size : (i+1)*patch_size]))
        
        print("Building feature vectors")
        
        feature_vectors = self._build_feature_vectors(chunk_labels, len(train_imgs))
        
        print("Training neural network")
        
        # train the nn on the feature vectors
#         self.neural_network.compile(optimizer='adam',
#                                   loss='mse',
#                                   metrics=['mse'])
        values = np.array([train_scores, train_std]).T
#         pdb.set_trace()
        self.regr.fit(feature_vectors, values)
#         self.neural_network.fit(feature_vectors, values, epochs=epochs, batch_size=32)
    
        # free memory
        del train_imgs
        
        
        
    def _quantize_imgs(self, images):
        """
        Slice each image into n * n number of chunks. 
        """
        quantized_chunks = []
        
        for img in images:
            img_slices = np.split(img, self.n, axis=1)
            
            for slice_ in img_slices:
                chunks = np.split(slice_, self.n, axis=0)
                chunks = list(map(lambda c : c.flatten(), chunks))
                quantized_chunks.extend(chunks)
                
        return quantized_chunks
    
    
    def _build_feature_vectors(self, labels,  num_of_img):
        """
        Construct histogram for each image as feature vector
        """
        feature_vectors = []
        num_chunk_per_img = self.n**2
        
        for i in range(num_of_img):
            data_labels = labels[i * num_chunk_per_img:(i+1)*num_chunk_per_img]
            feature_vectors.append(np.histogram(data_labels, 
                                          bins=np.arange(self.num_of_cluster + 1), 
                                          density=True)[0])
        return np.array(feature_vectors)
        
        
    def predict(self, df):
        imgs, _ , _ = self.load_data(df)
        
        print("Predicting VQModel:")
        print("Slicing test data")
        
        quantized_chunks = self._quantize_imgs(imgs)
        
        print("KMeans predicting")
        
        
        # construct feature vectors
        patch_size = 500
        chunk_labels = []
        for i in tqdm(range(int(np.ceil(len(imgs) / patch_size))), 
                      desc="Predicting test clusters"):
            chunk_labels.extend(self.cluster_km.predict(quantized_chunks
                                                        [i*patch_size : (i+1)*patch_size]))
        
        print("Building feature vector")
        feature_vectors = self._build_feature_vectors(chunk_labels, len(imgs))
        
        print("")
        pred_values = self.regr.predict(feature_vectors)
#         pred_values = self.neural_network.predict(feature_vectors)
        
        return pred_values[:, 0], pred_values[:, 1]
            
        
    def load(self, version='default'):
        # check if the version exists
        output_path = self.model_data_path(version)
        
        # if it doesn't do not load
        if not os.path.exists(output_path):
            return
        
        with open(output_path + '/cluster_km', 'rb') as f:
            self.cluster_km = pickle.load(f)
            
        self.neural_network = keras.models.load_model(output_path + '/neural_network.h5')

        
    def _save(self, output_path):
        cluster_km_s = pickle.dumps(self.cluster_km)
        with open(output_path + '/cluster_km', 'wb') as f:
            f.write(cluster_km_s)
            
        self.neural_network.save(output_path + '/neural_network.h5')

        
        