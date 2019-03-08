import pandas as pd
import matplotlib.pyplot as plt
import os
import numpy as np
import skimage.io
from scipy.stats.stats import pearsonr

class Model:
    def __init__(self):
        """
        Constructor for a model that initializes parameters by reading the config file
        """
        raise Exception('Not Implemented')
        
    def _configure(self, img_class, dir_path, feature, dim):
        """
        This function should be called at the beginning of the init func of every subclass of this model
        """
        self.img_class = img_class
        self.dir_path = dir_path
        self.feature = feature
        self.dim = dim
    
    
    def load(self, version_name='default'):
        """
        Load model from version history
        :param version: index of the version to load, or None to load the most recent
        """
        raise Exception('Not Implemented')
    
    
    def train(self, epochs=None):
        """
        Train model
        """
        raise Exception('Not Implemented')
    
    
    def predict(self, df):
        """
        Returns the predicted values. Type is based on its feature
        """
        raise Exception('Not Implemented')
        
        
    def _version_path(self, name):
        return '{}/versions/{}/'.format(self.dir_path, name)
    
    def _weight_path(self, vers_name):
        return self.version_path(vers_name) + '/weights/'
    
    def _stats_path(self, vers_name):
        return self.version_path(vers_name) + '/stats/'
    
    def create_version(self, name='default'):
        """
        • Saves the model
        • Evaluates the model from the testing data and saves the results
        :param version: index of the version to save to, or None to create a new version
        :return : version index
        """
        print("Creating version {}...".format(name))
        
        df = pd.read_csv('processed_data/{}/df.csv'.format(self.img_class))
        test_df = df[df['subset'] == 'test']
        test_df = self._clean_df(test_df)
        
        true_values = test_df[self.feature]
        
        pred_values = self.predict(test_df)
        
        
        
        # Save the statistics as a graph into stats folder
        plt.figure(figsize=(10, 5))
        
        score_corr = pearsonr(pred_scores, test_scores)[0]
        std_corr = pearsonr(pred_stds, test_stds)[0]
        
        print('Score Corr: {} | STD Corr: {}'.format(score_corr, std_corr))
        
        plt.subplot(1, 2, 1)
        plt.title('Scores with correlation {}'.format(round(score_corr, 3)))
        plt.xlabel('Predicted')
        plt.ylabel('Truth')
        plt.scatter(pred_scores, test_scores)
        
        plt.subplot(1, 2, 2)
        plt.title('STDs with correlation {}'.format(round(std_corr,3)))
        plt.xlabel('Predicted')
        plt.ylabel('Truth')
        plt.scatter(pred_stds, test_stds)
        
        stats_path = self.version_path(name) + '/stats/'.format(self.dir_name, name)
        if not os.path.exists(stats_path):
            os.makedirs(stats_path)
        plt.savefig(stats_path + 'stats.png')
        
        # Save the model into this name folder
        print('Saving model...')
        output_path = self.version_path(name)
        output_path += '/model_data/'
        if not os.path.exists(output_path):
            os.makedirs(output_path)
            
        self._save(output_path)
        
        print('Saved {} to version {}'.format(self.__class__.__name__, name))

    
    
    def _save(self, output_path):
        """
        Save the model into the specified output path.
        This function is abstract and must be defined by the subclasses
        :param output_path: the path to save into
        """
        
        raise Exception('Not implemented')
    
    def _clean_df(self, df):
        """
        Drops the rows of images that do not exist in the processed module
        """
        img_folder_path = "processed_data/image_pool/{0}_{0}/".format(self.dim)
        bad_row_idcs = []
        for i, row in df.iterrows():
            image_id = int(row['id'])
            if not os.path.exists('{}/{}.png'.format(img_folder_path, image_id)):
                bad_row_idcs.append(i)
        
        df = df.drop(bad_row_idcs).reset_index()
        
        return df
    
    def load_data(self, df):
        """
        Load images based on meta data df passed in
        Ignores rows with missing images
        """
        
        # Clean df
        df = self._clean_df(df)
        
        # Load images
        img_folder_path = "processed_data/image_pool/{0}_{0}/".format(self.dim)
        img_paths = list(map(lambda id: img_folder_path + str(id) + ".png", df["id"].values))
        list_of_images = list(skimage.io.imread_collection(img_paths))
        return list_of_images, df['norm_score'], df['norm_std']
    
    
