import pandas as pd
from PIL import Image
from torchvision import transforms
import torch

class AVA(torch.utils.data.Dataset):
    def __init__(self, aesthetic_path, transform=None):
        super().__init__()
        self.df = pd.read_csv('data/AVA_dataset/proc_AVA.csv')
        self.aesth_df = pd.read_csv(aesthetic_path, names=['Image ID'])
        self.df = self.df.merge(self.aesth_df)
        
        self.transform = transform
        
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        item = self.df.iloc[idx]
        img_id = item['Image ID']
        img = Image.open('data/AVA_dataset/images/{}.jpg'.format(img_id)).convert('RGB')
        
        if self.transform is not None:
            img = self.transform(img)
        
        prob_distr = item[[1,2,3,4,5,6,7,8,9,10]].values
        prob_distr = prob_distr / prob_distr.sum()
        prob_distr = torch.from_numpy(prob_distr).float()
        
        return img, prob_distr
        
class BinaryAVA(torch.utils.data.Dataset):
    def __init__(self, aesthetic_path, transform=None):
        super().__init__()
        self.df = pd.read_csv('data/AVA_dataset/proc_AVA.csv')
        self.aesth_df = pd.read_csv(aesthetic_path, names=['Image ID'])
        self.df = self.df.merge(self.aesth_df)
        
        beautiful_df = self.df.nlargest(top_10_perc_size, 'Mean Score')
        beautiful_df['Beautiful'] = 1
        
        ugly_df = self.df.nsmallest(top_10_perc_size, 'Mean Score')
        ugly_df['Beautiful'] = 0
        
        self.df = pd.concat([beautiful_df, ugly_df], ignore_index=True)
        
        self.transform = transform
        
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        item = self.df.iloc[idx]
        img_id = item['Image ID']
        img = Image.open('data/AVA_dataset/images/{}.jpg'.format(img_id)).convert('RGB')
        
        if self.transform is not None:
            img = self.transform(img)
        
        return img, item['Beautiful']