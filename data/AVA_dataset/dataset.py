import pandas as pd
from PIL import Image
from torchvision import transforms
import torch

class AVA(torch.utils.data.Dataset):
    def __init__(self, aesthetic_path, train, transform=None):
        super().__init__()
        self.df = pd.read_csv('data/AVA_dataset/proc_AVA.csv')
        
        if train is not None:
            self.df = self.df[self.df['Train'] == train]
        
        if aesthetic_path is not None:
            self.aesth_df = pd.read_csv(aesthetic_path, names=['Image ID'])
            self.df = self.df.merge(self.aesth_df)
        
        self.transform = transform
        
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        item = self.df.iloc[idx]
        img_id = item['Image ID']
        img = Image.open('data/AVA_dataset/images/{}.jpg'.format(int(img_id))).convert('RGB')
        
        if self.transform is not None:
            img = self.transform(img)
        
        prob_distr = item[[1,2,3,4,5,6,7,8,9,10]].astype(int).values
        prob_distr = prob_distr / prob_distr.sum()
        prob_distr = torch.from_numpy(prob_distr).float()
        
        return img, prob_distr
        
class AVA2(torch.utils.data.Dataset):
    def __init__(self, aesthetic_path, train, transform=None):
        super().__init__()
        self.df = pd.read_csv('data/AVA_dataset/proc_AVA.csv')
        
        if train is not None:
            self.df = self.df[self.df['Train'] == train]
        
        if aesthetic_path is not None:
            self.aesth_df = pd.read_csv(aesthetic_path, names=['Image ID'])
            self.df = self.df.merge(self.aesth_df)
        
        top_10_perc_size = int(len(self.df) * 0.1)
        
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
        img = Image.open('data/AVA_dataset/images/{}.jpg'.format(int(img_id))).convert('RGB')
        
        if self.transform is not None:
            img = self.transform(img)
        
        return img, item['Beautiful'].astype(float)
    
class AVA_ID(torch.utils.data.Dataset):
    def __init__(self, aesthetic_path, train, transform=None):
        super().__init__()
        self.df = pd.read_csv('data/AVA_dataset/proc_AVA.csv')
        
        if train is not None:
            self.df = self.df[self.df['Train'] == train]
        
        if aesthetic_path is not None:
            self.aesth_df = pd.read_csv(aesthetic_path, names=['Image ID'])
            self.df = self.df.merge(self.aesth_df)
        
        self.transform = transform
        
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        item = self.df.iloc[idx]
        img_id = item['Image ID']
        img = Image.open('data/AVA_dataset/images/{}.jpg'.format(int(img_id))).convert('RGB')
        
        if self.transform is not None:
            img = self.transform(img)
        
        return img, int(img_id)