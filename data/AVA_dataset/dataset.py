import torch.utils.data
import pandas as pd
from skimage import io
from torchvision import transforms

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
        img = io.imread('data/AVA_dataset/images/all_images/{}.jpg'.format(img_id))
        
        
        
        if self.transform is not None:
            img = self.transform(img)
            
        img = transforms.ToTensor()(img)
        
        return img, item[[1,2,3,4,5,6,7,8,9,10]]
        