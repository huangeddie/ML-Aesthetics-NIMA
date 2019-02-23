import pandas as pd
import matplotlib.pyplot as plt

from PIL import Image
import numpy as np
import os


def load_images(df, max_images):
    """
    Returns all images in their original dimensions from a dataframe
    """
    images = []
    for i, row in df.iterrows():
        image_id = int(row['id'])
        input_file_path = 'images/images/{}.jpg'.format(image_id)
        img = Image.open(input_file_path).convert('RGB')
        images.append(img)
        if len(images) >= max_images:
            break
    return images
