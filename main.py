import argparse
import os
import glob
import numpy as np
import torch
from PIL import Image
from torch import nn
from torchvision import models, transforms

def rate(img_path):
    """
    Returns: Scores, mean, std
    """
    # Number of classes in the dataset
    num_classes = 10

    model_ft = models.densenet121(pretrained=False)
    num_ftrs = model_ft.classifier.in_features
    model_ft.classifier = nn.Sequential(
        nn.Linear(num_ftrs, num_classes),
        nn.Softmax(1)
    )

    # Weight Path
    weight_path = 'weights/dense121_all.pt'

    # Load weights
    assert os.path.exists(weight_path)
    model_ft.load_state_dict(torch.load(weight_path))
    model_ft.eval()

    image_filenames = sorted(glob.glob(img_path))

    transform = transforms.Compose([
        transforms.Resize(224),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
    ])

    for index, filename in enumerate(image_filenames):
        img = Image.open(filename).convert('RGB')
        img = transform(img)

        with torch.no_grad():
            scores = model_ft(img.view(1, 3, 224, 224))
            weighted_votes = torch.arange(10, dtype=torch.float) + 1
            mean = torch.matmul(scores, weighted_votes)
            std = torch.sqrt((scores * torch.pow((weighted_votes - mean.view(-1, 1)), 2)).sum(dim=1))

        print("{:.4f} {:.4f}--- {}".format(mean.item(), std.item(), filename))
   


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Use DenseNet NIMA')

    parser.add_argument('--img_path', type=str, default="images/*", help='Path to input image')
    args = parser.parse_args()
    rate(args.img_path)
