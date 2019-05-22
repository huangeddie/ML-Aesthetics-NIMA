import argparse
import torch
from torch import nn
from torchvision import models
from torchvision import transforms
from PIL import Image
import os

def rate(img_path):
    """
    Returns: Scores, mean, std
    """
    # Number of classes in the dataset
    num_classes = 10

    model_ft = models.densenet121(pretrained=True)
    num_ftrs = model_ft.classifier.in_features
    model_ft.classifier = nn.Sequential(
        nn.Linear(num_ftrs,num_classes),
        nn.Softmax(1)
    )

    # Weight Path
    weight_path = 'weights/dense121_all.pt'

    # Load weights
    if os.path.exists(weight_path):
        model_ft.load_state_dict(torch.load(weight_path))
        print("Loaded saved weights from '{}'".format(weight_path))
    else:
        print("Starting weights from scratch")

    img = Image.open(img_path)
    transform = transforms.Compose([
            transforms.Resize(224),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
        ])
    img = transform(img)

    with torch.no_grad():
        scores = model_ft(img.view(1,3,224,224))
        weighted_votes = torch.arange(10, dtype=torch.float) + 1
        mean = torch.matmul(scores, weighted_votes)
        std = torch.sqrt((scores * torch.pow((weighted_votes - mean.view(-1,1)), 2)).sum(dim=1))
    return scores.numpy(), mean.item(), std.item()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Use DenseNet NIMA')

    parser.add_argument('img_path', type=str, help='Path to input image')
    args = parser.parse_args()

    img_path = args.img_path
    scores, mean, std = rate(img_path)
    print(scores, mean, std)