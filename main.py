import argparse
import os
import glob
import torch
from PIL import Image
# from torch import nn
from torchvision import models, transforms
from model import model_setenv, get_model, model_load, model_device
import pdb

def rate(img_path):
    """
    Returns: Scores, mean, std
    """
    # # Number of classes in the dataset
    # num_classes = 10

    # model_ft = models.densenet121(pretrained=False)
    # num_ftrs = model_ft.classifier.in_features
    # model_ft.classifier = nn.Sequential(
    #     nn.Linear(num_ftrs, num_classes),
    #     nn.Softmax(1)
    # )

    # # Weight Path
    # weight_path = 'weights/dense121_all.pt'

    # # Load weights
    # assert os.path.exists(weight_path)
    # model_ft.load_state_dict(torch.load(weight_path))
    # model_ft.eval()

    model_setenv()
    device = model_device()
    model = get_model()
    model_name = 'models/ImageNima.pth'
    model_load(model, model_name)
    model = model.to(device)
    model.eval()

    image_filenames = sorted(glob.glob(img_path))

    transform = transforms.Compose([
        transforms.Resize(224),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
    ])
    weighted_votes = torch.arange(10, dtype=torch.float) + 1
    weighted_votes = weighted_votes.to(device)

    for index, filename in enumerate(image_filenames):
        img = Image.open(filename).convert('RGB')
        img = transform(img).to(device)

        with torch.no_grad():
            scores = model(img.view(1, 3, 224, 224))
            mean = torch.matmul(scores, weighted_votes)
            std = torch.sqrt((scores * torch.pow((weighted_votes - mean.view(-1, 1)), 2)).sum(dim=1))

        print("{:.4f} {:.4f}--- {}".format(mean.item(), std.item(), filename))
   


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Use DenseNet NIMA')

    parser.add_argument('--input', type=str, default="images/*", help='Path to input image')
    args = parser.parse_args()
    rate(args.input)
