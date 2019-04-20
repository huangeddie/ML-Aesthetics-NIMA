from torch import nn

class Classifier(nn.Module):
    
    def __init__(self, num_classes):
        super().__init__()
        
        self.convs = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=(5,5), stride=2, padding=2, bias=False),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=(5,5), stride=2, padding=2, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=(5,5), stride=2, padding=2, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=(5,5), stride=2, padding=2, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 256, kernel_size=(5,5), stride=2, padding=2, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(),
        )
        self.linear = nn.Linear(256 * 8 * 8, num_classes)
        
    def forward(self, img):
        out = self.convs(img)
        out = self.linear(out.view(img.shape[0], -1))
        return out