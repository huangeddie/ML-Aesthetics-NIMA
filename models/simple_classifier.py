from torch import nn

class SimpleClassifier(nn.Module):
    
    def __init__(self, num_classes):
        super().__init__()
        
        self.linear = nn.Sequential(
            nn.Linear(256 * 256 * 3, num_classes),
            nn.Softmax(1)
        )
        
        
    def forward(self, img):
        out = self.linear(img)
        return out