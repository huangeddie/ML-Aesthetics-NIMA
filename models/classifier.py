from torch import nn

class Classifier(nn.Module):
    
    def __init__(self, input_size, num_classes):
        super().__init__()
        
        self.input_size = input_size
        
        self.main = nn.Sequential(
            nn.Linear(3 * (input_size ** 2), num_classes),
            nn.Softmax(1)
        )
        
    def forward(self, img):
        out = self.main(img.view(-1, 3 * (self.input_size ** 2)))
        return out