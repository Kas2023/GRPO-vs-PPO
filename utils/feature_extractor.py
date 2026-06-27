import torch.nn as nn

class MLPFeatureExtractor(nn.Module):
    def __init__(self, input_dim, feature_dim=64):
        super(MLPFeatureExtractor, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, feature_dim),
            nn.ReLU(),
        )
    
    def forward(self, x):
        return self.net(x)