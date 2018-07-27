import torch as tc
from torch import nn
from torch.nn import functional
from torch.utils.model_zoo import load_url as load_model
from torchvision import models

class AlexNetModel(nn.Module):
    def __init__(self, num_classes, dropout=0):
        super(Model, self).__init__()
        self.feature = nn.Sequential(
                self._build_block(1, 64, 7, 3),
                self._build_block(64, 128, 3, 2),
                self._build_block(128,256, 3, 2),
                self._build_block(256,128, 3, 2),
                self._build_block(128, 64, 3, 2)
                )
        self.classifier = nn.Sequential(
                nn.Dropout(dropout),
                nn.Linear(6*6*64, 576),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout),
                nn.Linear(576,576),
                nn.Relu(inplace=True),
                nn.Linear(576, num_classes))

    def _build_block(in_channel, out_channel, kernel_size, pool_size):
        return nn.Sequential(
            nn.Conv2d(in_channel, out_channel, kernel_size),
            nn.BatchNorm2d(out_channel),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(pool_size))

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), 64*6*6)
        x = self.classifier(x)
        return x

Model = AlexNetModel
