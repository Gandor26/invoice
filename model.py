import torch as tc
from torch import nn
from torch.nn import functional
from torch.utils.model_zoo import load_url as load_model
from torchvision import models
from math import ceil, floor

class SpatialPyramidPooling2d(nn.Module):
    def __init__(self, *levels, base_pooling=nn.MaxPool2d):
        super(SpatialPyramidPooling2d, self).__init__()
        self.pooling = base_pooling
        self.levels = levels

    def forward(self, x):
        b, c, h, w = x.size()
        features = []
        for level in self.levels:
            kernel_size = ceil(h/level), ceil(w/level)
            stride = floor(h/level), floor(w/level)
            #padding = tuple(map(lambda k, s: ))
            pooling_layer = self.pooling(kernel_size=kernel_size, stride=kernel_size, padding=padding)
            features.append(pooling_layer(x).view(b, -1))
        return tc.cat(features, dim=-1)

def make_conv_block(in_channel, out_channel, kernel_size, stride=1, padding=0, add_pooling=False):
    base = nn.Sequential(
            nn.Conv2d(in_channel, out_channel, kernel_size=kernel_size, stride=stride, padding=padding),
            nn.BatchNorm2d(out_channel),
            nn.ReLU(inplace=True)
            )
    if add_pooling:
        block = nn.Sequential(base,
                nn.MaxPool2d(kernel_size=3, stride=2))
    else:
        block = base
    return block

class AlexNet(nn.Module):
    '''
        Model architecture
            feature extractor with 5 layers of convolution;
            classifier as 2 two layer feed-forward net
        Args:
            num_classes: how many classes in the end
            dropout: the dropout rate inside the classifier
    '''
    def __init__(self, num_classes, dropout=0.5):
        super(AlexNet, self).__init__()
        self.features = nn.Sequential(
                make_conv_block(1, 64, kernel_size=11, stride=4, padding=2, add_pooling=True),
                make_conv_block(64, 192, kernel_size=5, padding=2, add_pooling=True),
                make_conv_block(192, 384, kernel_size=3, padding=1),
                make_conv_block(384, 256, kernel_size=3, padding=1),
                make_conv_block(256, 256, kernel_size=3, padding=1, add_pooling=True)
                )
        self.classifier = nn.Sequential(
                nn.Dropout(dropout),
                nn.Linear(256*6*6, 512),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout),
                nn.Linear(512, 512),
                nn.ReLU(inplace=True),
                nn.Linear(512, num_classes)
                )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), 256*6*6)
        x = self.classifier(x)
        return x

class AlexBoWNet(AlexNet):
    '''
        Ensemble model with the image classifier and BoW model
        Args:
            num_classes: the number of classes in the end
            vocab_size: size of vocabulary; the dimension of WordCountVectorizer output
            dropout: dropout rate in both image classifier and bow classifier
    '''
    def __init__(self, num_classes, vocab_size, dropout=0.5):
        super(AlexBoWNet, self).__init__(num_classes, dropout)
        self.bow = nn.Sequential(
                nn.Linear(vocab_size, 512),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout),
                nn.Linear(512, 512),
                nn.ReLU(inplace=True),
                nn.Linear(512, num_classes)
                )

    def forward(self, image, word_count):
        image = self.features(image)
        image = image.view(image.size(0), -1)
        image_prediction = self.classifier(image)
        bow_prediction = self.bow(word_count)
        return image_prediction + bow_prediction
