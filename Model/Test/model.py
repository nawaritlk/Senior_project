import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.models as models

model_urls = {
    'vgg11': 'https://download.pytorch.org/models/vgg11-bbd30ac9.pth',
    'vgg13': 'https://download.pytorch.org/models/vgg13-c768596a.pth',
    'vgg16': 'https://download.pytorch.org/models/vgg16-397923af.pth',
    'vgg19': 'https://download.pytorch.org/models/vgg19-dcbb9e9d.pth',
    'vgg11_bn': 'https://download.pytorch.org/models/vgg11_bn-6002323d.pth',
    'vgg13_bn': 'https://download.pytorch.org/models/vgg13_bn-abd245e5.pth',
    'vgg16_bn': 'https://download.pytorch.org/models/vgg16_bn-6c64b313.pth',
    'vgg19_bn': 'https://download.pytorch.org/models/vgg19_bn-c79401a0.pth',
}

import os
BASE_PATH = os.getcwd()

class VGG19(nn.Module):
    def __init__(self,vgg19_path=BASE_PATH+r'/models/vgg19-dcbb9e9d.pth'):
        super(VGG19, self).__init__()
        vgg19_features = models.vgg19(pretrained=False)
        print(vgg19_path)
        vgg19_features.load_state_dict(torch.load(vgg19_path),strict=False)
        self.features = vgg19_features.features

        for param in self.features.parameters():
            param.requires_grad = False

    def print(self):
        print("--VGG19--"+self.features)


    def forward(self, x):
        # Alllayers = {'0':'conv1_1',
        #     '1':'relu1_1',
        #     '2':'conv1_2',
        #     '3':'relu1_2',
        #     '4':'max_1',
        #     '5':'conv2_1',
        #     '6':'relu2_1',
        #     '7':'conv2_2',
        #     '8':'relu2_2',
        #     '9':'max_2',
        #     '10':'conv3_1',
        #     '11':'relu3_1',
        #     '12':'conv3_2',
        #     '13':'relu3_2',
        #     '14':'conv3_3',
        #     '15':'relu3_3',
        #     '16':'conv3_4',
        #     '17':'relu3_4',
        #     '18':'max_3',
        #     '19':'conv4_1',
        #     '20':'relu4_1',
        #     '21':'conv4_2',
        #     '22':'relu4_2',
        #     '23':'conv4_3',
        #     '24':'relu4_3',
        #     '25':'conv4_4',
        #     '26':'relu4_4',
        #     '27':'max_4',
        #     '28':'conv5_1',
        #     '29':'relu5_1',
        #     '30':'conv5_2',
        #     '31':'relu5_2',
        #     '32':'conv5_3',
        #     '33':'relu5_3',
        #     '34':'conv5_4',
        #     '35':'relu5_4',
        #     '36':'max_5'}
        layers = {'3': 'relu1_2', '8': 'relu2_2', '17': 'relu3_4', '22': 'relu4_2', '26': 'relu4_4', '35': 'relu5_4'}
        features = {}
        for name, layer in self.features._modules.items():
            x = layer(x)
            if name in layers:
                features[layers[name]] = x
        return features