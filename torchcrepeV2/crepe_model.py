import math
from torch import nn
import torch.nn.functional as F
import torch


def calc_same_pad(i: int, k: int, s: int, d: int) -> int:
        return max((math.ceil(i / s) - 1) * s + (k - 1) * d + 1 - i, 0)

class TorchCrepe(nn.Module):
    def __init__(self, model_capacity="full"):
        super().__init__()
        capacity_multiplier = {
            'tiny': 4, 'small': 8, 'medium': 16, 'large': 24, 'full': 32
        }[model_capacity]

        self.layers = [1, 2, 3, 4, 5, 6]
        self.filters = [n * capacity_multiplier for n in [32, 4, 4, 4, 8, 16]]
        self.widths = [512, 64, 64, 64, 64, 64]
        self.strides = [(4, 1), (1, 1), (1, 1), (1, 1), (1, 1), (1, 1)]

        self.conv_list = nn.ModuleList([nn.Conv2d(1, 1024, (512, 1), stride=(4, 1)),
                                        nn.Conv2d(1024, 128, (64, 1), stride=(1, 1)),
                                        nn.Conv2d(128, 128, (64, 1), stride=(1, 1)),
                                        nn.Conv2d(128, 128, (64, 1), stride=(1, 1)),
                                        nn.Conv2d(128, 256, (64, 1), stride=(1, 1)),
                                        nn.Conv2d(256, 512, (64, 1), stride=(1, 1))])
        self.batchnorm_list = nn.ModuleList([nn.BatchNorm2d(1024, eps=1e-3, momentum=0.99),
                                            nn.BatchNorm2d(128, eps=1e-3, momentum=0.99),
                                            nn.BatchNorm2d(128, eps=1e-3, momentum=0.99),
                                            nn.BatchNorm2d(128, eps=1e-3, momentum=0.99),
                                            nn.BatchNorm2d(256, eps=1e-3, momentum=0.99),
                                            nn.BatchNorm2d(512, eps=1e-3, momentum=0.99)])
        self.classifier = nn.Linear(2048, 360)
        self.drop_layer = nn.Dropout(0.25)

    def forward(self, x):
        x = x.unsqueeze(1).unsqueeze(-1)   # (1024, 1, 1)
        counter = 0
        for l, f, w, s in zip(self.layers, self.filters, self.widths, self.strides):
            pad_h = calc_same_pad(i=x.shape[-2], k=w, s=s[0], d=1)
            pad_w = calc_same_pad(i=x.shape[-1], k=1, s=s[1], d=1)
            x = F.pad(x, [pad_w // 2, pad_w - pad_w // 2, pad_h // 2, pad_h - pad_h // 2])
            x = self.conv_list[counter](x)
            x = nn.ReLU()(x)
            x = self.batchnorm_list[counter](x)
            x = nn.MaxPool2d((2, 1), (2, 1))(x)
            x = self.drop_layer(x)
            
            counter += 1

        x = torch.transpose(x, 1, 2)
        x = x.flatten(start_dim=1)
        x = self.classifier(x)
        x = nn.Sigmoid()(x)

        return x