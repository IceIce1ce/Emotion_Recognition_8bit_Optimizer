import torch
import torch.nn as nn
import torch.nn.functional as F

class VGG19(nn.Module):
    def __init__(self):
      super(VGG19, self).__init__()
      self.vgg_model = nn.Sequential(nn.Conv2d(3, 64, kernel_size=3, padding=1),
                                     nn.BatchNorm2d(64),
                                     nn.ReLU(inplace=True),
                                     nn.Conv2d(64, 64, kernel_size=3, padding=1),
                                     nn.BatchNorm2d(64),
                                     nn.ReLU(inplace=True),
                                     nn.MaxPool2d(kernel_size=2, stride=2),
                                     nn.Conv2d(64, 128, kernel_size=3, padding=1),
                                     nn.BatchNorm2d(128),
                                     nn.ReLU(inplace=True),
                                     nn.Conv2d(128, 128, kernel_size=3, padding=1),
                                     nn.BatchNorm2d(128),
                                     nn.ReLU(inplace=True),
                                     nn.MaxPool2d(kernel_size=2, stride=2),
                                     nn.Conv2d(128, 256, kernel_size=3, padding=1),
                                     nn.BatchNorm2d(256),
                                     nn.ReLU(inplace=True),
                                     nn.Conv2d(256, 256, kernel_size=3, padding=1),
                                     nn.BatchNorm2d(256),
                                     nn.ReLU(inplace=True),
                                     nn.Conv2d(256, 256, kernel_size=3, padding=1),
                                     nn.BatchNorm2d(256),
                                     nn.ReLU(inplace=True),
                                     nn.Conv2d(256, 256, kernel_size=3, padding=1),
                                     nn.BatchNorm2d(256),
                                     nn.ReLU(inplace=True),
                                     nn.MaxPool2d(kernel_size=2, stride=2),
                                     nn.Conv2d(256, 512, kernel_size=3, padding=1),
                                     nn.BatchNorm2d(512),
                                     nn.ReLU(inplace=True),
                                     nn.Conv2d(512, 512, kernel_size=3, padding=1),
                                     nn.BatchNorm2d(512),
                                     nn.ReLU(inplace=True),
                                     nn.Conv2d(512, 512, kernel_size=3, padding=1),
                                     nn.BatchNorm2d(512),
                                     nn.ReLU(inplace=True),
                                     nn.Conv2d(512, 512, kernel_size=3, padding=1),
                                     nn.BatchNorm2d(512),
                                     nn.ReLU(inplace=True),
                                     nn.MaxPool2d(kernel_size=2, stride=2),
                                     nn.Conv2d(512, 512, kernel_size=3, padding=1),
                                     nn.BatchNorm2d(512),
                                     nn.ReLU(inplace=True),
                                     nn.Conv2d(512, 512, kernel_size=3, padding=1),
                                     nn.BatchNorm2d(512),
                                     nn.ReLU(inplace=True),
                                     nn.Conv2d(512, 512, kernel_size=3, padding=1),
                                     nn.BatchNorm2d(512),
                                     nn.ReLU(inplace=True),
                                     nn.Conv2d(512, 512, kernel_size=3, padding=1),
                                     nn.BatchNorm2d(512),
                                     nn.ReLU(inplace=True),
                                     nn.MaxPool2d(kernel_size=2, stride=2),
                                     nn.AvgPool2d(kernel_size=1, stride=1))
      self.classifier = nn.Linear(512, 7)
    def forward(self, x):
      x = self.vgg_model(x)
      x = x.view(x.size(0), -1)
      x = F.dropout(x, p=0.5, training=self.training)
      return self.classifier(x)