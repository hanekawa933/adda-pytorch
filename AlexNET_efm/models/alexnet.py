"""AlexNET model for ADDA."""

import torch
import torch.nn.functional as F
from torch import nn


class AlexNetEncoder(nn.Module):
  """AlexNet encoder model for ADDA."""

  def __init__(self, global_params=None):
      """Init AlexNet encoder."""
      super(AlexNetEncoder, self).__init__()
      self.restored = False
      self.features = nn.Sequential(
          nn.Conv2d(1, 96, kernel_size=11, stride=4, padding=2),
          nn.ReLU(inplace=True),
          nn.MaxPool2d(kernel_size=3, stride=2),
          nn.Conv2d(96, 256, kernel_size=5, padding=2),
          nn.ReLU(inplace=True),
          nn.MaxPool2d(kernel_size=3, stride=2),
          nn.Conv2d(256, 384, kernel_size=3, padding=1),
          nn.ReLU(inplace=True),
          nn.Conv2d(384, 384, kernel_size=3, padding=1),
          nn.ReLU(inplace=True),
          nn.Conv2d(384, 256, kernel_size=3, padding=1),
          nn.ReLU(inplace=True),
          nn.MaxPool2d(kernel_size=3, stride=2)
      )
      self.avgpool = nn.AdaptiveAvgPool2d((6, 6))

  def forward(self, inputs):
      # See note [TorchScript super()]
      x = self.features(inputs)
      x = self.avgpool(x)
      x = torch.flatten(x, 1)
      return x

class AlexNetClassifier(nn.Module):
  def __init__(self):
      super(AlexNetClassifier, self).__init__()
      self.restored = False
      self.classifier = nn.Sequential(
          nn.Dropout(p=0.5),
          nn.Linear(256 * 6 * 6, 4096),
          nn.ReLU(inplace=True),
          nn.Dropout(p=0.5),
          nn.Linear(4096, 4096),
          nn.ReLU(inplace=True),
          nn.Linear(4096, 10),
      )
  
  def forward(self, inputs):
      """Forward the LeNet classifier."""
      out = self.classifier(inputs)
      return out
