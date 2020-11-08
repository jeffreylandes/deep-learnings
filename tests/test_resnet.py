from models.resnet import ResNet
import torch


def test_resnet():
    tensor = torch.rand((1, 3, 200, 200))
    model = ResNet()
    model(tensor)
