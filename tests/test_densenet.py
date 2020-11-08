from models.densenet import DenseNet
import torch


def test_densenet():
    tensor = torch.rand((1, 3, 200, 200))
    model = DenseNet()
    model(tensor)
