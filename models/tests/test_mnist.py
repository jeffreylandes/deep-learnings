from torchvision.datasets import MNIST
from torchvision.transforms import transforms
from torch.utils.data import DataLoader


def test_mnist():
    root_path = "models/datasets/data"
    transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.1307,), (0.3081,))
                ])
    training = MNIST(root_path, train=True, transform=transform)
    training_loader = DataLoader(training, batch_size=1)
    validation = MNIST(root_path, train=False, transform=transform)
    validation_loader = DataLoader(validation, batch_size=1)
    next(iter(training_loader))
    next(iter(validation_loader))
