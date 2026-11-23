import torch
import torch.nn as nn


class CNN_SVHN(nn.Module):
    def __init__(self, num_classes=10):
        super(CNN_SVHN, self).__init__()
        c = [32, 64, 128, 256]
        self.features = nn.Sequential(
            nn.Conv2d(3, c[0], kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(c[0]),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(c[0], c[1], kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(c[1]),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(c[1], c[2], kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(c[2]),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(c[2], c[3], kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(c[3]),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self.classifier = nn.Linear(c[3] * 4, num_classes, bias=False)

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

class CNN_MNIST(nn.Module):
    def __init__(self, num_classes=10):
        super(CNN_MNIST, self).__init__()
        c = [32, 64, 128, 256]
        self.features = nn.Sequential(
            nn.Conv2d(1, c[0], kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(c[0]),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(c[0], c[1], kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(c[1]),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(c[1], c[2], kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(c[2]),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(c[2], c[3], kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(c[3]),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self.classifier = nn.Linear(c[3], num_classes, bias=False)

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

if __name__ == "__main__":
    print("done")
