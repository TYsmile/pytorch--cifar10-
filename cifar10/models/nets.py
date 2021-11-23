from torchvision import models
import torch.nn as nn


# 创建与训练的网络模型，修改全连接层使之适用cifar10数据集的分类
def resnet18():
    # 返回resnet18模型
    net = models.resnet18(pretrained=True)

    net.fc = nn.Sequential(
        nn.Linear(512, 128),
        nn.ReLU(inplace=True),
        nn.Dropout(0.6),
        nn.Linear(128, 10))

    return net


def resnet34():
    # 返回resnet34模型
    net = models.resnet34(pretrained=True)

    net.fc = nn.Sequential(
        nn.Linear(512, 128),
        nn.ReLU(inplace=True),
        nn.Dropout(0.6),
        nn.Linear(128, 10))

    return net


def resnet50():
    # 返回resnet50模型
    net = models.resnet50(pretrained=True)

    net.fc = nn.Sequential(
        nn.Linear(2048, 512),
        nn.ReLU(inplace=True),
        nn.Dropout(0.6),

        nn.Linear(512, 128),
        nn.ReLU(inplace=True),
        nn.Dropout(0.5),
        nn.Linear(128, 10))

    return net


def resnet101():
    # 返回resnet101模型
    net = models.resnet101(pretrained=True)

    net.fc = nn.Sequential(
        nn.Linear(2048, 512),
        nn.ReLU(inplace=True),
        nn.Dropout(0.6),

        nn.Linear(512, 128),
        nn.ReLU(inplace=True),
        nn.Dropout(0.5),
        nn.Linear(128, 10))

    return net


def resnet152():
    # 返回resnet152模型
    net = models.resnet152(pretrained=True)

    net.fc = nn.Sequential(
        nn.Linear(2048, 512),
        nn.ReLU(inplace=True),
        nn.Dropout(0.5),

        nn.Linear(512, 128),
        nn.ReLU(inplace=True),
        nn.Dropout(0.5),
        nn.Linear(128, 10))

    return net


def vgg13():
    # 返回vgg13模型
    net = models.vgg13(pretrained=True)

    net.fc = nn.Sequential(
        nn.Linear(25088, 4096, bias=True),
        nn.ReLU(inplace=True),
        nn.Dropout(0.5),

        nn.Linear(4096, 1024, bias=True),
        nn.ReLU(inplace=True),
        nn.Dropout(0.5),
        nn.Linear(1024, 10, bias=True))

    return net


def vgg16():
    # 返回vgg16模型
    net = models.vgg16(pretrained=True)

    net.fc = nn.Sequential(
        nn.Linear(2048, 256),
        nn.ReLU(inplace=True),
        nn.Dropout(0.6),
        nn.Linear(256, 10))

    return net


# define the CNN architecture
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.main = nn.Sequential(
            # 3 * 32 * 32
            nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),  # 32*16*16
            nn.BatchNorm2d(32),

            nn.Conv2d(32, 64, kernel_size=3, padding=1),  # 32*16*16
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),  # 64*8*8
            nn.BatchNorm2d(64),

            nn.Conv2d(64, 128, 3, padding=1),  # 64*8*8
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),  # 128*4*4
            nn.BatchNorm2d(128),
        )
        self.fc = nn.Sequential(
            nn.Linear(128 * 4 * 4, 1024),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),

            nn.Linear(1024, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),

            nn.Linear(256, 10)
        )

    def forward(self, x):
        # Conv and Poolilng layers
        x = self.main(x)

        # Flatten before Fully Connected layers
        x = x.view(-1, 128 * 4 * 4)

        # Fully Connected Layer
        x = self.fc(x)
        return x


def cnn():
    # 自己构建的常用神经网络
    net = CNN()

    return net