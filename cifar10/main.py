import matplotlib.pyplot as plt
import time
import numpy as np
import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import torch.optim as optim
import argparse
from models.nets import *


# 记录程序开始时间
start = time.time()
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

parser = argparse.ArgumentParser(description='Pytorch--Cifar10')
parser.add_argument('--model', type=str, default='cnn')
parser.add_argument('--batch_size', type=int, default=128)
parser.add_argument('--num_epochs', type=int, default=50)
parser.add_argument('--image_size', type=int, default=32)
args = parser.parse_args()


# load datasets
# define a transform to normalize the data

transform_train = transforms.Compose([
    transforms.Resize((args.image_size, args.image_size)),
    transforms.RandomCrop(args.image_size, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),  # converting images to tensor
    transforms.Normalize(mean=(0.4914, 0.4822, 0.4465), std=(0.2023, 0.1994, 0.2010))
    # transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
    # if the image dataset is black and white image, there can be just one number.
])

transform_test = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=(0.4914, 0.4822, 0.4465), std=(0.2023, 0.1994, 0.2010))
    # transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
])

# 制作数据集
train_dataset = datasets.CIFAR10('./data/CIFAR10/',
                                 train=True,
                                 download=False,
                                 transform=transform_train)

train_loader = DataLoader(train_dataset,
                          batch_size=args.batch_size,
                          shuffle=True,
                          num_workers=4)

val_dataset = datasets.CIFAR10('./data/CIFAR10/',
                               train=True,
                               transform=transform_test)

val_loader = DataLoader(val_dataset,
                        batch_size=args.batch_size,
                        shuffle=False,
                        num_workers=4)

test_dataset = datasets.CIFAR10('./data/CIFAR10/',
                                train=False,
                                transform=transform_test)

test_loader = DataLoader(test_dataset,
                         batch_size=args.batch_size,
                         shuffle=False)

# declare classes in CIFAR10
classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

# hyper parameters
learning_rate = 0.001
train_losses = []
val_losses = []

# 选择模型
if args.model == 'resnet18':
    model = resnet18().to(device)
elif args.model == 'resnet34':
    model = resnet34().to(device)
elif args.model == 'resnet50':
    model = resnet50().to(device)
elif args.model == 'resnet101':
    model = resnet101().to(device)
elif args.model == 'resnet152':
    model = resnet152().to(device)
elif args.model == 'vgg13':
    model = vgg13().to(device)
elif args.model == 'vgg16':
    model = vgg16().to(device)
else:
    model = cnn().to(device)


# loss function and optimizer
criterion = nn.CrossEntropyLoss()
# optimizer = optim.Adam(resnet.parameters(), lr=learning_rate)
optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9, weight_decay=5e-4)


# define train function that trains the model using a CIFAR10 dataset
def train(model, epoch, num_epochs):
    model.train()

    total_batch = len(train_dataset) // args.batch_size

    for i, (images, labels) in enumerate(train_loader):
        X = images.to(device)
        Y = labels.to(device)

        # forward pass
        pred = model(X)
        # calculation of loss value
        cost = criterion(pred, Y)

        # gradient initialization
        optimizer.zero_grad()
        # backward pass
        cost.backward()
        # parameters update
        optimizer.step()

        # training stats
        if (i + 1) % total_batch == 0:
            print(time.strftime('%Y-%m-%d %H:%M:%S', time.localtime()), end=" ")
            print('Train, Epoch [%d/%d], Loss: %.4f' % (epoch + 1, num_epochs, np.average(train_losses)))

            train_losses.append(cost.item())


# def the validation function that validates the model using CIFAR10 dataset
def validation(model, epoch, num_epochs):
    model.eval()
    total_batch = len(val_dataset) // args.batch_size

    for i, (images, labels) in enumerate(val_loader):
        X = images.to(device)
        Y = labels.to(device)

        with torch.no_grad():
            pred = model(X)
            cost = criterion(pred, Y)

        if (i + 1) % total_batch == 0:
            print(time.strftime('%Y-%m-%d %H:%M:%S', time.localtime()), end=" ")
            print("Validation, Epoch [%d/%d], Loss: %.4f"
                  % (epoch + 1, num_epochs, np.average(val_losses)))

            val_losses.append(cost.item())


# 测试精度
def test(model):
    #  declare that the model is about to evaluate
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in test_dataset:
            images = images.unsqueeze(0).to(device)

            # forward pass
            outputs = model(images)

            _, predicted = torch.max(outputs.data, 1)
            total += 1
            correct += (predicted == labels).sum().item()

    print("Accuracy of Test Images: %f %%" % (100 * float(correct) / total))


# 绘制损失函数
def plot_losses(train_losses, val_losses):
    plt.figure(figsize=(5, 5))
    plt.plot(train_losses, label='Train', alpha=0.5)
    plt.plot(val_losses, label="Validation", alpha=0.5)
    plt.xlabel('Epochs')
    plt.ylabel("losses")
    plt.legend()
    plt.grid(b=True)
    plt.title('CIFAR 10 Train/Val Losses Over Epoch')
    # plt.savefig('./images/resnet152.jpg')
    plt.show()


# 计算程序运行时间
def running_time(seconds):
    minutes, second = divmod(seconds, 60)
    hour, minute = divmod(minutes, 60)

    return hour, minute, second


def main():
    print("{}模型已准备好，开始训练".format(args.model))
    for epoch in range(args.num_epochs):
        train(model, epoch, args.num_epochs)
        validation(model, epoch, args.num_epochs)

    plot_losses(train_losses, val_losses)

    test(model=model)

    # 测试模型在哪些类上表现良好
    class_correct = list(0. for i in range(10))
    class_total = list(0. for i in range(10))

    # 分别测试各个数据的精度
    print("各个数据的精度：")
    with torch.no_grad():
        for data in test_loader:
            images, labels = data
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)

            _, predicted = torch.max(outputs, 1)
            c = (predicted == labels).squeeze()

            for i in range(4):
                label = labels[i]
                class_correct[label] += c[i].item()
                class_total[label] += 1

    for i in range(10):
        print('Accuracy of %5s : %2d %%' % (
            classes[i], 100 * class_correct[i] / class_total[i]))

    # 计算程序运行时间
    end = time.time()
    seconds = (end - start)
    hour, minute, second = running_time(seconds)
    print("程序运行时间为：{}小时{}分钟{:.2f}秒".format(hour, minute, second))


# 按间距中的绿色按钮以运行脚本。
if __name__ == '__main__':
    main()
