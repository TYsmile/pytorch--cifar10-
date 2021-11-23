pytorch--cifar10 
===
用pytorch框架实现所种网络模型在cifar10数据集上的精度测试。

实验记录
---
lr=0.001, Tesla P100 <br>

| Model | Accuracy | epoch | batch_size | image_size |
| ---- | ---- | ---- | ----- | ----- |
| cnn | 88.44% | 300 | 128 | 32x32 |
| vgg13 | 92.75% | 50 | 128 | 224x224 |
| vgg16 | 93.89% | 50 | 128 | 224x224 |
| ResNet18 | 87.75% | 300 | 128 | 32x32 |
| ResNet18 | 96.14%% | 300 | 128 | 224x224 |
| ResNet34 | 96.82% | 50 | 128 | 224x224 |
| ResNet50 | 97.01% | 50 | 128 | 224x224 |
| ResNet101 | 97.58% | 50 | 64 | 224x224 |
| ResNet152 | 97.83% | 30 | 64 | 224x224 |

<br> 代码可以直接运行，命令端模式: python main.py --batch_size=xx --num_epochs=xx --image_size=xx --model=''<br>
第一次运行程序需要设置download=True
