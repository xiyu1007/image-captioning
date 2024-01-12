import torch
import torch.nn as nn

from ResNet.Block.basicblock import BasicBlock


# 定义ResNet-18模型
class ResNet18(nn.Module):
    def __init__(self, num_classes=1000):
        """
        ResNet-18模型的定义。

        参数:
        - num_classes (int): 分类任务的类别数量，默认为1000。

        注解:
        - ResNet-18是一个包含四个阶段的深度卷积神经网络，用于图像分类任务。
        """
        super(ResNet18, self).__init__()
        self.in_channels = 64

        # 第一层：卷积、批归一化、ReLU激活、最大池化
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        # 四个阶段的残差块
        self.layer1 = self._make_layer(BasicBlock, 64, 2, stride=1)
        self.layer2 = self._make_layer(BasicBlock, 128, 2, stride=2)
        self.layer3 = self._make_layer(BasicBlock, 256, 2, stride=2)
        self.layer4 = self._make_layer(BasicBlock, 512, 2, stride=2)

        # 全局平均池化和全连接层
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * BasicBlock.expansion, num_classes)

    def _make_layer(self, block, out_channels, num_blocks, stride):
        """
        创建ResNet中的每个阶段。

        参数:
        - block (nn.Module): 使用的残差块类型（如BasicBlock或BottleNeck）。
        - out_channels (int): 输出通道数，表示输出特征图的深度。决定了卷积核的数量。
        - num_blocks (int): 每个阶段包含的残差块的数量。
        - stride (int): 步幅，控制卷积层的步长。决定了卷积核的移动步长。

        返回:
        - nn.Sequential: 包含所有残差块的序列。
        """
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_channels, out_channels, stride))
            self.in_channels = out_channels * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        """
        前向传播函数。

        参数:
        - x (torch.Tensor): 输入张量

        返回:
        - torch.Tensor: 输出张量
        """
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x


# 创建ResNet-18模型实例
resnet18 = ResNet18()

# 打印模型结构
print(resnet18)
