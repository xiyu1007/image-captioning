import torch
import torch.nn as nn

# 定义基本的残差块
class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_channels, out_channels, stride=1):
        """
        BasicBlock类用于定义ResNet中的基本残差块。

        参数:
        - in_channels (int): 输入通道数，表示输入特征图的深度。由上一层输出的通道数决定。
        - out_channels (int): 输出通道数，表示输出特征图的深度。决定了卷积核的数量。
        - stride (int): 步幅，控制卷积层的步长，默认为1。决定了卷积核的移动步长。

        注解:
        - 通道数的选择通常是由网络设计者根据任务需求和计算资源进行权衡的结果。
        - 步幅的选择影响了特征图的尺寸。stride=1时，特征图尺寸不变；stride=2时，尺寸减半。
        """
        super(BasicBlock, self).__init__()

        # 第一个卷积层（3x3卷积）
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)  # 批归一化层
        self.relu = nn.ReLU(inplace=True)  # ReLU激活函数

        # 第二个卷积层（3x3卷积）
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)  # 批归一化层

        # 恒等映射，用于保持输入和输出的尺寸一致
        self.downsample = nn.Sequential()
        if stride != 1 or in_channels != self.expansion * out_channels:
            # 如果步幅不为1或者输入通道数不等于self.expansion * out_channels（为了保持维度一致）
            self.downsample = nn.Sequential(
                # 创建一个包含一个1x1卷积层和一个批归一化层的序列
                nn.Conv2d(in_channels, self.expansion * out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion * out_channels)
            )

    def forward(self, x):
        """
        前向传播函数。

        参数:
        - x (torch.Tensor): 输入张量

        返回:
        - torch.Tensor: 输出张量
        """
        identity = x  # 保存输入，用于恒等映射

        # 第一层卷积
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        # 第二层卷积
        out = self.conv2(out)
        out = self.bn2(out)

        # 恒等映射，将输入加到输出上
        out += self.downsample(identity)
        out = self.relu(out)

        return out
