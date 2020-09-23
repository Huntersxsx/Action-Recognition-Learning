import torch.nn as nn
import torch

class Conv3dBlock(nn.Module):
    def __init__(self, in_channel, out_channel, kernel_size, stride, padding=0):
        super(Conv3dBlock, self).__init__()
        self.conv = nn.Conv3d(in_channel, out_channel, kernel_size=kernel_size, stride=stride, padding=padding)
        self.bn = nn.BatchNorm3d(out_channel, eps=0.001)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.bn(self.conv(x)))
        return x


class Conv3dBlockwoReLU(nn.Module):
    def __init__(self, in_channel, out_channel, kernel_size, stride, padding=0):
        super(Conv3dBlockwoReLU, self).__init__()
        self.conv = nn.Conv3d(in_channel, out_channel, kernel_size=kernel_size, stride=stride, padding=padding)
        self.bn = nn.BatchNorm3d(out_channel, eps=0.001)

    def forward(self, x):
        x = self.bn(self.conv(x))
        return x


class P3DBlock(nn.Module):
    """
    Args:
        BlockType (str): the type of Pseudo-3D blocks: A、B、C
        in_channel (int): Number of channels in the input tensor
        out_channel (int): Number of channels produced by the convolution
        stride (int or tuple, optional): Stride of the convolution. Default: 1
    """
    def __init__(self, BlockType, in_channel, out_channel, stride=1):
        super(P3DBlock, self).__init__()
        self.bottleneck_factor = 4
        self.BlockType = BlockType
        self.conv1 = Conv3dBlock(in_channel, out_channel, kernel_size=1, stride=1)
        if self.BlockType == 'A':
            self.conv2d = Conv3dBlock(out_channel, out_channel, kernel_size=(1, 3, 3),
                                      stride=(1, stride, stride), padding=(0, 1, 1))
            self.conv1d = Conv3dBlock(out_channel, out_channel, kernel_size=(3, 1, 1),
                                      stride=(stride, 1, 1), padding=(1, 0, 0))
        elif self.BlockType == 'B':
            self.conv2d = Conv3dBlock(out_channel, out_channel, kernel_size=(1, 3, 3),
                                      stride=stride, padding=(0, 1, 1))
            self.conv1d = Conv3dBlockwoReLU(out_channel, out_channel, kernel_size=(3, 1, 1),
                                      stride=stride, padding=(1, 0, 0))
        elif self.BlockType == 'C':
            self.conv2d = Conv3dBlock(out_channel, out_channel, kernel_size=(1, 3, 3),
                                      stride=stride, padding=(0, 1, 1))
            self.conv1d = Conv3dBlockwoReLU(out_channel, out_channel, kernel_size=(3, 1, 1),
                                      stride=1, padding=(1, 0, 0))
        else:
            raise ValueError('BlockType must be A, B or C.')

        self.conv3 = Conv3dBlockwoReLU(out_channel, out_channel * self.bottleneck_factor, kernel_size=1, stride=1)
        self.stride = stride
        self.relu = nn.ReLU()

        if self.stride != 1 or in_channel != out_channel * self.bottleneck_factor:
            self.downsample = Conv3dBlockwoReLU(in_channel, out_channel * self.bottleneck_factor,
                                                kernel_size=1, stride=stride)
        else:
            self.downsample = None

    def forward(self, x):
        f_branch = self.conv1(x)

        if self.BlockType == 'A':
            f_branch = self.conv2d(f_branch)
            f_branch = self.conv1d(f_branch)
        elif self.BlockType == 'B':
            f_branch2d = self.conv2d(f_branch)
            f_branch1d = self.conv1d(f_branch)
            f_branch = self.relu(f_branch1d + f_branch2d)
        elif self.BlockType == 'C':
            f_branch2d = self.conv2d(f_branch)
            f_branch1d = self.conv1d(f_branch2d)
            f_branch = self.relu(f_branch2d + f_branch1d)
        else:
            raise ValueError('BlockType must be A, B or C.')

        f_branch = self.conv3(f_branch)

        if self.downsample is not None:
            x = self.downsample(x)

        x = self.relu(x + f_branch)
        return x


class P3D (nn.Module):
    """
    Args:
        num_classes (int): Number of classes in the data
    """
    # input size: 16 x 160 x 160
    def __init__(self, num_classes):
        super(P3D, self).__init__()
        self.bottleneck_factor = 4
        self.conv1 = Conv3dBlock(3, 64, kernel_size=(1, 7, 7),
                                 stride=(1, 2, 2), padding=(0, 3, 3))
        self.maxpool = nn.MaxPool3d(kernel_size=(1, 3, 3), stride=(1, 2, 2), padding=(0, 1, 1))
        self.conv2 = nn.Sequential(P3DBlock('A', 64, 64, 2),
                                   P3DBlock('B', 64 * self.bottleneck_factor, 64),
                                   P3DBlock('C', 64 * self.bottleneck_factor, 64))
        self.conv3 = nn.Sequential(P3DBlock('A', 64 * self.bottleneck_factor, 128, 2),
                                   P3DBlock('B', 128 * self.bottleneck_factor, 128),
                                   P3DBlock('C', 128 * self.bottleneck_factor, 128),
                                   P3DBlock('A', 128 * self.bottleneck_factor, 128))
        self.conv4 = nn.Sequential(P3DBlock('B', 128 * self.bottleneck_factor, 256, 2),
                                   P3DBlock('C', 256 * self.bottleneck_factor, 256),
                                   P3DBlock('A', 256 * self.bottleneck_factor, 256),
                                   P3DBlock('B', 256 * self.bottleneck_factor, 256),
                                   P3DBlock('C', 256 * self.bottleneck_factor, 256),
                                   P3DBlock('A', 256 * self.bottleneck_factor, 256))
        self.conv5 = nn.Sequential(P3DBlock('B', 256 * self.bottleneck_factor, 512, 2),
                                   P3DBlock('C', 512 * self.bottleneck_factor, 512),
                                   P3DBlock('A', 512 * self.bottleneck_factor, 512))
        self.average_pool = nn.AvgPool3d(kernel_size=(1, 3, 3))
        self.fc = nn.Linear(512 * self.bottleneck_factor, num_classes)

    def forward(self, x):
        # input x: torch.Size([8, 3, 16, 224, 224])
        x = self.conv1(x)  # torch.Size([8, 64, 16, 112, 112])
        x = self.maxpool(x)  # torch.Size([8, 64, 16, 56, 56])
        x = self.conv2(x)  # torch.Size([8, 256, 8, 28, 28])
        x = self.conv3(x)  # torch.Size([8, 512, 4, 14, 14])
        x = self.conv4(x)  # torch.Size([8, 1024, 2, 7, 7])
        x = self.conv5(x)  # torch.Size([8, 2048, 1, 4, 4])
        x = self.average_pool(x)  # torch.Size([8, 2048, 1, 1, 1])
        x = x.view(x.size(0), -1)  # torch.Size([8, 2048])
        x = self.fc(x)  # torch.Size([8, 101])
        return x


if __name__ == "__main__":
    inputs = torch.rand(8, 3, 16, 224, 224)  # B * C * L * H * W
    P3Dmodel = P3D(num_classes=101)
    outputs = P3Dmodel(inputs)
    print(outputs.size())   # torch.Size([8, 101])
