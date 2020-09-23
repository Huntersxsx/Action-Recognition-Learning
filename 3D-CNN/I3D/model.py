import torch
import torch.nn as nn


class Conv3dBlock(nn.Module):
    def __init__(self, in_channel, out_channel, kernel_size, stride, padding=0):
        super(Conv3dBlock, self).__init__()
        self.conv = nn.Conv3d(in_channel, out_channel, kernel_size=kernel_size, stride=stride, padding=padding)
        self.bn = nn.BatchNorm3d(out_channel, eps=0.001)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.bn(self.conv(x)))
        return x


class InceptionBlock(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(InceptionBlock, self).__init__()

        self.branch1 = Conv3dBlock(in_channel, out_channel[0], kernel_size=1, stride=1)
        self.branch2 = nn.Sequential(
            Conv3dBlock(in_channel, out_channel[1], kernel_size=1, stride=1),
            Conv3dBlock(out_channel[1], out_channel[2], kernel_size=3, stride=1, padding=1)
        )
        self.branch3 = nn.Sequential(
            Conv3dBlock(in_channel, out_channel[3], kernel_size=1, stride=1),
            Conv3dBlock(out_channel[3], out_channel[4], kernel_size=3, stride=1, padding=1)
        )
        self.branch4 = nn.Sequential(
            nn.MaxPool3d(kernel_size=3, stride=1, padding=1),
            Conv3dBlock(in_channel, out_channel[5], kernel_size=1, stride=1),
        )

    def forward(self, x):
        x1 = self.branch1(x)
        x2 = self.branch2(x)
        x3 = self.branch3(x)
        x4 = self.branch4(x)
        return torch.cat([x1, x2, x3, x4], dim=1)


class I3D(nn.Module):
    """
    Args:
        num_classes (int): Number of classes in the data
    """
    def __init__(self, num_classes):
        super(I3D, self).__init__()

        self.conv1 = Conv3dBlock(3, 64, kernel_size=7, stride=2, padding=3)
        self.pool1 = nn.MaxPool3d(kernel_size=(1, 3, 3), stride=(1, 2, 2), padding=(0, 1, 1))
        self.conv2 = Conv3dBlock(64, 64, kernel_size=1, stride=1)
        self.conv3 = Conv3dBlock(64, 192, kernel_size=3, stride=1, padding=1)
        self.pool2 = nn.MaxPool3d(kernel_size=(1, 3, 3), stride=(1, 2, 2), padding=(0, 1, 1))
        self.Inception1 = nn.Sequential(InceptionBlock(192, [64, 96, 128, 16, 32, 32]),
                                        InceptionBlock(256, [128, 128, 192, 32, 96, 64]))
        self.pool3 = nn.MaxPool3d(kernel_size=(3, 3, 3), stride=(2, 2, 2), padding=(1, 1, 1))
        self.Inception2 = nn.Sequential(InceptionBlock(480, [192, 96, 208, 16, 48, 64]),
                                        InceptionBlock(512, [160, 112, 224, 24, 64, 64]),
                                        InceptionBlock(512, [128, 128, 256, 24, 64, 64]),
                                        InceptionBlock(512, [112, 144, 288, 32, 64, 64]),
                                        InceptionBlock(528, [256, 160, 320, 32, 128, 128]))
        self.pool4 = nn.MaxPool3d(kernel_size=(2, 2, 2), stride=2)
        self.Inception3 = nn.Sequential(InceptionBlock(832, [256, 160, 320, 32, 128, 128]),
                                        InceptionBlock(832, [384, 192, 384, 48, 128, 128]))
        self.avg_pool = nn.AvgPool3d(kernel_size=(2, 7, 7))  # Default value of stride is kernel_size
        self.dropout = nn.Dropout(0.4)
        self.linear = nn.Linear(1024, num_classes)

    def forward(self, x):
        # input x: torch.Size([8, 3, 16, 224, 224])
        x = self.conv1(x)  # torch.Size([8, 64, 8, 112, 112])
        x = self.pool1(x)  # torch.Size([8, 64, 8, 56, 56])
        x = self.conv2(x)  # torch.Size([8, 64, 8, 56, 56])
        x = self.conv3(x)  # torch.Size([8, 192, 8, 56, 56])
        x = self.pool2(x)  # torch.Size([8, 192, 8, 28, 28])
        x = self.Inception1(x)  # torch.Size([8, 480, 8, 28, 28])
        x = self.pool3(x)  # torch.Size([8, 480, 4, 14, 14])
        x = self.Inception2(x)  # torch.Size([8, 832, 4, 14, 14])
        x = self.pool4(x)  # torch.Size([8, 832, 2, 7, 7])
        x = self.Inception3(x)  # torch.Size([8, 1024, 2, 7, 7])
        x = self.avg_pool(x)  # torch.Size([8, 1024, 1, 1, 1])
        x = self.dropout(x.view(x.size(0), -1))
        return self.linear(x)  # torch.Size([8, 101])


if __name__ == "__main__":
    inputs = torch.rand(8, 3, 16, 224, 224)  # B * C * L * H * W
    I3Dmodel = I3D(num_classes=101)
    outputs = I3Dmodel(inputs)
    print(outputs.size())   # torch.Size([8, 101])

