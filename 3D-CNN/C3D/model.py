import torch
import torch.nn as nn


class C3D(nn.Module):
    """
    The C3D network.
    Args:
        num_classes (int): Number of classes in the data
    """
    def __init__(self, num_classes):
        super(C3D, self).__init__()

        self.conv1 = nn.Conv3d(3, 64, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.pool1 = nn.MaxPool3d(kernel_size=(1, 2, 2), stride=(1, 2, 2))

        self.conv2 = nn.Conv3d(64, 128, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.pool2 = nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2))

        self.conv3a = nn.Conv3d(128, 256, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.conv3b = nn.Conv3d(256, 256, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.pool3 = nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2))

        self.conv4a = nn.Conv3d(256, 512, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.conv4b = nn.Conv3d(512, 512, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.pool4 = nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2))

        self.conv5a = nn.Conv3d(512, 512, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.conv5b = nn.Conv3d(512, 512, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.pool5 = nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2), padding=(0, 1, 1))

        self.fc6 = nn.Linear(8192, 4096)
        self.fc7 = nn.Linear(4096, 4096)
        self.fc8 = nn.Linear(4096, num_classes)

        self.softmax = nn.Softmax()
        self.dropout = nn.Dropout(p=0.5)
        self.relu = nn.ReLU()

    def forward(self, x):

        x = self.relu(self.conv1(x))  # L*112*112 --> L*112*112
        x = self.pool1(x)   # L*112*112 --> L*56*56

        x = self.relu(self.conv2(x))  # L*56*56 --> L*56*56
        x = self.pool2(x)  # L*56*56 --> (L/2)*28*28

        x = self.relu(self.conv3a(x))  # (L/2)*28*28 --> (L/2)*28*28
        x = self.relu(self.conv3b(x))  # (L/2)*28*28 --> (L/2)*28*28
        x = self.pool3(x)  # (L/2)*28*28 --> (L/4)*14*14

        x = self.relu(self.conv4a(x))  # (L/4)*14*14 --> (L/4)*14*14
        x = self.relu(self.conv4b(x))  # (L/4)*14*14 --> (L/4)*14*14
        x = self.pool4(x)  # (L/4)*14*14 --> (L/8)*7*7

        x = self.relu(self.conv5a(x))  # (L/8)*7*7 --> (L/8)*7*7
        x = self.relu(self.conv5b(x))  # (L/8)*7*7 --> (L/8)*7*7
        x = self.pool5(x)  # (L/8)*7*7 --> (L/16)*4*4

        x = x.view(-1, 8192)  # (B, (L/16)*4*4 * 512)  == (B, 8192)
        x = self.relu(self.fc6(x))  # (B, 4096)
        x = self.dropout(x)
        x = self.relu(self.fc7(x))  # (B, 4096)
        x = self.dropout(x)

        logits = self.fc8(x)  # (B, 101)
        logits = self.softmax(logits)

        return logits


if __name__ == "__main__":
    inputs = torch.rand(8, 3, 16, 112, 112)  # B * C * L * H * W
    C3Dmodel = C3D(num_classes=101)
    outputs = C3Dmodel(inputs)
    print(outputs.size())   # torch.Size([8, 101])

