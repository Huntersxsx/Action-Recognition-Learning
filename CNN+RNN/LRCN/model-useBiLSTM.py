import torch
import torch.nn as nn
import torchvision


class LRCN(nn.Module):

    def __init__(self, input_dim, hidden_dim, num_layers, bidirectional, num_classes):
        super(LRCN, self).__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        self.num_classes = num_classes
        self.num_dirs = 2 if self.bidirectional else 1

        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers=num_layers, batch_first=True, bidirectional=bidirectional)
        self.conv = nn.Sequential(*list(torchvision.models.resnet101().children())[:-1])
        self.fc = nn.Linear(self.hidden_dim * self.num_dirs, self.num_classes)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x, hidden_state=None):

        B, C, L, H, W = x.size()
        if hidden_state is not None:
            raise NotImplementedError()
        else:
            hidden_state = self._init_hidden(batch_size=B)

        cnn_input = x.permute(0, 2, 1, 3, 4).contiguous().view(B*L, C, H, W)
        cnn_feature = self.conv(cnn_input).view(B, L, -1)
        lstm_output, _ = self.lstm(cnn_feature, hidden_state)

        avg_prob = self.softmax(self.fc(torch.mean(lstm_output, dim=1)))

        return avg_prob

    def _init_hidden(self, batch_size):
        h = torch.zeros(self.num_layers * self.num_dirs, batch_size, self.hidden_dim)  # h0
        c = torch.zeros(self.num_layers * self.num_dirs, batch_size, self.hidden_dim)  # c0
        return (h, c)


if __name__ == "__main__":
    inputs = torch.rand(8, 3, 16, 112, 112)  # B * C * L * H * W
    LRCNmodel = LRCN(input_dim=2048, hidden_dim=2048, num_layers=2, bidirectional=True, num_classes=101)
    outputs = LRCNmodel(inputs)
    print(outputs.size())   # torch.Size([8, 101])
