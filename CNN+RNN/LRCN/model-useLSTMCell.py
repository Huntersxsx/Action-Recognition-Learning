import torch
import torch.nn as nn
import torchvision


class LRCN(nn.Module):

    def __init__(self, input_dim, hidden_dim, num_layers, num_classes):
        super(LRCN, self).__init__()

        # Make sure that `hidden_dim` are lists having len == num_layers
        hidden_dim = self._extend_for_multilayer(hidden_dim, num_layers)
        if not len(hidden_dim) == num_layers:
            raise ValueError('Inconsistent list length.')

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.num_classes = num_classes
        cell_list = []
        for i in range(0, self.num_layers):
            cur_input_dim = self.input_dim if i == 0 else self.hidden_dim[i - 1]
            cell_list.append(nn.LSTMCell(cur_input_dim, self.hidden_dim[i]))
        self.cell_list = nn.ModuleList(cell_list)
        self.conv = nn.Sequential(*list(torchvision.models.resnet101().children())[:-1])
        self.fc = nn.Linear(self.hidden_dim[-1], self.num_classes)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x, hidden_state=None):

        B, C, L, H, W = x.size()
        if hidden_state is not None:
            raise NotImplementedError()
        else:
            hidden_state = self._init_hidden(batch_size=B)

        cur_layer_input = x

        for layer_idx in range(self.num_layers):
            h, c = hidden_state[layer_idx]
            output_inner = []
            for t in range(L):
                if layer_idx == 0:
                    cnn_feature = torch.squeeze(self.conv(cur_layer_input[:, :, t, :, :]))  # (B, 2048)
                    h, c = self.cell_list[layer_idx](cnn_feature, (h, c))
                else:
                    h, c = self.cell_list[layer_idx](cur_layer_input[:, t, :], (h, c))

                if self.num_layers == layer_idx + 1:
                    output_inner.append(self.fc(h))
                else:
                    output_inner.append(h)
            layer_output = torch.stack(output_inner, dim=1)
            cur_layer_input = layer_output

        output_prob = self.softmax(layer_output)
        avg_prob = torch.mean(output_prob, dim=1)

        return avg_prob

    def _init_hidden(self, batch_size):
        init_states = []
        for i in range(self.num_layers):
            init_states.append(
                [torch.zeros(batch_size, self.hidden_dim[i]), torch.zeros(batch_size, self.hidden_dim[i])])
        return init_states

    @staticmethod
    def _extend_for_multilayer(param, num_layers):
        if not isinstance(param, list):
            param = [param] * num_layers
        return param


if __name__ == "__main__":
    inputs = torch.rand(8, 3, 16, 112, 112)  # B * C * L * H * W
    LRCNmodel = LRCN(input_dim=2048, hidden_dim=2048, num_layers=2, num_classes=101)
    outputs = LRCNmodel(inputs)
    print(outputs.size())   # torch.Size([8, 101])
