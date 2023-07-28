import torch.nn as nn
import collections

class BasicBlock(nn.Module):
    def __init__(self, in_channels, out_channels, n_groups=8):
        super(BasicBlock, self).__init__()
        self.conv_init = nn.Conv3d(in_channels, out_channels, kernel_size=4, stride=2, padding=1, bias=False)
        self.gn1 = nn.GroupNorm(n_groups, out_channels)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv3d(out_channels, out_channels, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.gn2 = nn.GroupNorm(n_groups, out_channels)
        self.relu2 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv3d(out_channels, out_channels, kernel_size=(3, 3, 3), padding=(1, 1, 1))

    def forward(self, x):
        x = self.conv_init(x)
        residul = x
        x = self.relu1(self.gn1(x))
        x = self.conv1(x)
        x = self.relu2(self.gn2(x))
        x = self.conv2(x)
        x = x + residul

        return x


class Encoder3D(nn.Module):
    def __init__(self, cin=1, cout=6, nf=32, n_downsampling_3d=4):
        super(Encoder3D, self).__init__()
        self.network3d = collections.OrderedDict()
        # downsampling
        for ii in range(n_downsampling_3d):
            c_in = cin if ii == 0 else nf * (2 ** ii) // 2
            c_out = nf * (2 ** ii)
            self.network3d.update({'BasicBlock_' + str(ii): BasicBlock(c_in, c_out, n_groups=8 * (2 ** ii))})

        self.basic_block = nn.Sequential(self.network3d)

        self.n_layer_3d = len(self.network3d)
        self.avgpool = nn.AvgPool3d((1, 16, 16), stride=1)
        self.fc = nn.Linear(256, cout)

    def forward(self, x):
        x = self.basic_block(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        x = nn.Tanh()(x)
        return x.reshape(x.size(0), -1)


class Alignment(nn.Module):
    def __init__(self, is_train=True):
        super(Alignment, self).__init__()
        self.is_train = is_train

        # rotation angle and translation
        self.netV = Encoder3D(cin=1, cout=6, nf=32)

    def forward(self, x):
        view = self.netV(x)
        return view
