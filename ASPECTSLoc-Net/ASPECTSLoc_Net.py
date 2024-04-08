import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict
import torch.nn.init as init
import math
import os

import torchvision
import pretrainedmodels
import timm

class ASPECTSLoc_Net(nn.Module):
    def __init__(self):
        super(ASPECTSLoc_Net, self).__init__()
        self.model_name = "resnet50"
        input_channels = 3
        self.num_classes = 2
        self.drop_out = 0.

        in_features = 2048
        self.hidden_size = 512
        bidirectional = True
        recurrent_features = self.hidden_size * 2 if bidirectional else self.hidden_size
        self.num_recurrent_layers = 2
        dropout = 0
        num_classes = 2

        backbone = pretrainedmodels.__dict__[
            self.model_name](num_classes=1000, pretrained="imagenet")
        in_features = backbone.last_linear.in_features

        if hasattr(backbone, "layer0"):
            self.layer0 = backbone.layer0
        else:
            layer0_modules = [
                ('conv1', backbone.conv1),
                ('bn1', backbone.bn1),
                ('relu', backbone.relu),
                ('maxpool', backbone.maxpool)
            ]
            self.layer0 = nn.Sequential(OrderedDict(layer0_modules))
        self.layer1 = backbone.layer1
        self.layer2 = backbone.layer2
        self.layer3 = backbone.layer3
        self.layer4 = backbone.layer4

        del backbone

        self.fc = nn.Linear(in_features, self.num_classes)
        nn.init.zeros_(self.fc.bias.data)

        self.num_fea_conv_in = in_features
        self.num_fea_conv_out = 512
        self.ratio_fea_down = 1 / 4
        self.drop_out = 0.

        self.feature_conv = nn.Sequential(
            nn.Dropout2d(self.drop_out),
            nn.Conv2d(self.num_fea_conv_in, 512, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0), bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.Dropout2d(self.drop_out),
            nn.Conv2d(512, 128, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0), bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Dropout2d(self.drop_out)
        )

        self.feature_out = nn.Sequential(nn.Conv2d(128, self.num_classes, kernel_size=(1, 1), stride=(1, 1),
                                                   padding=(0, 0), bias=True))

        self.first_recurrent = nn.LSTM(input_size=2048,
                                       hidden_size=self.hidden_size,
                                       dropout=dropout,
                                       num_layers=self.num_recurrent_layers,
                                       bidirectional=bidirectional,
                                       batch_first=True)

        self.first_recurrent_final = nn.Sequential(nn.Conv2d(1, self.num_classes, kernel_size=(1, self.hidden_size * 2),
                                                             stride=(1, 1), padding=(0, 0), dilation=1, bias=True))

        ratio = 4
        self.conv_concat = nn.Sequential(
            nn.Conv2d(2, 128 * ratio, kernel_size=(5, 1),
                      stride=(1, 1), padding=(2, 0), dilation=1, bias=False),
            nn.BatchNorm2d(128 * ratio),
            nn.ReLU(),
            nn.Conv2d(128 * ratio, 64 * ratio, kernel_size=(3, 1),
                      stride=(1, 1), padding=(2, 0), dilation=2, bias=False),
            nn.BatchNorm2d(64 * ratio),
            nn.ReLU()
        )

        self.conv_multi_res = nn.Sequential(
            nn.Conv2d(64 * ratio, 64 * ratio, kernel_size=(3, 1),
                      stride=(1, 1), padding=(4, 0), dilation=4, bias=False),
            nn.BatchNorm2d(64 * ratio),
            nn.ReLU(),
            nn.Conv2d(64 * ratio, 64 * ratio, kernel_size=(3, 1),
                      stride=(1, 1), padding=(2, 0), dilation=2, bias=False),
            nn.BatchNorm2d(64 * ratio),
            nn.ReLU()
        )

        self.conv_final = nn.Sequential(
            nn.Conv2d(64*ratio, 1, kernel_size=(3, 1), stride=(1, 1),
                      padding=(1, 0), dilation=1, bias=False))

        self.second_recurrent = nn.LSTM(input_size=64 * ratio * 2,
                                        hidden_size=int(self.hidden_size / ratio),
                                        dropout=dropout,
                                        num_layers=self.num_recurrent_layers,
                                        bidirectional=bidirectional,
                                        batch_first=True)

        self.final = nn.Linear(int(self.hidden_size / ratio * 2), self.num_classes)
        nn.init.zeros_(self.final.bias.data)

    def _features(self, x):
        x = self.layer0(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        return x

    def forward(self, x, seq_len):
        fea = self._features(x)
        fea = F.adaptive_avg_pool2d(fea, 1)
        cnn_out = torch.flatten(fea, 1)
        cnn_out = self.fc(cnn_out)  
        cnn_out_sigmoid = cnn_out.clone()

        batch_size = int(fea.shape[0] / seq_len)
        fea = fea.view(batch_size, seq_len, -1).contiguous()
        fea = fea.view(batch_size, seq_len, -1, 1).contiguous()
        fea = fea.permute(0, 2, 1, 3).contiguous()

        fea_lstm = fea.view(batch_size, 2048, -1).contiguous()  
        fea_lstm = fea_lstm.permute(0, 2, 1).contiguous()  
        fea_lstm, _ = self.first_recurrent(fea_lstm)  
        fea_lstm = fea_lstm.view(batch_size, 1, -1, self.hidden_size * 2)
        fea_lstm_out = self.first_recurrent_final(fea_lstm)
        fea_lstm_out = fea_lstm_out.permute(0, 3, 2, 1)

        feature_out_sigmoid = fea_lstm_out.clone()
        cnn_out_sigmoid = cnn_out_sigmoid.view(batch_size, 1, seq_len, -1).contiguous()
        x = torch.cat([cnn_out_sigmoid, feature_out_sigmoid], dim=1)

        x = self.conv_concat(x)
        x = self.conv_multi_res(x)

        x = x.permute(0, 2, 1, 3).contiguous()
        x = x.view(batch_size, x.size()[1], -1).contiguous()
        x_lstm_out, _ = self.second_recurrent(x)

        out = self.final(x_lstm_out)
        out = out.view(-1, self.num_classes).contiguous()

        return out
