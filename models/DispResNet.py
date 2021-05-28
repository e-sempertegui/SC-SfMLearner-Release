from __future__ import absolute_import, division, print_function
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from .resnet_encoder import *


import numpy as np
from collections import OrderedDict

class ConvBlock(nn.Module):
    """Layer to perform a convolution followed by ELU
    """
    def __init__(self, in_channels, out_channels):
        super(ConvBlock, self).__init__()

        self.conv = Conv3x3(in_channels, out_channels)
        self.nonlin = nn.ELU(inplace=True)

    def forward(self, x):
        out = self.conv(x)
        out = self.nonlin(out)
        return out

class Conv3x3(nn.Module):
    """Layer to pad and convolve input
    """
    def __init__(self, in_channels, out_channels, use_refl=True):
        super(Conv3x3, self).__init__()

        if use_refl:
            self.pad = nn.ReflectionPad2d(1)
        else:
            self.pad = nn.ZeroPad2d(1)
        self.conv = nn.Conv2d(int(in_channels), int(out_channels), 3)

    def forward(self, x):
        out = self.pad(x)
        out = self.conv(out)
        return out

def upsample(x):
    """Upsample input tensor by a factor of 2
    """
    return F.interpolate(x, scale_factor=2, mode="nearest")

class DepthDecoder(nn.Module):
    def __init__(self, num_ch_enc, scales=range(4), num_output_channels=1, use_skips=True):
        super(DepthDecoder, self).__init__()

        self.alpha = 10
        self.beta = 0.01

        self.num_output_channels = num_output_channels
        self.use_skips = use_skips
        self.upsample_mode = 'nearest'
        self.scales = scales

        self.num_ch_enc = num_ch_enc
        self.num_ch_dec = np.array([16, 32, 64, 128, 256])

        # decoder
        self.convs = OrderedDict()
        for i in range(4, -1, -1):
            # upconv_0
            num_ch_in = self.num_ch_enc[-1] if i == 4 else self.num_ch_dec[i + 1]
            num_ch_out = self.num_ch_dec[i]
            self.convs[("upconv", i, 0)] = ConvBlock(num_ch_in, num_ch_out)

            # upconv_1
            num_ch_in = self.num_ch_dec[i]
            if self.use_skips and i > 0:
                num_ch_in += self.num_ch_enc[i - 1]
            num_ch_out = self.num_ch_dec[i]
            self.convs[("upconv", i, 1)] = ConvBlock(num_ch_in, num_ch_out)

        for s in self.scales:
            self.convs[("dispconv", s)] = Conv3x3(self.num_ch_dec[s], self.num_output_channels)

        self.decoder = nn.ModuleList(list(self.convs.values()))
        self.sigmoid = nn.Sigmoid()

    def forward(self, input_features):
        self.outputs = []
        self.features = {}

        # decoder
        x = input_features[-1]
        for i in range(4, -1, -1):
            x = self.convs[("upconv", i, 0)](x)
            x = [upsample(x)]
            if self.use_skips and i > 0:
                x += [input_features[i - 1]]
            x = torch.cat(x, 1)
            x = self.convs[("upconv", i, 1)](x)
            if i in self.scales:
                self.outputs.append(self.alpha * self.sigmoid(self.convs[("dispconv", i)](x)) + self.beta)
                self.features[("features", i)] = x

        self.outputs = self.outputs[::-1]
        return self.outputs, self.features

class Uncertainty_Layers_Block(nn.Module):
    def __init__(self, num_ch_decoder, scales=range(4), num_output_channels=1):
        super(Uncertainty_Layers_Block, self).__init__()

        self.num_ch_dec = num_ch_decoder
        self.scales = scales
        self.num_output_channels = num_output_channels

        self.convs = OrderedDict()

        for s in self.scales:
            self.convs[("uncertconv", s)] = Conv3x3(self.num_ch_dec[s], self.num_output_channels)

        self.uncertainty_layers = nn.ModuleList(list(self.convs.values()))
    
    def forward(self, features_0, features_1, features_2, features_3):
        uncertainty_3 = self.convs[("uncertconv", 3)](features_3)
        uncertainty_2 = self.convs[("uncertconv", 2)](features_2)
        uncertainty_1 = self.convs[("uncertconv", 1)](features_1)
        uncertainty_0 = self.convs[("uncertconv", 0)](features_0)

        return uncertainty_0, uncertainty_1, uncertainty_2, uncertainty_3


class DispResNet(nn.Module):

    def __init__(self, num_layers = 18, pretrained = True, enable_uncert = False):
        super(DispResNet, self).__init__()
        self.enable_uncert = enable_uncert
        self.encoder = ResnetEncoder(num_layers = num_layers, pretrained = pretrained, num_input_images=1)
        self.decoder = DepthDecoder(self.encoder.num_ch_enc)
        if self.enable_uncert:
            print("Uncertainty Layers activated!")
            self.uncertainty_layers_block = Uncertainty_Layers_Block(self.decoder.num_ch_dec)

    def init_weights(self):
        pass

    def forward(self, x):
        features = self.encoder(x)
        outputs, features_dec = self.decoder(features)
        if self.enable_uncert:
            uncertainty_map, _, _, _ = self.uncertainty_layers_block(features_dec[("features", 0)], features_dec[("features", 1)], features_dec[("features", 2)], features_dec[("features", 3)])
            outputs.append(uncertainty_map)
            print("Added uncertainty map to disparity outputs list")
            print("Size of output list = ", len(outputs))
            
        if self.training:
            return outputs[:len(outputs)-1], outputs[4]
        else:
            return outputs[0]


if __name__ == "__main__":

    torch.backends.cudnn.benchmark = True

    model = DispResNet().cuda()
    model.train()

    B = 12

    tgt_img = torch.randn(B, 3, 256, 832).cuda()
    ref_imgs = [torch.randn(B, 3, 256, 832).cuda() for i in range(2)]

    tgt_depth = model(tgt_img)

    print(tgt_depth[0].size())


