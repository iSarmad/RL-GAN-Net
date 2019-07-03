from __future__ import print_function
import argparse
import os
import random
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
import torchvision.transforms as transforms
import torchvision.utils as vutils
from torch.autograd import Variable
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import pdb
import torch.nn.functional as F
from .layers import *
from torch.nn.init import kaiming_normal

__all__= ['decoder_sonet']



class DecoderLinear(nn.Module):
    def __init__(self, opt):
        super(DecoderLinear, self).__init__()
        self.opt = opt
        self.feature_num = opt.feature_num
        self.output_point_number = opt.output_fc_pc_num

        self.linear1 = MyLinear(self.feature_num, self.output_point_number*2, activation=self.opt.activation, normalization=self.opt.normalization)
        self.linear2 = MyLinear(self.output_point_number*2, self.output_point_number*3, activation=self.opt.activation, normalization=self.opt.normalization)
        self.linear3 = MyLinear(self.output_point_number*3, self.output_point_number*4, activation=self.opt.activation, normalization=self.opt.normalization)
        self.linear_out = MyLinear(self.output_point_number*4, self.output_point_number*3, activation=None, normalization=None)

        # special initialization for linear_out, to get uniform distribution over the space
        self.linear_out.linear.bias.data.uniform_(-1, 1)




    def forward(self, x):
        # reshape from feature vector NxC, to NxC
        x = self.linear1(x)
        x = self.linear2(x)
        x = self.linear3(x)
        x = self.linear_out(x)

        return x.view(-1, 3, self.output_point_number)


class ConvToPC(nn.Module):
    def __init__(self, in_channels, opt):
        super(ConvToPC, self).__init__()
        self.in_channels = in_channels
        self.opt = opt

        self.conv1 = MyConv2d(self.in_channels, int(self.in_channels), kernel_size=1, stride=1, padding=0, bias=True, activation=opt.activation, normalization=opt.normalization)
        self.conv2 = MyConv2d(int(self.in_channels), 3, kernel_size=1, stride=1, padding=0, bias=True, activation=None, normalization=None)

        # special initialization for conv2, to get uniform distribution over the space
        # self.conv2.conv.bias.data.normal_(0, 0.3)
        self.conv2.conv.bias.data.uniform_(-1, 1)

        # self.conv2.conv.weight.data.normal_(0, 0.01)
        # self.conv2.conv.bias.data.uniform_(-3, 3)

    def forward(self, x):
        x = self.conv1(x)
        return self.conv2(x)


class DecoderConv(nn.Module):
    def __init__(self, opt):
        super(DecoderConv, self).__init__()
        self.opt = opt
        self.feature_num = opt.feature_num
        self.output_point_num = opt.output_conv_pc_num

        # __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=0, output_padding=0, bias=True, activation=None, normalization=None)
        # 1x1 -> 2x2
        self.deconv1 = UpConv(self.feature_num, int(self.feature_num), activation=self.opt.activation, normalization=self.opt.normalization)
        # 2x2 -> 4x4
        self.deconv2 = UpConv(int(self.feature_num), int(self.feature_num/2), activation=self.opt.activation, normalization=self.opt.normalization)
        # 4x4 -> 8x8
        self.deconv3 = UpConv(int(self.feature_num/2), int(self.feature_num/4), activation=self.opt.activation, normalization=self.opt.normalization)
        # 8x8 -> 16x16
        self.deconv4 = UpConv(int(self.feature_num/4), int(self.feature_num/8), activation=self.opt.activation, normalization=self.opt.normalization)
        self.conv2pc4 = ConvToPC(int(self.feature_num/8), opt)
        # 16x16 -> 32x32
        self.deconv5 = UpConv(int(self.feature_num/8), int(self.feature_num/8), activation=self.opt.activation, normalization=self.opt.normalization)
        self.conv2pc5 = ConvToPC(int(self.feature_num/8), opt)
        # 32x32 -> 64x64
        self.deconv6 = UpConv(int(self.feature_num/8), int(self.feature_num/8), activation=self.opt.activation, normalization=self.opt.normalization)
        self.conv2pc6 = ConvToPC(int(self.feature_num/8), opt)


    def forward(self, x):
        # reshape from feature vector NxC, to NxCx1x1
        x = x.view(-1, self.feature_num, 1, 1)
        x = self.deconv1(x)
        x = self.deconv2(x)
        x = self.deconv3(x)
        x = self.deconv4(x)
        self.pc4 = self.conv2pc4(x)
        x = self.deconv5(x)
        self.pc5 = self.conv2pc5(x)
        x = self.deconv6(x)
        self.pc6 = self.conv2pc6(x)

        return self.pc6


class Decoder(nn.Module):
    def __init__(self, opt):
        super(Decoder, self).__init__()
        self.opt = opt
        if self.opt.output_fc_pc_num > 0:
            self.fc_decoder = DecoderLinear(opt)
        self.conv_decoder = DecoderConv(opt)

    def forward(self, x):
        if self.opt.output_fc_pc_num > 0:
            self.linear_pc = self.fc_decoder(x)

        if self.opt.output_conv_pc_num > 0:
            self.conv_pc6 = self.conv_decoder(x).view(-1, 3, 4096)
            self.conv_pc4 = self.conv_decoder.pc4.view(-1, 3, 256)
            self.conv_pc5 = self.conv_decoder.pc5.view(-1, 3, 1024)

        if self.opt.output_fc_pc_num == 0:
            if self.opt.output_conv_pc_num == 4096:
                return self.conv_pc6
            elif self.opt.output_conv_pc_num == 1024:
                return self.conv_pc5
        else:
            if self.opt.output_conv_pc_num == 4096:
                return torch.cat([self.linear_pc, self.conv_pc6], 2), self.conv_pc5, self.conv_pc4
            elif self.opt.output_conv_pc_num == 1024:
                l = torch.cat([self.linear_pc, self.conv_pc5], 2), self.conv_pc4
                return l
            else:
                return self.linear_pc

class DecoderS(nn.Module):
    def __init__(self, opt):
        super(DecoderS, self).__init__()
        self.opt = opt
        self.feature_num = opt.feature_num
        self.output_point_number = opt.output_fc_pc_num

        self.linear1 = MyLinear(128, 256, activation=self.opt.activation, normalization=self.opt.normalization)
        self.linear2 = MyLinear(256, 256, activation=self.opt.activation, normalization=self.opt.normalization)
        self.linear_out = MyLinear(256, 6144, activation=None, normalization=None)

        # special initialization for linear_out, to get uniform distribution over the space
        self.linear_out.linear.bias.data.uniform_(-1, 1)

    def forward(self, x):
        # reshape from feature vector NxC, to NxC
        x = self.linear1(x)
        x = self.linear2(x)
        x = self.linear_out(x)

        return x.view(-1, 3, 2048)


class Decoder_sonet(nn.Module):
    def __init__(self,args):
        super(Decoder_sonet, self).__init__()
        self.decoder = DecoderS(args)

    def forward(self, x):
        decoder = self.decoder(x)
        return decoder





def decoder_sonet(args,data=None):

    model = Decoder_sonet(args)
    model.decoder.load_state_dict(data['state_dict_decoder'])

    return model