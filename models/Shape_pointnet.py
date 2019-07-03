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

__all__= ['shape_pointnet']

class STN3d(nn.Module):
    def __init__(self, num_points = 2500):
        super(STN3d, self).__init__()
        self.num_points = num_points
        self.conv1 = torch.nn.Conv1d(3, 64, 1)
        self.conv2 = torch.nn.Conv1d(64, 128, 1)
        self.conv3 = torch.nn.Conv1d(128, 1024, 1)
        self.mp1 = torch.nn.MaxPool1d(num_points)
        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 9)
        self.relu = nn.ReLU()

        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(1024)
        self.bn4 = nn.BatchNorm1d(512)
        self.bn5 = nn.BatchNorm1d(256)

        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d) or isinstance(m, nn.Conv1d) or isinstance(m, nn.Linear):
                kaiming_normal(m.weight.data)  # initialize weigths with normal distribution
                if m.bias is not None:
                    m.bias.data.zero_()  # initialize bias as zero
            elif isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.BatchNorm1d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

        # self.att = nn.Sequential(nn.Linear(1024,1024//16),
        #                          nn.ReLU(inplace=True),
        #                          nn.Linear(1024 // 16,1024 // 16),
        #                          nn.BatchNorm1d(1024 // 16),
        #                          nn.ReLU(inplace=True),
        #                          nn.Linear(1024 // 16,1024),
        #                          nn.Sigmoid(),
        #
        #                          )

    def forward(self, x):
        batchsize = x.size()[0]
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = self.mp1(x)
        x = x.view(-1, 1024)

        # x = self.att(x)

        x = F.relu(self.bn4(self.fc1(x)))
        x = F.relu(self.bn5(self.fc2(x)))
        x = self.fc3(x)

        iden = Variable(torch.from_numpy(np.array([1,0,0,0,1,0,0,0,1]).astype(np.float32))).view(1,9).repeat(batchsize,1)
        if x.is_cuda:
            iden = iden.cuda()
        x = x + iden
        x = x.view(-1, 3, 3)
        return x


class Encoder(nn.Module):
    def __init__(self, num_points = 2500, global_feat = True):
        super(Encoder, self).__init__()
        self.stn = STN3d(num_points = num_points)
        self.conv1 = torch.nn.Conv1d(3, 64, 1)
        self.conv2 = torch.nn.Conv1d(64, 128, 1)
        self.conv3 = torch.nn.Conv1d(128, 1024, 1)
        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(1024)
        self.mp1 = torch.nn.MaxPool1d(num_points)
        self.num_points = num_points
        self.global_feat = global_feat
    def forward(self, x):
        batchsize = x.size()[0]
        trans = self.stn(x)
        x = x.transpose(2,1)
        x = torch.bmm(x, trans)
        x = x.transpose(2,1)
        x = F.relu(self.bn1(self.conv1(x)))
        pointfeat = x
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.bn3(self.conv3(x))
        x = self.mp1(x)
        x = x.view(-1, 1024)
        if self.global_feat:
            return x, trans
        else:
            x = x.view(-1, 1024, 1).repeat(1, 1, self.num_points)
            return torch.cat([x, pointfeat], 1), trans


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


class Shape_pointnet(nn.Module):
    def __init__(self,args,num_points=2048,global_feat= True):
        super(Shape_pointnet, self).__init__()
        self.encoder = Encoder(num_points = num_points, global_feat = global_feat)
        self.decoder = Decoder(args)



        for param in self.decoder.parameters():
            param.requires_grad = False

    def forward(self, x):
        x = torch.squeeze(x)
        x = torch.transpose(x,1,2)
        [encoder,_] = self.encoder(x)
        #encoder_nograd = encoder.detach();
       # with torch.no_grad():
        decoder = self.decoder(encoder)



        return decoder[0], decoder[1], decoder[2], encoder



def shape_pointnet(args,num_points = 2048,global_feat = True,data=None,data_decoder=None):

    model= Shape_pointnet(args,num_points,global_feat)
    model.decoder.load_state_dict(data_decoder['state_dict_decoder'])
    model.encoder.load_state_dict(data_decoder['state_dict_encoder'])
    if data is not None:
        model.load_state_dict(data['state_dict'])
        model.decoder.load_state_dict(data_decoder['state_dict_decoder'])
    return model

