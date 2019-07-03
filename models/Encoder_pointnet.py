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

__all__= ['encoder_pointnet']

class STN3d(nn.Module):
    def __init__(self, num_points = 2500):
        super(STN3d, self).__init__()
        self.num_points = num_points
        self.conv1 = torch.nn.Conv1d(3, 64, 1)
        self.conv2 = torch.nn.Conv1d(64, 128, 1)
        self.conv3 = torch.nn.Conv1d(128, 1024, 1)
        self.mp1 = torch.nn.MaxPool1d(num_points)
        self.amp1 = torch.nn.AdaptiveMaxPool1d(1)

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
        x = self.amp1(x)
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
        self.amp1 = torch.nn.AdaptiveMaxPool1d(1)

        self.num_points = num_points
        self.global_feat = global_feat


        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d) or isinstance(m, nn.Conv1d) or isinstance(m, nn.Linear):
                kaiming_normal(m.weight.data)  # initialize weigths with normal distribution
                if m.bias is not None:
                    m.bias.data.zero_()  # initialize bias as zero
            elif isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.BatchNorm1d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()


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
        x = self.amp1(x)
        x = x.view(-1, 1024)
        if self.global_feat:
            return x, trans
        else:
            x = x.view(-1, 1024, 1).repeat(1, 1, self.num_points)
            return torch.cat([x, pointfeat], 1), trans

class EncoderS(nn.Module):
    def __init__(self, num_points=2500, global_feat=True):
        super(EncoderS, self).__init__()
        self.conv1 = torch.nn.Conv1d(3, 64, 1)
        self.conv2 = torch.nn.Conv1d(64, 128, 1)
        self.conv3 = torch.nn.Conv1d(128, 256, 1)
        self.conv4 = torch.nn.Conv1d(256, 128, 1)

        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(256)
        self.bn4 = nn.BatchNorm1d(128)

        self.mp1 = torch.nn.MaxPool1d(num_points)
        self.amp1 = torch.nn.AdaptiveMaxPool1d(1)
        self.num_points = num_points
        self.global_feat = global_feat

        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d) or isinstance(m, nn.Conv1d) or isinstance(
                    m, nn.Linear):
                kaiming_normal(m.weight.data)  # initialize weigths with normal distribution
                if m.bias is not None:
                    m.bias.data.zero_()  # initialize bias as zero
            elif isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.BatchNorm1d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward(self, x):
        batchsize = x.size()[0]
        x = F.relu(self.bn1(self.conv1(x)))
        pointfeat = x
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = self.bn4(self.conv4(x))
        #x = self.mp1(x)
        x= self.amp1(x)
        x = x.view(-1, 128)
        if self.global_feat:
            return x,None
        else:
            x = x.view(-1, 1024, 1).repeat(1, 1, self.num_points)
            return torch.cat([x, pointfeat], 1)

class Encoder_pointnet(nn.Module):
    def __init__(self,args,num_points=2048,global_feat= True,calc_loss=True):
        super(Encoder_pointnet, self).__init__()
        self.encoder = EncoderS(num_points = num_points, global_feat = global_feat)
        self.mse_loss = nn.MSELoss()
        self.calc_loss = calc_loss

        for param in self.encoder.parameters():
            param.requires_grad = False

    def forward(self, x,encoder_pr=None):
       # with torch.no_grad():
            x = torch.squeeze(x,dim=1)
            x = torch.transpose(x,1,2)
            [encoder,_] = self.encoder(x)
            if self.calc_loss == True:
                loss = self.mse_loss(encoder,encoder_pr)
                return loss

            else:
                return encoder



def encoder_pointnet(args,num_points = 2048,global_feat = True,data=None,calc_loss = True):

    model = Encoder_pointnet(args,num_points,global_feat,calc_loss)
    model.encoder.load_state_dict(data['state_dict_encoder'])

    return model

