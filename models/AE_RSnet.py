import torch
import torch.nn as nn
from torch.nn.init import kaiming_normal
import math
#import torch.autograd.Variable

#from models.slice_pool_layer.slice_pool_layer import *
#from models.slice_unpool_layer.slice_unpool_layer import *



__all__= ['ae_rsnet']






class AE_RSnet(nn.Module):
    def __init__(self):
        super(AE_RSnet, self).__init__()

        # input: B, 1, N, 3

        # -- conv block 1
        self.conv_1 = nn.Conv2d(1, 64, kernel_size=(1, 3), stride=(1, 1))
        self.bn_1 = nn.BatchNorm2d(64)

        self.conv_2 = nn.Conv2d(64, 64, kernel_size=(1, 1), stride=(1, 1))
        self.bn_2 = nn.BatchNorm2d(64)

        self.conv_3 = nn.Conv2d(64, 64, kernel_size=(1, 1), stride=(1, 1))
        self.bn_3 = nn.BatchNorm2d(64)


        self.conv_6 = nn.Conv2d(64, 64, kernel_size=(1, 1), stride=(1, 1))
        self.bn_6 = nn.BatchNorm2d(64)
        self.pred_3 = nn.Conv2d(64, 3, kernel_size=(1, 1), stride=(1, 1))



        self.conv_7 = nn.Conv2d(64, 64, kernel_size=(1, 1), stride=(1, 1))
        self.bn_7 = nn.BatchNorm2d(64)
        self.pred_2 = nn.Conv2d(64, 3, kernel_size=(1, 1), stride=(1, 1))

        self.dp = nn.Dropout(p=0.3)

        self.conv_8 = nn.Conv2d(64, 3, kernel_size=(1, 1), stride=(1, 1))

        self.relu = nn.ReLU(inplace=True)


        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
                kaiming_normal(m.weight.data)  # initialize weigths with normal distribution
                if m.bias is not None:
                    m.bias.data.zero_()  # initialize bias as zero
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    # def weight_parameters(self):
    #     return [param for name,param in self.named_parameters() if 'weight' in name]
    #
    # def bias_parameters(self):
    #     return [param for name, param in self.named_parameters() if 'bias' in name]

    def forward(self, x):

        # -- conv block 1
        conv_1 = self.relu(self.bn_1(self.conv_1(x)))  # num_batch, 64, num_points, 1
        conv_2 = self.relu(self.bn_2(self.conv_2(conv_1)))  # num_batch, 64, num_points, 1
        conv_3 = self.relu(self.bn_3(self.conv_3(conv_2)))  # num_batch, 64, num_points, 1


        conv_6 = self.relu(self.bn_6(self.conv_6(conv_3)))  # num_batch, 512, num_points, 1
        pc_3 = self.pred_3(conv_6)
        pc_3 = torch.squeeze(pc_3)


        conv_7 = self.relu(self.bn_7(self.conv_7(conv_6)))  # num_batch, 256, num_points, 1

        droped = self.dp(conv_7)
        pc_2 = self.pred_2(droped)
        pc_2 = torch.squeeze(pc_2)

        conv_8 = self.conv_8(droped)
        pc_1 = torch.squeeze(conv_8)


        return pc_1, pc_2, pc_3


def ae_rsnet(data=None):

    model= AE_RSnet()

    if data is not None:
        model.load_state_dict(data['state_dict'])
    return model
