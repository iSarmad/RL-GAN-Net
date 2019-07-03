
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from models.misc.spectral import SpectralNorm
import numpy as np

__all__= ['critic_net']



class CriticNet(nn.Module):
    def __init__(self,args):
        super(CriticNet, self).__init__()
        state_dim = args.state_dim
        action_dim = args.z_dim

        self.l1 = nn.Linear(state_dim, 400)
        self.l2 = nn.Linear(400 + action_dim, 300)
        self.l3_additional = nn.Linear(300, 300)
        self.l3 = nn.Linear(300, 1)

    def forward(self, x, u):
        x = F.relu(self.l1(x))
        x = F.relu(self.l2(torch.cat([x, u], 1)))
        x = self.l3_additional(x)
        x = self.l3(x)
        return x





def critic_net(args,data=None):

    model = CriticNet(args)
    model.load_state_dict(data)

    return model