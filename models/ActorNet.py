
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from models.misc.spectral import SpectralNorm
import numpy as np

__all__= ['actor_net']







class ActorNet(nn.Module):
    def __init__(self, args):
        super(ActorNet, self).__init__()
        state_dim = args.state_dim
        action_dim = args.z_dim
        max_action = args.max_action
        self.args =args
        self.l1 = nn.Linear(state_dim, 400)  # 400
        self.l2 = nn.Linear(400, 400)
        self.l2_additional = nn.Linear(400, 300)
        self.l3 = nn.Linear(300, action_dim)

        self.max_action = max_action

    def forward(self, x):
        x = torch.FloatTensor(x.reshape(1, -1)).to(self.args.device)
        x = F.relu(self.l1(x))
        x = F.relu(self.l2(x))
        x = F.relu(self.l2_additional(x))
        x = self.max_action * torch.tanh(self.l3(x))
        return x.cpu().data.numpy().flatten()





def actor_net(args,data=None):

    model = ActorNet(args)
    model.load_state_dict(data)

    return model