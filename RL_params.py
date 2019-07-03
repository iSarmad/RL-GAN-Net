
import argparse
import torch
import torch.nn as nn
import torch.utils.data
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import time
from models.lossess import ChamferLoss, NLL, MSE, Norm


import Datasets
import models

from collections import OrderedDict

import numpy as np
import os
import argparse
import datetime
import torchvision.transforms as transforms

import gpv_transforms
import pc_transforms
from visualizer import Visualizer
from torch.autograd.variable import Variable
from tensorboardX import SummaryWriter
from utils import save_checkpoint,AverageMeter,get_n_params
from RL import TD3, OurDDPG, DDPG

import utils

import Datasets

import argparse

def str2bool(v):
    return v.lower() in ('true')

def get_parameters():
    dataset_names = sorted(name for name in Datasets.__all__)

    parser = argparse.ArgumentParser(description='RL Agent Training',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)


    # arguments for Saving Models
    parser.add_argument('--save_path', default='./RL_ckpt', help='Path to Checkpoints')
    parser.add_argument('--save', default=True, help='Save Models or not ?')  # TODO
  #  parser.add_argument('--pretrained_enc_dec',
  #                      default='/home/sarmad/PycharmProjects/pointShapeComplete/ckpts/shapenet/08-08-20:41/ae_pointnet,Adam,200epochs,b24,lr0.001/model_best.pth.tar',
  #                      help='Use Pretrained Encoder and Decoder for training')
    parser.add_argument('--pretrained_enc_dec',
                        default='/home/sarmad/PycharmProjects/pointShapeComplete/ckpts/shapenet/09-04-21:05/ae_pointnet,Adam,1000epochs,b50,lr0.0005/model_best.pth.tar',
                        help='Use Pretrained Model for Encoder and Decoder')
 #   parser.add_argument('--pretrained_enc_dec',
 #                       default='/home/sarmad/PycharmProjects/pointShapeComplete/ckpts/shapenet/09-12-21:00/ae_pointnet,Adam,1000epochs,b24,lr0.001/model_best.pth.tar',
 #                       help='Use Pretrained Model for testing or resuming training')

    parser.add_argument('--pretrained_G',
                        default='/media/sarmad/hulk/self attention GAN backup/models/sagan_celeb_1dim/999810_G.pth', #997920 24570  /home/sarmad/Desktop/GANs/Self-Attention-GAN-master/models/sagan_celeb   /media/sarmad/hulk/self attention GAN backup/models/sagan_celeb_1dim/999810_G.pth
                        help='Use Pretrained Generator')#/media/sarmad/hulk/self attention GAN backup/models/sagan_celeb_16 dim # /home/sarmad/Desktop/GANs/Self-Attention-GAN-master/models/sagan_celeb/999810_G.pth
    parser.add_argument('--pretrained_D',
                        default='/media/sarmad/hulk/self attention GAN backup/models/sagan_celeb_1dim/999810_D.pth',# 997920
                        help='Use Pretrained Discriminator')
    parser.add_argument('--pretrained_Actor',
                        default='/home/ymkim/ShapeCompletion/pointShapeComplete/pytorch_models/DDPG_RLGAN_actor.pth',  # 997920
                        help='Use Pretrained Actor')
    parser.add_argument('--pretrained_Critic',
                        default='/home/ymkim/ShapeCompletion/pointShapeComplete/pytorch_models/DDPG_RLGAN_critic.pth',  # 997920
                        help='Use Pretrained Critic')
    parser.add_argument('--test_only', default=False, help='Only Test the pre-trained Agent')


    # Data Loader settings
    parser.add_argument('-d', '--data', metavar='DIR',

                        default = '',# Add Path to Complete Train Data set
                        help='Path to Complete Point Cloud Data Set')
    parser.add_argument('--dataIncomplete', metavar='DIR',
                        default='',# Add Path to Incomplete Validation Data set
                        help='Path to Complete Point Cloud Data Set')
    parser.add_argument('--dataIncompLarge', metavar='DIR',
                        default='',# Add PAth to Incomplete Test Data set
                        help='Path to Incomplete Point Cloud Data Set')
    parser.add_argument('-s', '--split_value', default=0.9, help='Ratio of train and test data split')
    parser.add_argument('-n', '--dataName', metavar='Data Set Name', default='shapenet', choices=dataset_names)

    # Arguments for Torch Data Loader
    parser.add_argument('-b', '--batch_size', type=int, default=1, help='input batch size')
    parser.add_argument('-w', '--workers', type=int, default=8, help='Set the number of workers')


    # Hyper parameters for RL
    parser.add_argument('--attempts', default=5, type=int)  # Number of tries to give to RL Agent
    parser.add_argument("--policy_name", default="DDPG")  # Policy name TD3 OurDDPG
    parser.add_argument("--env_name", default="RLGAN")  # Policy name TD3 OurDDPG
    parser.add_argument("--state_dim", default=128, type=int)  # State Dimesnions #TODO equal to GFV dims
    parser.add_argument("--max_action", default=10, type=int)  # For Normal Distribution 2.5 is feasible ?

    parser.add_argument("--start_timesteps", default=1e4, # 1e4
                        type=int)  # How many time steps purely random policy is run for
    parser.add_argument("--eval_freq", default=5e3, type=float)  # How often (time steps) we evaluate
    parser.add_argument("--max_timesteps", default=1e6  , type=float)  # Max time steps to run environment for
    parser.add_argument("--expl_noise", default=0.1, type=float)  # Std of Gaussian exploration noise
    parser.add_argument("--save_models", default=True) # Save Pytorch Models?
    parser.add_argument("--batch_size_actor", default=100, type=int)  # Batch size for both actor and critic
    parser.add_argument("--discount", default=0.99, type=float)  # Discount factor
    parser.add_argument("--tau", default=0.005, type=float)  # Target network update rate
    parser.add_argument("--policy_noise", default=0.2, type=float)  # Noise added to target policy during critic update
    parser.add_argument("--noise_clip", default=0.5, type=float)  # Range to clip target policy noise
    parser.add_argument("--policy_freq", default=2, type=int)  # Frequency of delayed policy updates
    parser.add_argument("--max_episodes_steps", default=5, type=int)  # Frequency of delayed policy updates


    # Model Hype-Parameter
    parser.add_argument('--image_size', default=32, type=int)  # TODO original value 64
    parser.add_argument('--z_dim', type=int, default=1)#
    parser.add_argument('--g_conv_dim', type=int, default=64)
    parser.add_argument('--d_conv_dim', type=int, default=64)

    # Model Settings
    parser.add_argument('-nt', '--net_name', default='auto_encoder', help='Chose The name of your network',
                        choices=['auto_encoder', 'shape_completion'])
    parser.add_argument('--model_generator', default='self_gen_net', help='Chose Your Generator Model Here',
                        choices=['self_gen_net'])
    parser.add_argument('--model_discriminator', default='self_disc_net', help='Chose Your Discriminator Model Here',
                        choices=['self_disc_net'])
    parser.add_argument('--model_encoder', default='encoder_pointnet', help='Chose Your Encoder Model Here',
                        choices=['encoder_pointnet'])
    parser.add_argument('--model_decoder', default='decoder_sonet', help='Chose Your Decoder Model Here',
                        choices=['decoder_sonet'])

    # Visualizer Settings
    parser.add_argument('--name', type=str, default='train',
                        help='name of the experiment. It decides where to store samples and models')
    parser.add_argument('--display_winsize', type=int, default=256, help='display window size')
    parser.add_argument('--display_id', type=int, default=1001, help='window id of the web display')
    parser.add_argument('--print_freq', type=int, default=40, help='Print Frequency')
    parser.add_argument('--port_id', type=int, default=8102, help='Port id for browser')

    # Setting for Decoder
    # parser.add_argument('--output_pc_num', type=int, default=1280, help='# of output points')
    parser.add_argument('--output_fc_pc_num', type=int, default=256, help='# of fc decoder output points')
    parser.add_argument('--output_conv_pc_num', type=int, default=4096, help='# of conv decoder output points')
    parser.add_argument('--feature_num', type=int, default=1024, help='length of encoded feature')
    parser.add_argument('--activation', type=str, default='relu', help='activation function: relu, elu')
    parser.add_argument('--normalization', type=str, default='batch', help='normalization function: batch, instance')

    # GPU settings
    parser.add_argument('--gpu_id', type=int, default=0, help='gpu ids: e.g. 0, 1. -1 is no GPU')



    return parser.parse_args()