# Written by Muhammad Sarmad
# Date : 31st July
# This file is only to ascertain the value of ZMAX and ZMIN

import torch
import torch.utils.data
import torch.nn.parallel

import Datasets
import models

import numpy as np
import argparse
import torchvision.transforms as transforms

import pc_transforms

np.random.seed(2)
torch.manual_seed(2)

dataset_names = sorted(name for name in Datasets.__all__)
model_names = sorted(name for name in models.__all__)

parser = argparse.ArgumentParser(
    description='Point Cloud Training Autoencoder and Shapecompletion Training on Three Datasets',
    formatter_class=argparse.ArgumentDefaultsHelpFormatter)

# Arguments for Data Loader
parser.add_argument('-d', '--data', metavar='DIR',
                    default='/home/sarmad/Desktop/data/shape_net_core_uniform_samples_2048', help='Path to Data Set')
parser.add_argument('-n', '--dataName', metavar='Data Set Name', default='shapenet', choices=dataset_names)
parser.add_argument('-ad', '--adddata', metavar='aDIR', default='', help='Additional path to dataset')
parser.add_argument('-s', '--split_value', default=0.9, help='Ratio of train and test data split')

# Arguments for Torch Data Loader
parser.add_argument('-b', '--batch_size', type=int, default=8, help='input batch size')
parser.add_argument('-w', '--workers', type=int, default=8, help='Set the number of workers')

# Arguments for Model Settings
parser.add_argument('-m', '--model', default='AE_RSnet', help='Chose Your Model Here')
parser.add_argument('--bs', default=1.0, help='size of each block')

parser.add_argument('--stride', default=0.5, help='stride of block')
parser.add_argument('--rx', default=0.02, help='slice resolution in x axis')
parser.add_argument('--ry', default=0.02, help='slice resolution in y axis')
parser.add_argument('--rz', default=0.02, help='slice resolution in z axis')
parser.add_argument('--ZMAX', default=0.499, help='ZMax of the data')  # Check Zmax by running script zmax
parser.add_argument('--ZMIN', default=-0.499, help='ZMax of the data')  # Check Zmax by running script zmax

args = parser.parse_args()


def main():
    co_transform = pc_transforms.Compose([
        pc_transforms.ArrayToTensor(),
        transforms.Normalize(mean=[0.5, 0.5], std=[1, 1])
    ])

    input_transforms = transforms.Compose([
        pc_transforms.ArrayToTensor(),
        #   transforms.Normalize(mean=[0.5,0.5],std=[1,1])
    ])

    target_transforms = transforms.Compose([
        pc_transforms.ArrayToTensor(),
        #  transforms.Normalize(mean=[0.5, 0.5], std=[1, 1])
    ])

    """Data Loader"""
    #  x

    [train_dataset, valid_dataset] = Datasets.__dict__[args.dataName](input_root=args.data, target_root=None,
                                                                      split=args.split_value, net_name='auto_encoder',
                                                                      input_transforms=input_transforms,
                                                                      target_transforms=target_transforms)
    input, target = train_dataset[1]

    omax = 0.0
    omin = 0.0
    for i, (input_train, target) in enumerate(train_dataset):
        nmax = torch.max(input_train)
        nmax = np.max([torch.Tensor.numpy(nmax), omax])
        omax = nmax

        nmin = torch.min(input_train)
        nmin = np.min([torch.Tensor.numpy(nmin), omin])
        omin = nmin
    # 0.499
    # - 0.499

    for i, (input_valid, target) in enumerate(valid_dataset):
        nmax = torch.max(input_valid)
        nmax = np.max([torch.Tensor.numpy(nmax), omax])
        omax = nmax

        nmin = torch.min(input_valid)
        nmin = np.min([torch.Tensor.numpy(nmin), omin])
        omin = nmin

    print('ZMAX:',omax)
    print('ZMIN:', omin)

if __name__=='__main__':
    main()
