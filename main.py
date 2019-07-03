
import torch
import torch.utils.data
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import time
from models.lossess import ChamferLoss

import Datasets
import models

from collections import OrderedDict

import numpy as np
import os
import argparse
import datetime
import torchvision.transforms as transforms

import pc_transforms
from visualizer import Visualizer
from torch.autograd.variable import Variable
from tensorboardX import SummaryWriter
from utils import save_checkpoint,AverageMeter,get_n_params


np.random.seed(5)
torch.manual_seed(5)

dataset_names = sorted(name for name in Datasets.__all__)
model_names = sorted(name for name in models.__all__)

parser = argparse.ArgumentParser(description= 'Point Cloud Training Autoencoder and Shapecompletion Training on Three Datasets', formatter_class=argparse.ArgumentDefaultsHelpFormatter)

#arguments for Saving Models
parser.add_argument('--save_path',default='./ckpts', help='Path to Checkpoints')
parser.add_argument('--save',default= True,help= 'Save Models or not ?')
parser.add_argument('--pretrained',default=None,help= 'Use Pretrained Model for testing or resuming training') ## TODO
parser.add_argument('--test_only',default= False,help='Only Test the pre-trained Model')

# Arguments for Data Loader
#Path to train dataset # TODO
parser.add_argument('-d', '--data', metavar='DIR', default='', help='Path to Complete Point Cloud Data Set')
#Path to test dataset # TODO
parser.add_argument('-dw', '--datatest',  default='', help='Path to Complete Point Cloud Data Set')

parser.add_argument('-n', '--dataName', metavar='Data Set Name', default='shapenet', choices= dataset_names)
parser.add_argument('-ad', '--adddata', metavar='aDIR', default='', help='Additional path to dataset')
parser.add_argument('-s','--split_value',default = 0.95, help='Ratio of train and validation data split')

# Arguments for Torch Data Loader
parser.add_argument('-b','--batch_size', type=int, default=24, help='input batch size') #
parser.add_argument('-w','--workers',type=int, default=8, help='Set the number of workers')

# Arguments for Model Settings
parser.add_argument('-m','--model',default='ae_pointnet',help='Chose Your Model Here',choices=['ae_pointnet','ae_rsnet','shape_pointnet']) # TODO
parser.add_argument('-nt','--net_name',default='auto_encoder',help='Chose The name of your network',choices=['auto_encoder','shape_completion'])

# Optimizer Settings
parser.add_argument('-op','--optim',default= 'Adam',help='Specify the Optimizer to use')
parser.add_argument('--lr',default=0.001,help='Learning Rate for the optimizer') #
parser.add_argument('--momentum',default=0.9,help='Momentum for the adam optimizer')
parser.add_argument('--beta',default=0.999,help='beta for the adam optimizer')
parser.add_argument('--milestones',default=[60,120,180,500,800],help='For learning rate scheduler, will decay learning rate by gamma after each milestone')
parser.add_argument('--gamma',default=0.5,help='gamma for the learning rate scheduler')
parser.add_argument('--bias_decay',default=0,help='bias decay')

# Loss Settings
parser.add_argument('--gpu_id', type=int, default=1, help='gpu ids: e.g. 0, 1. -1 is no GPU')

# Training Settings

parser.add_argument('--epochs',default=400,help='Number of epochs to run')
parser.add_argument('--start_epoch',default=0,help='Starting Epoch')

# Visualizer Settings

parser.add_argument('--name', type=str, default='train',help='name of the experiment. It decides where to store samples and models')
parser.add_argument('--display_winsize', type=int, default=5, help='display window size')
parser.add_argument('--display_id', type=int, default=1001, help='window id of the web display')
parser.add_argument('--print_freq', type=int, default=200, help='Print Frequency')
parser.add_argument('--port_id', type=int, default=8102, help='Port id for browser')

parser.add_argument('--output_fc_pc_num', type=int, default=256, help='# of fc decoder output points')
parser.add_argument('--output_conv_pc_num', type=int, default=4096, help='# of conv decoder output points')
parser.add_argument('--feature_num', type=int, default=1024, help='length of encoded feature')
parser.add_argument('--activation', type=str, default='relu', help='activation function: relu, elu')
parser.add_argument('--normalization', type=str, default='batch', help='normalization function: batch, instance')



args = parser.parse_args()
args.device = torch.device("cuda:%d" % (args.gpu_id) if torch.cuda.is_available() else "cpu") # for selecting device for chamfer loss
#cuda.select_device
torch.cuda.set_device(args.gpu_id)
print('Using Tintan xp GPU : ',torch.cuda.current_device())



def main():


    """ Save Path """
    train_writer = None
    valid_writer = None
    test_writer = None

    if args.save == True:
        save_path = '{},{},{}epochs,b{},lr{}'.format(
            args.model,
            args.optim,
            args.epochs,
            args.batch_size,
            args.lr)
        time_stamp = datetime.datetime.now().strftime("%m-%d-%H:%M")
        save_path = os.path.join(time_stamp,save_path)
        save_path = os.path.join(args.dataName,save_path)
        save_path = os.path.join(args.save_path,save_path)
        print('==> Will save Everything to {}',save_path)

        if not os.path.exists(save_path):
            os.makedirs(save_path)

        #""" Setting for TensorboardX """

        train_writer = SummaryWriter(os.path.join(save_path, 'train'))
        valid_writer = SummaryWriter(os.path.join(save_path, 'valid'))
        test_writer  = SummaryWriter(os.path.join(save_path, 'test'))
       # output_writer = SummaryWriter(os.path.join(save_path, 'Output_Writer'))


    """ Transforms/ Data Augmentation Tec """
    co_transforms = pc_transforms.Compose([
       #  pc_transforms.Delete(num_points=1466)
       # pc_transforms.Jitter_PC(sigma=0.01,clip=0.05),
       # pc_transforms.Scale(low=0.9,high=1.1),
      #  pc_transforms.Shift(low=-0.01,high=0.01),
       # pc_transforms.Random_Rotate(),
      #  pc_transforms.Random_Rotate_90(),

       # pc_transforms.Rotate_90(args,axis='x',angle=-1.0),# 1.0,2,3,4
       # pc_transforms.Rotate_90(args, axis='z', angle=2.0),
       # pc_transforms.Rotate_90(args, axis='y', angle=2.0),
       # pc_transforms.Rotate_90(args, axis='shape_complete') TODO this is essential for Angela's data set
    ])

    input_transforms = transforms.Compose([

        pc_transforms.ArrayToTensor(),
     #   transforms.Normalize(mean=[0.5,0.5],std=[1,1])
    ])

    target_transforms = transforms.Compose([
        pc_transforms.ArrayToTensor(),
      #  transforms.Normalize(mean=[0.5, 0.5], std=[1, 1])
    ])


    """-----------------------------------------------Data Loader----------------------------------------------------"""


    if(args.net_name=='auto_encoder'):
        [train_dataset, valid_dataset] = Datasets.__dict__[args.dataName](input_root=args.data,
                                                                          target_root= None,
                                                                          split= args.split_value,
                                                                          net_name=args.net_name,
                                                                          input_transforms=input_transforms,
                                                                          target_transforms=target_transforms,
                                                                          co_transforms=co_transforms)
        [test_dataset, _] = Datasets.__dict__[args.dataName](input_root=args.datatest,
                                                                          target_root=None,
                                                                          split=None,
                                                                          net_name=args.net_name,
                                                                          input_transforms=input_transforms,
                                                                          target_transforms=target_transforms,
                                                                          co_transforms=co_transforms)
   

    train_loader = torch.utils.data.DataLoader(train_dataset,batch_size=args.batch_size,
                                               num_workers = args.workers,
                                               shuffle = True,
                                               pin_memory=True)
    valid_loader = torch.utils.data.DataLoader(valid_dataset,
                                               batch_size=args.batch_size,
                                               num_workers=args.workers,
                                               shuffle=False,
                                               pin_memory=True)
    test_loader = torch.utils.data.DataLoader(test_dataset,
                                               batch_size=args.batch_size,
                                               num_workers=args.workers,
                                               shuffle=False,
                                               pin_memory=True)

    """----------------------------------------------Model Settings--------------------------------------------------"""


    print('Model:',args.model)

    if args.pretrained:
        network_data = torch.load(args.pretrained)



        args.model = network_data['model']
        print("==> Using Pre-trained Model '{}' saved at {} ".format(args.model,args.pretrained))
    else:
        network_data = None

    if(args.model=='ae_pointnet'):
        model = models.__dict__[args.model](args, num_points = 2048, global_feat = True, data = network_data).cuda()
    else:
        model = models.__dict__[args.model](network_data).cuda()
  #  model = torch.nn.DataParallel(model.cuda(),device_ids=[0,1]) TODO To make dataparallel run do Nigels Fix """https://github.com/pytorch/pytorch/issues/1637#issuecomment-338268158"""



    params = get_n_params(model)
    print('| Number of parameters [' + str(params) + ']...')


    """-----------------------------------------------Optimizer Settings---------------------------------------------"""

    cudnn.benchmark = True
    print('Settings {} Optimizer'.format(args.optim))


    # param_groups = [{'params': model.module.bias_parameters(), 'weight_decay': args.bias_decay},
    #                  {'params': model.module.weight_parameters(), 'weight_decay':args.weight_decay}
    #                  ]
    if args.optim == 'Adam':
        optimizer = torch.optim.Adam(model.parameters(),lr=args.lr, betas=(args.momentum,args.beta))
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer,milestones=args.milestones,gamma=args.gamma)
   # scheduler =  torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer)

    """-------------------------------------------------Visualer Initialization-------------------------------------"""

    visualizer = Visualizer(args)

    args.display_id = args.display_id +10
    args.name = 'Validation'
    vis_Valid = Visualizer(args)
    vis_Valida = []
    args.display_id = args.display_id + 10

    for i in range(1,12):

        vis_Valida.append(Visualizer(args))
        args.display_id = args.display_id +10

    """---------------------------------------------------Loss Setting-----------------------------------------------"""

    chamfer = ChamferLoss(args)

    best_loss = -1
    valid_loss = 1000

    if args.test_only == True:
        epoch = 0
        test_loss, _, _ = test(valid_loader, model, epoch, args, chamfer, vis_Valid,vis_Valida,test_writer)
        test_writer.add_scalar('mean Loss', test_loss, epoch)

        print('Average Loss :{}'.format(test_loss))
    else:

        """------------------------------------------------Training and Validation-----------------------------------"""
        for epoch in range(args.start_epoch,args.epochs):

            scheduler.step()

            train_loss, _, _ = train(train_loader,model,optimizer,epoch,args,chamfer,visualizer,train_writer)
            train_writer.add_scalar('mean Loss',train_loss,epoch)


            valid_loss, _, _ = validation(test_loader,model,epoch,args,chamfer,vis_Valid,vis_Valida,valid_writer)
            valid_writer.add_scalar('mean Loss', valid_loss, epoch)

            if  best_loss < 0:
                best_loss = valid_loss

            is_best = valid_loss < best_loss

            best_loss = min(valid_loss,best_loss)

            if args.save == True:
                save_checkpoint({
                    'epoch':epoch +1,
                    'model' : args.model,
                    'state_dict': model.state_dict(), # TODO if data parallel is fized write model.module.state_dict()
                    'state_dict_encoder': model.encoder.state_dict(),
                    'state_dict_decoder': model.decoder.state_dict(),
                    'best_loss' : best_loss
                },is_best,save_path)


def test(valid_loader,model,epoch,args,chamfer,vis_Valid,vis_Valida,test_writer):
    batch_time = AverageMeter()
    lossess = AverageMeter()

    model.eval()
    end = time.time()
    epoch_size = len(valid_loader)
    j = 1;
    for i,(input) in enumerate(valid_loader):

        with torch.no_grad():

            input = input.cuda(async = True)

            input_var = Variable(input,requires_grad = True)

            #pc_1, pc_2, pc_3 = model(input_var)
            pc_1 = model(input_var)
            trans_input = torch.squeeze(input_var)
            trans_input = torch.transpose(trans_input, 1, 2)
            trans_input_temp = trans_input[1, :, :]
            pc_1_temp = pc_1[1, :, :]




        loss = chamfer(trans_input, pc_1) # instantaneous loss of batch items
        lossess.update(loss.item(),input.size(0)) # loss and batch size as input

        # measured elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % 30 == 0:
            print('Epoch: [{0}][{1}/{2}]\t Batch Time: {3} sec\t  Loss: {4}'.format(epoch, i,epoch_size, batch_time,loss))
            visuals = OrderedDict(
                [('Validation Input_pc', trans_input_temp.detach().cpu().numpy()),
                 ('Validation Predicted_pc', pc_1_temp.detach().cpu().numpy())])
            #vis_Valid.display_current_results(visuals, epoch, i)
            vis_Valida[j].display_current_results(visuals, epoch, i)
            j += 1

        errors = OrderedDict([('loss', loss.item())])  # plotting average loss
        vis_Valid.plot_current_errors(epoch, float(i) / epoch_size, args, errors)
        test_writer.add_scalar('test_loss', loss.item(), epoch)
    return lossess.avg, input_var, pc_1


def validation(valid_loader,model,epoch,args,chamfer,vis_Valid,vis_Valida,valid_writer):
    batch_time = AverageMeter()
    lossess = AverageMeter()

    model.eval()
    end = time.time()
    epoch_size = len(valid_loader)
    j = 1;
    for i,(input) in enumerate(valid_loader):

        with torch.no_grad():

            input = input.cuda(async = True)

            input_var = Variable(input,requires_grad = True)

            #pc_1, pc_2, pc_3 = model(input_var)
            pc_1 = model(input_var)

            trans_input = torch.squeeze(input_var,dim=1)
            trans_input = torch.transpose(trans_input, 1, 2)
            trans_input_temp = trans_input[0, :, :]
            pc_1_temp = pc_1[0, :, :]




        loss = chamfer(trans_input, pc_1) # instantaneous loss of batch items
        lossess.update(loss.item(),input.size(0)) # loss and batch size as input

        # measured elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % 20 == 0:
            print('Epoch: [{0}][{1}/{2}]\t Batch Time: {3} sec\t  Loss: {4}'.format(epoch, i,epoch_size, batch_time,loss))
            visuals = OrderedDict(
                [('Validation Input_pc', trans_input_temp.detach().cpu().numpy()),
                 ('Validation Predicted_pc', pc_1_temp.detach().cpu().numpy())])
            #vis_Valid.display_current_results(visuals, epoch, i)
            vis_Valida[j].display_current_results(visuals, epoch, i)
            j += 1

        errors = OrderedDict([('loss', loss.item())])  # plotting average loss
        vis_Valid.plot_current_errors(epoch, float(i) / epoch_size, args, errors)
        valid_writer.add_scalar('Valid_loss', loss.item(), epoch)

    return lossess.avg, input_var, pc_1


def train(train_loader,model,optimizer,epoch,args,chamfer,visualizer,train_writer):

    batch_time = AverageMeter()
    data_time = AverageMeter()
    lossess = AverageMeter()

    model.train()

    end = time.time()
    epoch_iter = 0
    for i, (input) in enumerate(train_loader):
        #measuring loading time
        epoch_iter += args.batch_size
        data_time.update(time.time() - end)

      #  target = target.cuda(async=True) #
        input = input.cuda(async=True)

        #input = [j.cuda() for j in input] """ For list mainly"""
        #target = [j.cuda() for j in target]

        input_var = Variable(input,requires_grad = True)
       # target_var = torch.autograd.Variable(target)

        trans_input = torch.squeeze(input_var)
        trans_input = torch.transpose(trans_input,1,2)

        #pc_1,pc_2,pc_3 = model(input_var)

        pc_1  = model(input_var)

        trans_input_temp = trans_input[1,:,:]
        pc_1_temp = pc_1[1,:,:]

        visuals = OrderedDict([('Train_input_pc', trans_input_temp.detach().cpu().numpy()), ('Train_predicted_pc', pc_1_temp.detach().cpu().numpy())])

        loss_1 = chamfer(trans_input, pc_1) #
       # loss_2 = chamfer(trans_input, pc_2)
      #  loss_3 = chamfer(trans_input, pc_3)



        loss = loss_1 #+ loss_2 + loss_3

        if not (epoch == 0 and i <= 20):
          lossess.update(loss.item(),input.size(0))
   #       errors = OrderedDict([('loss', loss.item()),('loss_1', loss_1.item()),('loss_2', loss_2.item()),('loss_3', loss_3.item())])
          errors = OrderedDict([('loss', loss.item()), ('loss_1', loss_1.item())])

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        #measured elapsed time
        batch_time.update(time.time() - end)
        end = time.time()
        epoch_size = len(train_loader)

        if i % args.print_freq == 0:
            print('Epoch: [{0}][{1}/{2}]\t Batch Time: {3} sec\t Data Load Time: {4} sec \t Loss{5}'.format(epoch,i,epoch_size,batch_time, data_time, loss))
            visualizer.display_current_results(visuals, epoch, i)
          #  output_writer.add_embedding(torch.transpose(trans_input_temp,0,1),global_step=epoch)
            if not (epoch == 0 and i <= 20):
                visualizer.plot_current_errors(epoch,float(i)/epoch_size,args,errors)
        
        train_writer.add_scalar('train_loss',loss.item(),epoch)

    return lossess.avg, input_var,  pc_1







if __name__=='__main__':
    main()