
import torch
import torch.utils.data
import torch.nn.parallel
import time
from models.lossess import ChamferLoss

import Datasets
import models

from collections import OrderedDict

import numpy as np
import os
import argparse
import datetime

from visualizer import Visualizer
from torch.autograd.variable import Variable
from utils import  AverageMeter,get_n_params

np.random.seed(5)
torch.manual_seed(5)

dataset_names = sorted(name for name in Datasets.__all__)


parser = argparse.ArgumentParser(description= 'Point Cloud Training Autoencoder and Shapecompletion Training on Three Datasets', formatter_class=argparse.ArgumentDefaultsHelpFormatter)

#arguments for Saving Models
parser.add_argument('--save_path',default='./GFV', help='Path to Data Set')
parser.add_argument('--save',default= True,help= 'Save Models or not ?')#
#parser.add_argument('--pretrained',default='/home/sarmad/PycharmProjects/pointShapeComplete/ckpts/shapenet/08-08-20:41/ae_pointnet,Adam,200epochs,b24,lr0.001/model_best.pth.tar',help= 'Use Pretrained Model for testing or resuming training') ## TODO
#parser.add_argument('--pretrained',default='/home/sarmad/PycharmProjects/pointShapeComplete/ckpts/shapenet/09-04-21:05/ae_pointnet,Adam,1000epochs,b50,lr0.0005/model_best.pth.tar',help= 'Use Pretrained Model for testing or resuming training') ## TODO
parser.add_argument('--pretrained',default='/media/sarmad/hulk/pointShapeComplete/ckpts/shapenet/01-15-14:42/ae_pointnet,Adam,400epochs,b24,lr0.001/model_best.pth.tar',help= 'Use Pretrained Model for testing or resuming training') ## TODO


# Arguments for Model Settings
parser.add_argument('-me','--model_encoder',default='encoder_pointnet',help='Chose Your Encoder Model Here',choices=['encoder_pointnet']) # TODO
parser.add_argument('-md','--model_decoder',default='decoder_sonet',help='Chose Your Decoder Model Here',choices=['decoder_sonet']) # TODO
parser.add_argument('-nt','--net_name',default='auto_encoder',help='Choose The name of your network',choices=['auto_encoder']) #TODO


# Arguments for Data Loader
#  TODO Add Path to Training Data here
parser.add_argument('-d', '--data', metavar='DIR', default='', help='Path to Complete Point Cloud Data Set')
parser.add_argument('-s','--split_value',default = None, help='Ratio of train and test data split')
parser.add_argument('-n', '--dataName', metavar='Data Set Name', default='shapenet', choices= dataset_names)

# Arguments for Torch Data Loader
parser.add_argument('-b','--batch_size', type=int, default=1, help='input batch size')
parser.add_argument('-w','--workers',type=int, default=8, help='Set the number of workers')



# Visualizer Settings
parser.add_argument('--name', type=str, default='GFV',help='name of the experiment. It decides where to store samples and models')
parser.add_argument('--display_winsize', type=int, default=256, help='display window size')
parser.add_argument('--display_id', type=int, default=2000, help='window id of the web display')
parser.add_argument('--port_id', type=int, default=8099, help='Port id for browser')
parser.add_argument('--print_freq', type=int, default=10, help='Print Frequency')


# Setting for Decoder
#parser.add_argument('--output_pc_num', type=int, default=1280, help='# of output points')
parser.add_argument('--output_fc_pc_num', type=int, default=256, help='# of fc decoder output points')
parser.add_argument('--output_conv_pc_num', type=int, default=4096, help='# of conv decoder output points')
parser.add_argument('--feature_num', type=int, default=1024, help='length of encoded feature')
parser.add_argument('--activation', type=str, default='relu', help='activation function: relu, elu')
parser.add_argument('--normalization', type=str, default='batch', help='normalization function: batch, instance')




# GPU settings
parser.add_argument('--gpu_id', type=int, default=1, help='gpu ids: e.g. 0, 1. -1 is no GPU')





args = parser.parse_args()
args.device = torch.device("cuda:%d" % (args.gpu_id) if torch.cuda.is_available() else "cpu") # for selecting device for chamfer loss

torch.cuda.set_device(args.gpu_id)
print('Using Titian Xp GPu # :', torch.cuda.current_device())

def main():

     """------------------------------ Path to save the GFV files-------------------------------------------------- """


     if args.save == True:
           save_path = '{}'.format(
               args.model_encoder)
           time_stamp = datetime.datetime.now().strftime("%m-%d-%H:%M")
           save_path = os.path.join(time_stamp, save_path)
           save_path = os.path.join(args.dataName, save_path)
           save_path = os.path.join(args.save_path, save_path)
           print('==> Will save Everything to {}', save_path)

           if not os.path.exists(save_path):
                os.makedirs(save_path)

     """------------------------------------- Data Loader---------------------------------------------------------- """
     [train_dataset, valid_dataset] = Datasets.__dict__[args.dataName](input_root=args.data,
                                                                  target_root=None,
                                                                  split=args.split_value,
                                                                  net_name=args.net_name,
                                                                  input_transforms=None,
                                                                  target_transforms=None,
                                                                  co_transforms=None,
                                                                  give_name =True)


     train_loader = torch.utils.data.DataLoader(train_dataset,batch_size=args.batch_size,
                                               num_workers = args.workers,
                                               shuffle = True,
                                               pin_memory=True)

     valid_loader = torch.utils.data.DataLoader(valid_dataset,
                                               batch_size=args.batch_size,
                                               num_workers=args.workers,
                                               shuffle=False,
                                               pin_memory=True)

     """----------------------------------------------Model Settings--------------------------------------------------"""

     print('Model:', args.model_encoder)




     network_data = torch.load(args.pretrained)

     model_encoder = models.__dict__[args.model_encoder](args, num_points=2048, global_feat=True,
                                                         data=network_data,calc_loss = False).cuda()
     model_decoder = models.__dict__[args.model_decoder](args,data=network_data).cuda()

     params = get_n_params(model_encoder)
     print('| Number of Encoder parameters [' + str(params) + ']...')

     params = get_n_params(model_decoder)
     print('| Number of Decoder parameters [' + str(params) + ']...')

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

     chamfer = ChamferLoss(args)

     epoch = 0

     test_loss = test(train_loader,valid_loader,model_encoder,model_decoder,epoch,args,chamfer,vis_Valid,vis_Valida,save_path)

     print('Average Loss :{}'.format(test_loss))



def test(train_loader,valid_loader,model_encoder,model_decoder,epoch,args,chamfer,vis_Valid,vis_Valida,save_path):
    batch_time = AverageMeter()
    lossess = AverageMeter()

    model_encoder.eval()
    model_decoder.eval()



    end = time.time()




    print('==> Will save Validation Clean GFV to {}', save_path)

    if not os.path.exists(save_path):
        os.makedirs(save_path)

    epoch_size = len(valid_loader)

    j = 1;

    for i,(input,input_name) in enumerate(valid_loader):
        save_path_old = save_path

        input_name =input_name[0]
        save_path = os.path.join(save_path,input_name[:8])
        root_name = os.path.basename(input_name)

        if not os.path.exists(save_path):
             os.makedirs(save_path)

        save_file = os.path.join(save_path, root_name)

        with torch.no_grad():

            input = input.cuda(async = True)

            input_var = Variable(input,requires_grad = True)

            encoder_out = model_encoder(input_var,)

            np.save(save_file,encoder_out)

            load_file = save_file+'.npy'

            encoder_numpy = np.load(load_file)

            encoder_load = torch.tensor(encoder_numpy).cuda()

            #pc_1, pc_2, pc_3 = model_decoder(encoder_load)
            pc_1 = model_decoder(encoder_load)
            #trans_input = torch.squeeze(input_var)
            trans_input = torch.transpose(input_var, 1, 2)
            trans_input_temp = trans_input[0, :, :]
            pc_1_temp = pc_1[0, :, :]

        loss = chamfer(trans_input, pc_1)  # instantaneous loss of batch items
        lossess.update(loss.item(), input.size(0))  # loss and batch size as input

        # measured elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % 8000 == 0:
            print('Epoch: [{0}][{1}/{2}]\t Batch Time: {3} sec\t  Loss: {4}'.format(epoch, i,epoch_size, batch_time,loss))
            visuals = OrderedDict(
                [('Validation Input_pc', trans_input_temp.detach().cpu().numpy()),
                 ('Validation Predicted_pc', pc_1_temp.detach().cpu().numpy())])
            #vis_Valid.display_current_results(visuals, epoch, i)
            vis_Valida[j].display_current_results(visuals, epoch, i)
            j += 1

        errors = OrderedDict([('loss', loss.item())])  # plotting average loss
        vis_Valid.plot_current_errors(epoch, float(i) / epoch_size, args, errors)
        save_path = save_path_old
    return lossess.avg



if __name__ =='__main__':
    main()