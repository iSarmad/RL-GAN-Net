
from parameter import *
from trainer import Trainer
# from tester import Tester
from data_loader import Data_Loader
from torch.backends import cudnn
from utils import make_folder
import torch
import gpv_transforms
import torchvision.transforms as transforms
import Datasets
from visualizer import Visualizer
from models.lossess import ChamferLoss
import models


def main(args):
    # For fast training

    cudnn.benchmark = True

    args.device = torch.device(
        "cuda:%d" % (args.gpu_id) if torch.cuda.is_available() else "cpu")  # for selecting device for chamfer loss
    torch.cuda.set_device(args.gpu_id)
    print('Using Tintan xp GPU : ', torch.cuda.current_device())

    # Data loader
    # data_loader = Data_Loader(args.train, args.dataset, args.image_path, args.imsize,
    #                          args.batch_size, shuf=args.train)



    """ -------------- Transfroms/ Data Augmentation-------------------------------"""

    co_transforms = gpv_transforms.Compose([
          gpv_transforms.oneD2twoD(img_size=args.imsize_new)
    ])

    input_transforms = transforms.Compose([
        gpv_transforms.ArrayToTensor(),
        #   transforms.Normalize(mean=[0.5,0.5],std=[1,1])
    ])


    """----------------Data Loader-----------------------------------------------"""

    [train_dataset, valid_dataset] = Datasets.__dict__[args.dataName](input_root=args.data,
                                                                      split=args.split_value,
                                                                      input_transforms=input_transforms,
                                                                      co_transforms=co_transforms)

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size,
                                               num_workers=args.num_workers,
                                               shuffle=True,
                                               pin_memory=True)
    valid_loader = torch.utils.data.DataLoader(valid_dataset,
                                               batch_size=args.batch_size,
                                               num_workers=args.num_workers,
                                               shuffle=False,
                                               pin_memory=True)

    """----------------------------------------------Model Settings--------------------------------------------------"""

    print('Model:', args.model_encoder)

    network_data = torch.load(args.pretrained)

    model_encoder = models.__dict__[args.model_encoder](args, num_points=2048, global_feat=True,
                                                        data=network_data, calc_loss=False).cuda()
    model_decoder = models.__dict__[args.model_decoder](args, data=network_data).cuda()

    params = get_n_params(model_encoder)
    print('| Number of Encoder parameters [' + str(params) + ']...')

    params = get_n_params(model_decoder)
    print('| Number of Decoder parameters [' + str(params) + ']...')

    """-------------------------------------------------Visualer Initialization-------------------------------------"""

    visualizer = Visualizer(args)

    args.display_id = args.display_id + 10
    args.name = 'Validation'
    vis_Valid = Visualizer(args)
    vis_Valida = []
    args.display_id = args.display_id + 10

    for i in range(1, 12):
        vis_Valida.append(Visualizer(args))
        args.display_id = args.display_id + 10

    chamfer = ChamferLoss(args)



    # Create directories if not exist
    make_folder(args.model_save_path, args.version)
    make_folder(args.sample_path, args.version)
    make_folder(args.log_path, args.version)
    make_folder(args.attn_path, args.version)



    if args.train:
        if args.model=='sagan':
            trainer = Trainer(None, args, train_loader,model_decoder,chamfer,vis_Valida) #data_loader.loader()
        elif args.model == 'qgan':
            trainer = qgan_trainer(None, args) # data_loader.loader()
        trainer.train()
    else:
        tester = Tester(data_loader.loader(), args, valid_loader)
        tester.test()

def get_n_params(model):
    pp=0
    for p in list(model.parameters()):
        nn=1
        for s in list(p.size()):
            nn= nn*s
        pp += nn
    return pp


if __name__ == '__main__':
    args = get_parameters()
    print(args)
    main(args)


