import os.path
import glob
from .util import split2list
from .listdataset import ListDataset
from random import shuffle

def make_dataset(input_dir,split,net_name,target_dir=None):
    plyfiles = []
    if(net_name== 'GAN'):
        for dirs in os.listdir(input_dir):
            tempDir = os.path.join(input_dir, dirs)
            for input in glob.iglob(os.path.join(tempDir, '*.npy')):
                input = os.path.basename(input)
                root_filename = input[:-4]
                plyinput = dirs + '/' + root_filename + '.npy'
                plyfiles.append([plyinput])



    if(net_name == 'auto_encoder'):
        target_dir = input_dir
        for dirs in os.listdir(target_dir):
            tempDir = os.path.join(input_dir,dirs)
            for target in glob.iglob(os.path.join(tempDir,'*.ply')):
                target = os.path.basename(target)
                root_filename = target[:-4]
                plytarget = dirs + '/' + root_filename + '.ply'

                plyinput = plytarget
                plyfiles.append([[plyinput],[plytarget]])

    if (net_name == 'shape_completion'): # TODO remove this sometime

        for dirs in os.listdir(input_dir):
            temp_In_Dir = os.path.join(input_dir, dirs)
            temp_Tgt_Dir = os.path.join(target_dir, dirs)

            for target in glob.iglob(os.path.join(temp_In_Dir, '*.ply')):
                target = os.path.basename(target)
                root_filename = target[:-9]
                plytarget = dirs + '/' + root_filename + '.ply'

                plyin = dirs + '/' + target

                plyfiles.append([[plyin], [plytarget]])

    if split== None:
        return plyfiles, plyfiles
    else:
        return split2list(plyfiles, split, default_split=split)

def shapenet(input_root, target_root, split, net_name='auto_encoder', co_transforms= None, input_transforms = None, target_transforms= None, args=None,give_name=False):

    [train_list,valid_list] = make_dataset(input_root, split,net_name, target_root)

    train_dataset = ListDataset(input_root,target_root,train_list,net_name, co_transforms, input_transforms, target_transforms,args,mode='train',give_name=give_name)

    shuffle(valid_list)

    valid_dataset = ListDataset(input_root,target_root,valid_list,net_name, co_transforms, input_transforms, target_transforms,args,mode='valid',give_name=give_name)

    return  train_dataset,valid_dataset