import os.path
import glob
from .util import split2list
from .listdataset import ListDataset
from random import shuffle

def make_dataset(input_dir,split):
    plyfiles = []

    for dirs in os.listdir(input_dir):
        tempDir = os.path.join(input_dir,dirs)
        for input in glob.iglob(os.path.join(tempDir,'*.npy')):
            input = os.path.basename(input)
            root_filename = input[:-4]
            plyinput = dirs + '/' + root_filename + '.npy'

            plyfiles.append([plyinput])

    if split== None:
        return plyfiles
    else:
        return split2list(plyfiles, split, default_split=split)

def shapenet(input_root, split, co_transforms= None, input_transforms = None,args=None,give_name=False):

    [train_list,valid_list] = make_dataset(input_root, split)

    train_dataset = ListDataset(input_root,train_list,co_transforms, input_transforms,args,mode='train',give_name=give_name)

    shuffle(valid_list)

    valid_dataset = ListDataset(input_root,valid_list,co_transforms, input_transforms,args,mode='valid',give_name=give_name)

    return  train_dataset,valid_dataset