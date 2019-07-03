import torch.utils.data as data
import os
import os.path
#from plyfile import PlyData, PlyElement
from Datasets.plyfile.plyfile import PlyData
import numpy as np
#import main import args as args

def load_ply(dir,file_name, with_faces=False, with_color=False):
    path = os.path.join(dir,file_name)
    ply_data = PlyData.read(path)
    points = ply_data['vertex']
    points = np.vstack([points['x'], points['y'], points['z']]).T
    ret_val = [points]

    if with_faces:
        faces = np.vstack(ply_data['face']['vertex_indices'])
        ret_val.append(faces)

    if with_color:
        r = np.vstack(ply_data['vertex']['red'])
        g = np.vstack(ply_data['vertex']['green'])
        b = np.vstack(ply_data['vertex']['blue'])
        color = np.hstack((r, g, b))
        ret_val.append(color)

    if len(ret_val) == 1:  # Unwrap the list
        ret_val = ret_val[0]

    return ret_val

def npy_loader(dir,file_name):
    path = os.path.join(dir,file_name)
    output = np.load(path)
    return output

class ListDataset(data.Dataset):

    def __init__(self, input_root,target_root, path_list, net_name, co_transforms = None, input_transforms = None, target_transforms = None,args=None,mode=None,give_name = False):
        self.input_root = input_root

        if net_name=='auto_encoder' : # As target root is same as input root for auto encoder
            self.target_root = input_root
        else:
            self.target_root = target_root

        self.path_list = path_list
        self.net_name = net_name
        if(self.net_name=='GAN'):
            self.loader = npy_loader
        else:
            self.loader = load_ply
        self.input_transforms = input_transforms
        self.target_transforms = target_transforms
        self.co_transforms = co_transforms
        self.args = args
        self.mode = mode
        self.give_name =give_name

    def __getitem__(self,index):
        inputs_list,targets_list = self.path_list[index]

        input_name = inputs_list[0]
        input_name = input_name[:-4]

        target_name = targets_list[0]
        target_name = target_name[:-4]

        inputs =  self.loader(self.input_root,inputs_list[0])
        targets = self.loader(self.target_root,targets_list[0])

        if self.mode == 'train':
            if self.co_transforms is not None:
                if self.net_name=='GAN':                                           # No target transform on GFV
                    inputs = self.co_transforms(inputs)
                else:
                    inputs,targets = self.co_transforms(inputs,targets)

        if self.input_transforms is not None:
                inputs = self.input_transforms(inputs)

        # if self.target_transforms is not None:
        #     targets = self.target_transforms(targets)

        if(self.give_name==True):
            return inputs, input_name
        else:
            return inputs

    def __len__(self):
        return len(self.path_list)

