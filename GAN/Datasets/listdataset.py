import torch.utils.data as data
import os
import os.path
import numpy as np

def npy_loader(dir,file_name):
    path = os.path.join(dir,file_name)
    output = np.load(path)
    return output


class ListDataset(data.Dataset):

    def __init__(self, input_root, path_list, co_transforms = None, input_transforms = None,args=None,mode=None,give_name = False):
        self.input_root = input_root

        self.path_list = path_list
        self.loader = npy_loader
        self.input_transforms = input_transforms
        self.co_transforms = co_transforms
        self.args = args
        self.give_name =give_name
        self.mode = mode

    def __getitem__(self,index):
        inputs_list = self.path_list[index]

        input_name = inputs_list[0]
        input_name = input_name[:-4]

        inputs =  self.loader(self.input_root,inputs_list[0])

        if self.mode == 'train':
            if self.co_transforms is not None:
                inputs = self.co_transforms(inputs)

        if self.input_transforms is not None:
            inputs = self.input_transforms(inputs)

        if(self.give_name==True):
            return inputs,  input_name
        else:
            return inputs

    def __len__(self):
        return len(self.path_list)

