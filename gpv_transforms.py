import numpy as np
import torch
import torch.nn as nn

class Compose(object):
    def __init__(self,co_transforms):
        self.co_transforms = co_transforms

    def __call__(self, inputs):
        for transforms in self.co_transforms:
            inputs = transforms(inputs) #
        return inputs

class ArrayToTensor(object):
    def __call__(self,array):
        assert(isinstance(array,np.ndarray))
        #array = np.transpose(array, (2,0,1))
        # handle numpy array
        tensor = torch.from_numpy(array.copy())
        tensor = torch.unsqueeze(tensor,dim=0)
        return tensor.float()

class oneD2twoD(object):
    def __init__(self,img_size=32):
        self.img_size = img_size
    def __call__(self,inputs):

        inputs = np.reshape(inputs,(self.img_size,self.img_size))

        return inputs

