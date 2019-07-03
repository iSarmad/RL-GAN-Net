import numpy as np
import torch
import torch.nn as nn

class Compose(object):
    def __init__(self,co_transforms):
        self.co_transforms = co_transforms

    def __call__(self, inputs, targets):
        for transforms in self.co_transforms:
            inputs,targets = transforms(inputs,targets) #
        return inputs,targets

class ArrayToTensor(object):
    def __call__(self,array):
        assert(isinstance(array,np.ndarray))
        #array = np.transpose(array, (2,0,1))
        # handle numpy array
        tensor = torch.from_numpy(array.copy())
        tensor = torch.unsqueeze(tensor,dim=0)
        return tensor.float()



class Jitter_PC(object):
    def __init__(self,sigma, clip):
        self.sigma = sigma
        self.clip = clip
        assert (clip > 0)

    def __call__(self,input,target):
        N,C = input.shape
        jittered_data_input = np.clip(self.sigma * np.random.randn(N, C), -1*self.clip, self.clip)
        jittered_data_input += input

        N,C = target.shape
        jittered_data_output = np.clip(self.sigma * np.random.randn(N, C), -1*self.clip, self.clip)
        jittered_data_output += target

        return jittered_data_input,jittered_data_output


class Scale(object):
    def __init__(self,low, high):
        self.low = low
        self.high = high

    def __call__(self,input,target):
        scale = np.random.uniform(low=self.low, high=self.high)

        input = input * scale
        target = target * scale

        return input, target


class Shift(object):
    def __init__(self,low, high):
        self.low = low
        self.high = high

    def __call__(self,input,target):

        shift = np.random.uniform(self.low, self.high,(1,3)) #

        input += shift
        target += shift

        return input, target


class Random_Rotate(object):
    def __call__(self,input,target):

        rotation_angle = np.random.uniform() * 2 * np.pi
        cosval = np.cos(rotation_angle)
        sinval = np.sin(rotation_angle)
        rotation_matrix = np.array([[cosval, 0, sinval],
                                    [0, 1, 0],
                                    [-sinval, 0, cosval]])
        rotated_input = np.dot(input.reshape((-1, 3)), rotation_matrix)

        rotated_target = np.dot(target.reshape((-1, 3)), rotation_matrix)

        return rotated_input, rotated_target

class Random_Rotate_90(object):

    def __call__(self,input,target):

        rotation_angle = np.random.randint(low=0, high=4) * (np.pi / 2.0)
        cosval = np.cos(rotation_angle)
        sinval = np.sin(rotation_angle)
        rotation_matrix = np.array([[cosval, 0, sinval],
                                    [0, 1, 0],
                                    [-sinval, 0, cosval]])
        rotated_input = np.dot(input.reshape((-1, 3)), rotation_matrix)

        rotated_target = np.dot(target.reshape((-1, 3)), rotation_matrix)

        return rotated_input, rotated_target


class Rotate_90(object):

    def __init__(self,args,axis,angle=1.0):
        self.angle = angle;
        self.args = args;
        self.axis = axis
    def __call__(self,input,target):
        if self.args.net_name == 'shape_completion':

            rotation_angle = self.angle * (np.pi / 2.0)
            cosval = np.cos(rotation_angle)
            sinval = np.sin(rotation_angle)
            if self.axis =='x':
                rotation_matrix = np.array([[1, 0, 0],
                                            [0, cosval, -sinval],
                                            [0, sinval, cosval]])

            if self.axis == 'y':
                rotation_matrix = np.array([[cosval, 0, sinval],
                                            [0, 1, 0],
                                            [-sinval, 0, cosval]])

            if self.axis == 'z':
                  rotation_matrix = np.array([[cosval, -sinval, 0],
                                            [sinval, cosval, 0],
                                            [0, 0, -1]])

            if self.axis == 'shape_complete':
                    rotation_matrix = np.array([[1.0, 0.0, 0.0],
                                            [0.0, 0.0, 1.0],
                                            [0.0, 1.0, 0.0]])
        # np.array([0.173178189568194, 0.378401247653964, - 0.909297426825682],
        #          [0.172881825917964, - 0.920591658450853, - 0.350175488374015],
        #          [0.969598467885110, 0.096558242344360, 0.224845095366153]])

            rotated_input = np.dot(input.reshape((-1, 3)), rotation_matrix)


            return rotated_input, target

        else:

            return input,target