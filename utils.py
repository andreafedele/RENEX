import torch
import numpy as np
import scipy as sp

def mean_confidence_interval(data, confidence=0.95):
    a = 1.0 * np.array(data)
    n = len(a)
    m, se = np.mean(a), sp.stats.sem(a)
    h = se * sp.stats.t._ppf((1 + confidence) / 2.0, n - 1)
    return m, h

def init_device(no_cuda, no_mps):
    if not no_cuda and torch.cuda.is_available():
        return torch.device("cuda")
    elif not no_mps and torch.backends.mps.is_available():
        return torch.device("mps")
    else:
        return torch.device("cpu")
    
def get_dataset_mean_std(dataset_name):
    if dataset_name == 'CUB100':
        return [0.4675, 0.4677, 0.3942], [0.1965, 0.1939, 0.2007]
    elif dataset_name == 'CIFAR100':
        return [0.4584, 0.4977, 0.5183], [0.2103, 0.2013, 0.2024]
    elif dataset_name == 'Imagenet':
        return [0.4713, 0.4503, 0.4043], [0.2173, 0.2147, 0.2149]
    else:
        # print("Dataset Mean and Std Not Defined. Returning empty arrays!")
        return [], []

class Rotate(object):
    def __init__(self, angle):
        self.angle = angle
    def __call__(self, x, mode="reflect"):
        x = x.rotate(self.angle)
        return x