import os
import random
import numpy as np
from PIL import Image

import torch
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.sampler import Sampler
import torchvision.transforms as transforms

# Import settings
from settings import data_resize_shape, seed, dataset_type, model_prefix

# Import utils
from utils import Rotate, get_dataset_mean_std

# Ensure seed reproducibility
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)

class FewShotTask(object):
    def __init__(self, character_folders, num_classes, train_num, test_num):
        self.character_folders = character_folders
        self.num_classes = num_classes
        self.train_num = train_num
        self.test_num = test_num

        # Randomly select classes and assign labels.
        class_folders = random.sample(self.character_folders, self.num_classes)
        labels = dict(zip(class_folders, list(range(len(class_folders)))))
        samples = dict()

        self.train_roots = []
        self.test_roots = []
        for c in class_folders:
            temp = [os.path.join(c, x) for x in os.listdir(c)]
            samples[c] = random.sample(temp, len(temp))
            random.shuffle(samples[c]) 
            self.train_roots += samples[c][:train_num]
            self.test_roots += samples[c][train_num:train_num + test_num]
        self.train_labels = [labels[self.get_class(x)] for x in self.train_roots]
        self.test_labels = [labels[self.get_class(x)] for x in self.test_roots]

    def get_class(self, sample):
        # Assumes the class is the name of the folder containing the image.
        return os.path.join(*sample.split(os.sep)[:-1])

class FewShotDataset(Dataset):
    def __init__(self, task, split='train', transform=None, target_transform=None):
        self.transform = transform
        self.target_transform = target_transform
        self.task = task
        self.split = split
        if self.split == 'train':
            self.image_roots = self.task.train_roots
            self.labels = self.task.train_labels
        else:
            self.image_roots = self.task.test_roots
            self.labels = self.task.test_labels

    def __len__(self):
        return len(self.image_roots)

    def __getitem__(self, idx):
        raise NotImplementedError("This is an abstract class. Subclass it for your dataset.")

class ImageDataset(FewShotDataset):
    def __init__(self, *args, **kwargs):
        super(ImageDataset, self).__init__(*args, **kwargs)

    def __getitem__(self, idx):
        image_root = self.image_roots[idx]
        if dataset_type == 'bw':
            image = Image.open(image_root).convert('L')
            image = image.resize((data_resize_shape, data_resize_shape), resample=Image.LANCZOS)
        else:
            image = Image.open(image_root).convert('RGB')
            image = image.resize((data_resize_shape, data_resize_shape))

        if self.transform is not None:
            image = self.transform(image)
        label = self.labels[idx]
        if self.target_transform is not None:
            label = self.target_transform(label)
        return image, label

# -------------------------
#   Sampler Classes
# -------------------------

class ClassBalancedSamplerTrain(Sampler):
    def __init__(self, num_per_class, num_cl, num_inst, shuffle=True):
        self.num_per_class = num_per_class
        self.num_cl = num_cl
        self.num_inst = num_inst
        self.shuffle = shuffle

    def __iter__(self):
        if self.shuffle:
            batch = [[i + j * self.num_inst for i in torch.randperm(self.num_inst)[:self.num_per_class]]
                     for j in range(self.num_cl)]
        else:
            batch = [[i + j * self.num_inst for i in list(range(self.num_inst))[:self.num_per_class]]
                     for j in range(self.num_cl)]
        batch = [item for sublist in batch for item in sublist]
        if self.shuffle:
            random.shuffle(batch)
        return iter(batch)

    def __len__(self):
        return 1

class ClassBalancedSamplerOld(Sampler):
    def __init__(self, num_per_class, num_cl, num_inst, shuffle=True):
        self.num_per_class = num_per_class
        self.num_cl = num_cl
        self.num_inst = num_inst
        self.shuffle = shuffle

    def __iter__(self):
        if self.shuffle:
            batch = [[i + j * self.num_inst for i in torch.randperm(self.num_inst)[:self.num_per_class]]
                     for j in range(self.num_cl)]
        else:
            batch = [[i + j * self.num_inst for i in list(range(self.num_inst))[:self.num_per_class]]
                     for j in range(self.num_cl)]
        batch = [item for sublist in batch for item in sublist]
        if self.shuffle:
            random.shuffle(batch)
        return iter(batch)

    def __len__(self):
        return 1

class ClassBalancedSamplerTest(Sampler):
    def __init__(self, num_cl, num_inst, shuffle=True):
        self.num_cl = num_cl
        self.num_inst = num_inst
        self.shuffle = shuffle

    def __iter__(self):
        if self.shuffle:
            batches = [[i + j * self.num_inst for i in torch.randperm(self.num_inst)]
                       for j in range(self.num_cl)]
        else:
            batches = [[i + j * self.num_inst for i in list(range(self.num_inst))]
                       for j in range(self.num_cl)]
        batches = [[batches[j][i] for j in range(self.num_cl)] for i in range(self.num_inst)]
        if self.shuffle:
            random.shuffle(batches)
            for sublist in batches:
                random.shuffle(sublist)
        batches = [item for sublist in batches for item in sublist]
        return iter(batches)

    def __len__(self):
        return 1

# --------------
#   Data Loader 
# --------------

def get_data_loader(task, num_per_class=1, split='train', shuffle=False, use_old_sampler=False, rotation=0):
    # Get dataset mean and std for normalization purpouses 
    mean, std = get_dataset_mean_std(model_prefix)

    if dataset_type == 'rgb':
        transform = transforms.Compose([
            transforms.Resize((data_resize_shape, data_resize_shape)),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
        ])
    else:
        # black and white uses rotation
        transform = transforms.Compose([
            transforms.Resize((data_resize_shape, data_resize_shape)),
            Rotate(rotation),
            transforms.ToTensor(),
        ])

    dataset = ImageDataset(task, split=split, transform=transform)
    
    if dataset_type == 'rgb':
        if split == 'train':
            if use_old_sampler:
                sampler = ClassBalancedSamplerOld(num_per_class, task.num_classes, task.train_num, shuffle=shuffle)
            else:
                sampler = ClassBalancedSamplerTrain(num_per_class, task.num_classes, task.train_num, shuffle=shuffle)
            batch_size = num_per_class * task.num_classes
        else:
            if use_old_sampler:
                sampler = ClassBalancedSamplerTest(task.num_classes, task.test_num, shuffle=shuffle)
                batch_size = task.num_classes
            else:
                sampler = ClassBalancedSamplerTrain(num_per_class, task.num_classes, task.test_num, shuffle=shuffle)
                batch_size = num_per_class * task.num_classes
        loader = DataLoader(dataset, batch_size=batch_size, sampler=sampler)
    else:
        # black and white
        if split == 'train':
            sampler = ClassBalancedSamplerOld(num_per_class, task.num_classes, task.train_num,shuffle=shuffle)
        else:
            sampler = ClassBalancedSamplerOld(num_per_class, task.num_classes, task.test_num,shuffle=shuffle)

        loader = DataLoader(dataset, batch_size=num_per_class*task.num_classes, sampler=sampler)

    return loader

# ----------------------
#  Meta Folder Function
# ----------------------

def get_metafolders(data_dir, mode):
    """
    Returns the lists of folders for meta-training and meta-testing.
    In training mode: use subfolders "train" and "val".
    In testing mode: use subfolders "train" and "test".
    """
    train_folder = os.path.join(data_dir, "train")
    if mode == "train":
        test_folder = os.path.join(data_dir, "val")
    elif mode == "test":
        test_folder = os.path.join(data_dir, "test")
    else:
        raise ValueError("Unknown mode: choose 'train' or 'test'.")

    metatrain_folders = [os.path.join(train_folder, label)
                          for label in os.listdir(train_folder)
                          if os.path.isdir(os.path.join(train_folder, label))]
    metatest_folders = [os.path.join(test_folder, label)
                        for label in os.listdir(test_folder)
                        if os.path.isdir(os.path.join(test_folder, label))]
    random.shuffle(metatrain_folders)
    random.shuffle(metatest_folders)
    return metatrain_folders, metatest_folders

def get_omniglot_metafolders(data_dir):
    character_folders = [os.path.join(data_dir, family, character) \
                for family in os.listdir(data_dir) \
                if os.path.isdir(os.path.join(data_dir, family)) \
                for character in os.listdir(os.path.join(data_dir, family))]
    random.seed(1)
    random.shuffle(character_folders)
    
    # Custom split on omniglet as the original authors.
    num_train = 1200
    metatrain_character_folders = character_folders[:num_train]
    metaval_character_folders = character_folders[num_train:]

    return metatrain_character_folders,metaval_character_folders