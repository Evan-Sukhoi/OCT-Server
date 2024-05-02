import math

import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader
from iMED.data.dataset import CFDataset
from iMED.model.net import *
from torchsummary import summary

import yaml


def load_train_val_dataloader(file_path, input_height, input_width, use_oct=False, batch_size=1, test_size=0.2, shuffle=False,
                              seed=None):
    # test_size是划分数据集为训练集和测试集的比例，test_size=0.2表示20%的数据为测试集
    dataset = CFDataset(file_path, input_height, input_width, use_oct)
    train_size = int(len(dataset) * (1 - test_size))
    test_size = len(dataset) - train_size
    if not shuffle:
        train_index = range(train_size)
        test_index = range(train_size, len(dataset))
        train_dataset = torch.utils.data.Subset(dataset, train_index)
        test_dataset = torch.utils.data.Subset(dataset, test_index)
    else:
        if seed is not None:
            train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size],
                                                                        generator=torch.Generator().manual_seed(seed))
        else:
            train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)
    input_size = dataset[0][0].shape
    return train_loader, test_loader, input_size


def load_model(c1=None, c2=None, n=None, input_size=(3, 128, 128), is_new=False):
    model = PatchGAN(c1, c2, n, is_new=is_new)
    if input_size is not None:
        summary(model, input_size=input_size, device='cpu')
    return model


def load_optim(optimizer_name: str, model: nn.Module, lr_0: float, *args):
    optimizer_class = getattr(optim, optimizer_name)
    if args[0] == 'None':
        optimizer = optimizer_class(model.parameters(), lr=lr_0)
    else:
        optimizer = optimizer_class(model.parameters(), lr=lr_0, *args)
    return optimizer


def load_lr_scheduler(optimizer: torch.optim.Optimizer, lr_f: float, epochs: int, step_size: int):
    gamma = math.pow(lr_f, step_size / epochs)
    s_optim = torch.optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)
    return s_optim


def load_config(yaml_file_path: str):
    with open(yaml_file_path, 'r') as file:
        data = yaml.safe_load(file)
    return data
