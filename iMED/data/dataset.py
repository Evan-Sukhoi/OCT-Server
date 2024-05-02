import torch
import numpy as np
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset
from torchvision.transforms import transforms


class CFDataset(Dataset):
    def __init__(self, file_path, input_height, input_width, use_oct=False):
        self.df = pd.read_csv(file_path)
        self.input_height = input_height
        self.input_width = input_width
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])
        self.use_oct = use_oct

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):

        proj_path = self.df.iloc[idx]['2D_data_path_cf']
        proj = Image.open(proj_path).resize((self.input_height, self.input_width))
        proj = np.array(proj, dtype=np.float32)
        channel = proj.shape[-1] + 1 if self.use_oct else proj.shape[-1]
        projs = np.zeros(
            (self.input_width, self.input_height, channel), dtype=np.float32)
        projs[:, :, 0:3] = proj[:, :, :]

        if self.use_oct:
            proj_path = self.df.iloc[idx]['2D_data_path_oct']
            proj = Image.open(proj_path).resize((self.input_height, self.input_width))
            proj = np.array(proj, dtype=np.float32)
            if len(proj.shape) > 2:
                projs[:, :, -1] = proj[:, :, 0]
            else:
                projs[:, :, -1] = proj

        projs = self.transform(projs)

        vol_path = self.df.iloc[idx]['3D_data_path']

        volume = np.load(vol_path)
        volume = torch.from_numpy(volume).float()

        return projs, volume
