import argparse
import os
import re

import numpy as np
from concurrent.futures import *
import pandas as pd
from PIL import Image
from utils.exception import DataDirNotDetectedError


def generate_data_path(path: str):
    pattern = r'^\d{4}_cut\.jpg$'
    names = path.split("\\")
    cf_path, oct_single_path, volume_path = None, None, None
    for filename in os.listdir(path):
        suffix = os.path.splitext(filename)[-1]
        if re.match(pattern, filename):
            cf_path = os.path.join(path, filename)
            new_path = os.path.join(path, '1.png')
            os.rename(cf_path, new_path)
            cf_path = new_path
        elif filename == '1.png':
            cf_path = os.path.join(path, filename)
        if filename == 'octscan_64.png':
          oct_single_path = os.path.join(path, filename)
        if suffix == '.fda_OCT_pngs':
            images_path = os.path.join(path, filename)
            volume = np.zeros((128, 128, 128), dtype=np.int64)
            numbers = []
            for i in range(128):
                numbers.append(i)
            for i in range(len(numbers)):
                oct_path = "octscan_%i.jpg" % (numbers[i] + 1)
                new_oct_path = "octscan_%i.png" % (numbers[i] + 1)
                oct_path = os.path.join(images_path, oct_path)
                new_oct_path = os.path.join(images_path, new_oct_path)
                try:
                    os.rename(oct_path, new_oct_path)
                except Exception as e:
                    pass
                oct_path = new_oct_path
                img = Image.open(oct_path)
                if np.array(img, dtype=np.float32).shape[-1] == 3:
                    img = img.resize((128, 128))
                    img = np.array(img, dtype=np.float32)
                    img = img[:, :, 0]
                else:
                    img = img.crop((0, 30, 512, 542))
                    img = img.resize((128, 128))
                    img = np.array(img, dtype=np.float32)

                volume[i, :, :] = img
            volume_path = os.path.join(path, "volume.npy")
            np.save(volume_path, volume)
    assert cf_path is not None
    assert volume_path is not None
    assert oct_single_path is not None
    return cf_path, oct_single_path, volume_path, names[-1]


class PathEntity:
    def __init__(self, data_path_2d_cf, data_path_2d_oct, data_path_3d, no):
        self.data_path_2d_cf = data_path_2d_cf
        self.data_path_2d_oct = data_path_2d_oct
        self.data_path_3d = data_path_3d
        self.no = no


def cmp_func(ele):
    return ele.no


def generate():
    path = os.getcwd()

    parser = argparse.ArgumentParser()
    parser.add_argument("--filename", "-f", default="/data/student/ljb/2D-OCT-to-3D-OCT-GAN-dp0.1-CFTIF/data/")
    args = parser.parse_args()

    data_dir = None
    if os.path.exists(args.filename):
        data_dir = os.listdir(args.filename)

    if data_dir is None:
        raise DataDirNotDetectedError("The directory for the data doesn't detect in the current directory!")

    ignore_dirs = ['cf', 'log']
    data_dir = [os.path.join(path, args.filename, sub_dir) for sub_dir in data_dir if sub_dir not in ignore_dirs]

    paths = []
    oct_data_paths, cf_data_paths, volume_data_paths = [], [], []

    with ThreadPoolExecutor(max_workers=40) as t:
        obj_list = []
        for sub_dir in data_dir:
            obj = t.submit(generate_data_path, sub_dir)
            obj_list.append(obj)

        for obj in as_completed(obj_list):
            result = obj.result()
            paths.append(PathEntity(result[0], result[1], result[2], result[3]))

    paths.sort(key=cmp_func)

    for p in paths:
        oct_data_paths.append(p.data_path_2d_oct)
        cf_data_paths.append(p.data_path_2d_cf)
        volume_data_paths.append(p.data_path_3d)

    data = {"2D_data_path_cf": cf_data_paths,
            "2D_data_path_oct": oct_data_paths,
            "3D_data_path": volume_data_paths}

    df = pd.DataFrame(data)
    data_path = os.path.join(path, "data_path.csv")
    df.to_csv(data_path, index=False)
    print("Successfully Generating the data!")