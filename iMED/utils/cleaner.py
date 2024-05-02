import argparse
import os
from concurrent.futures import *
from utils.exception import DataDirNotDetectedError


def clean_data_npy(path: str):
    for filename in os.listdir(path):
        if filename == 'volume.npy':
            filename = os.path.join(path, filename)
            os.remove(filename)


def clean():
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

    with ThreadPoolExecutor(max_workers=40) as t:
        obj_list = []
        for sub_dir in data_dir:
            obj = t.submit(clean_data_npy, sub_dir)
            obj_list.append(obj)
        for obj in as_completed(obj_list):
            obj.result()

    file_path = "data_path.csv"
    if os.path.exists(file_path):
        os.remove(file_path)
        print("Successfully delete {}!".format(file_path))

    print("Successfully delete all volume.npy files!")