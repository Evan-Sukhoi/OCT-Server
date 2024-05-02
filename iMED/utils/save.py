import os
import re
import numpy as np
from matplotlib import pyplot as plt


def generate_file_path(save_dir, **kwargs):
    file_path = save_dir
    for (key, value) in kwargs.items():
        tmp_path = "{}{}".format(str(key), str(value))
        file_path = os.path.join(file_path, tmp_path)
        if not str(value).endswith(".png") and not os.path.exists(file_path):
            os.mkdir(file_path)
    return file_path


def find_max_index(name, path):
    max_index = -1
    pattern = re.compile(rf"{re.escape(name)}(\d+)")
    for item in os.listdir(path):
        match = pattern.match(item)
        if match:
            index = int(match.group(1))
            max_index = max(max_index, index)
    return max_index

# get the save folder
def generate_folder(name, path, train=True):
    if name == 'None':
        if train:
            name = 'train'
        else:
            name = 'test'
    if not os.path.exists(path):
        os.makedirs(path)
    max_index = find_max_index(name, path)
    next_index = max_index + 1
    new_folder_name = f"{name}{next_index}"
    new_folder_path = os.path.join(path, new_folder_name)
    os.makedirs(new_folder_path, exist_ok=True)
    return new_folder_path


def save_comparison_images(output, target, mode, save_dir, **kwargs):
    output = np.array(output).astype(np.float64)
    target = np.array(target).astype(np.float64)
    for batch in range(len(output)):
        output_batch = output[batch]
        target_batch = target[batch]
        seq = range(output_batch.shape[0])
        for idx in seq:
            pd = output_batch[idx, :, :]
            gt = target_batch[idx, :, :]
            
            
            pd = np.array(pd).astype(np.float64)  # pd的shape是（128 128）
            gt = np.array(gt).astype(np.float64)  # gt的shape是（128 128）

            file_path_output = generate_file_path(save_dir=save_dir, mode=mode, **kwargs, batch=batch,
                                                  img="output_no{}.png".format(idx))
            file_path_target = generate_file_path(save_dir=save_dir, mode=mode, **kwargs, batch=batch,
                                                  img="target_no{}.png".format(idx))
            plt.imsave(file_path_output, pd, cmap='gray')
            plt.imsave(file_path_target, gt, cmap='gray')


def save_test_images(output, mode, save_dir):
    output = np.array(output).astype(np.float64)
    for batch in range(len(output)):
        output_batch = output[batch]
        seq = range(output_batch.shape[0])
        for idx in seq:
            pd = output_batch[idx, :, :]
                
            pd = np.array(pd).astype(np.float64)  # pd的shape是（128 128）
    
            file_path_output = generate_file_path(save_dir=save_dir,mode=mode, img="output_no{}.png".format(idx))
 
            plt.imsave(file_path_output, pd, cmap='gray')
