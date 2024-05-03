import os
import numpy as np
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from skimage.metrics import normalized_root_mse as compare_nrmse
from skimage.metrics import structural_similarity as compare_ssim
from skimage.metrics import peak_signal_noise_ratio as compare_psnr
from PIL import Image
import subprocess

from scipy.ndimage import convolve
from scipy.stats import skew, kurtosis
import cv2

import lpips

from niqe import niqe

def find_png_files(directory):
    index = 0
    target_paths = []
    output_paths = []
    for root, dirs, files in os.walk(directory):
        for file in files:
            if (file.endswith('.png')):
                if file.startswith('imgtarget'):
                    new_name = 'target'+str(index)+'.png'
                    os.rename(os.path.join(root, file), os.path.join(root, new_name))
                    target_paths.append(os.path.join(root, new_name))
                if file.startswith('imgoutput'):
                    new_name = 'output'+str(index)+'.png'
                    os.rename(os.path.join(root, file), os.path.join(root, new_name))
                    output_paths.append(os.path.join(root, new_name))
                if file.startswith('target'):
                    target_paths.append(os.path.join(root, file))
                if file.startswith('output'):
                    output_paths.append(os.path.join(root, file))
                index += 1
    return target_paths, output_paths

# def get_error_metrics(im_pd, im_gt):
#     # print(im_pd.size)
#     im_pd = np.array(im_pd).astype(np.float64)
#     # print(im_pd.shape)
#     if np.max(im_pd) != 0:
#         im_pd = im_pd / np.max(im_pd)
#     im_gt = np.array(im_gt).astype(np.float64)
#     if np.max(im_gt) != 0:
#         im_gt = im_gt / np.max(im_gt)
#     im_pd = im_pd[0]
#     im_gt = im_gt[0]
#     size = im_pd.shape[0]
#     mse, mae, rmse, psnr, ssim = 0, 0, 0, 0, 0
#     for i in range(size):
#         pd = im_pd[i]
#         gt = im_gt[i]
#         # print(pd.shape, gt.shape)
#         assert (pd.flatten().shape == gt.flatten().shape)
#         mse_pred = mean_squared_error(y_true=gt.flatten(), y_pred=pd.flatten())
#         mae_pred = mean_absolute_error(y_true=gt.flatten(), y_pred=pd.flatten())
#         rmse_pred = compare_nrmse(image_true=gt, image_test=pd)
#         mse += mse_pred
#         mae += mae_pred
#         rmse += rmse_pred
#     mse = mean_squared_error(im_gt.flatten(), im_pd.flatten())
#     psnr = compare_psnr(im_gt, im_pd, data_range=1)
#     ssim = compare_ssim(im_gt, im_pd, data_range=1)
#     return mse, mae, rmse, psnr, ssim


def get_error_metrics(im_pd, im_gt):
    im_pd = np.array(im_pd).astype(np.float64).flatten()
    im_gt = np.array(im_gt).astype(np.float64).flatten()

    if np.max(im_pd) != 0:
        im_pd = im_pd / np.max(im_pd)
    if np.max(im_gt) != 0:
        im_gt = im_gt / np.max(im_gt)

    mse = mean_squared_error(im_gt.flatten(), im_pd.flatten())
    mae = mean_absolute_error(im_gt.flatten(), im_pd.flatten())
    rmse = np.sqrt(mse)
    psnr = compare_psnr(im_gt, im_pd, data_range=1)
    ssim_score, _ = compare_ssim(im_gt, im_pd, full=True, data_range=1)

    return mse, mae, rmse, psnr, ssim_score

target_paths, output_paths = find_png_files('D:\\2D-OCT-to-3D-OCT-GAN-dp0.1-CFTIF\\size128\\result\\modevalidation\\epoch299\\iter0\\batch0')

print(len(target_paths), len(output_paths))

# for i in range(len(target_paths)):
#     im_pd = Image.open(output_paths[i])
#     im_gt = Image.open(target_paths[i])
#     mse, mae, rmse, psnr, ssim = get_error_metrics(im_pd, im_gt)
#     print(
#         'img{i} mse: {mse_pred:.4f} | mae: {mae_pred:.4f} | rmse: {rmse_pred:.4f} |'
#         ' psnr: {psnr_pred:.4f} | ssim: {ssim_pred:.4f}'.format(i=i, mse_pred=mse, mae_pred=mae, rmse_pred=rmse, psnr_pred=psnr, ssim_pred=ssim))

result_enhanced_folder = "D:\\Projects\\Real-ESRGAN\\realesrgan\\evaluation\\results_enhanced"
target_enhanced_folder = "D:\\Projects\\Real-ESRGAN\\realesrgan\\evaluation\\targets_enhanced"

# for i in range(len(target_paths)):
#     result_file_path = output_paths[i]
#     target_file_path = target_paths[i]
#     command_1 = 'python inference_realesrgan.py --model_path D:\\Projects\\Real-ESRGAN\experiments\\train_RealESRNetx4plus_1000k_B12G4\\models\\net_g_795000.pth -i ' + result_file_path +' -o ' + result_enhanced_folder + ' --fp32 -s 4'
#     command_2 = 'python inference_realesrgan.py --model_path D:\\Projects\\Real-ESRGAN\experiments\\train_RealESRNetx4plus_1000k_B12G4\\models\\net_g_795000.pth -i ' + target_file_path +' -o ' + target_enhanced_folder + ' --fp32 -s 4'
#     subprocess.call(command_1, shell=True)
#     subprocess.call(command_2, shell=True)


def get_error_metrics_folder(target_paths, output_paths):
    mse_list = []
    mae_list = []
    rmse_list = []
    psnr_list = []
    ssim_list = []
    niqe_list = []
    niqe_gt_list = []
    lpips_list = []
    for i in range(len(target_paths)):
        im_pd = Image.open(output_paths[i]).convert('L').resize((128, 128))
        im_gt = Image.open(target_paths[i]).convert('L').resize((128, 128))
        mse, mae, rmse, psnr, ssim = get_error_metrics(im_pd, im_gt)
        loss_fn = lpips.LPIPS(net='vgg').cuda()
        lpips_dist = loss_fn(im_pd, im_gt)
        niqe_score = niqe(np.array(im_pd))
        niqe_score_gt = niqe(np.array(im_gt))
        mse_list.append(mse)
        mae_list.append(mae)
        rmse_list.append(rmse)
        psnr_list.append(psnr)
        ssim_list.append(ssim)
        niqe_list.append(niqe_score)
        niqe_gt_list.append(niqe_score_gt)
        lpips_list.append(lpips_dist.item())

    print(
        'mse: {mse_avg:.4f} | mae: {mae_avg:.4f} | rmse: {rmse_avg:.4f} |'
        ' psnr: {psnr_avg:.4f} | ssim: {ssim_avg:.4f} | niqe: {niqe_avg:.4f} | niqe_gt: {niqe_gt_avg:.4f} | '
        'lpips: {lpips:.4f}'.format(mse_avg=np.mean(mse_list),
                                                                mae_avg=np.mean(mae_list),
                                                                rmse_avg=np.mean(rmse_list),
                                                                psnr_avg=np.mean(psnr_list),
                                                                ssim_avg=np.mean(ssim_list),
                                                                niqe_avg=np.mean(niqe_list),
                                                                niqe_gt_avg=np.mean(niqe_gt_list),
                                                                lpips=np.mean(lpips_list)))
    return mse_list, mae_list, rmse_list, psnr_list, ssim_list, niqe_list


output_file_path = 'output.txt'
mse_list, mae_list, rmse_list, psnr_list, ssim_list, niqe_list = get_error_metrics_folder(target_paths, output_paths)
with open(output_file_path, 'w') as f:
    f.write('Before Enhancement:\n   mse: {mse_avg:.4f} | mae: {mae_avg:.4f} | rmse: {rmse_avg:.4f} |'
            ' psnr: {psnr_avg:.4f} | ssim: {ssim_avg:.4f} | niqe: {niqe_avg:.4f}\n'.format(mse_avg=np.mean(mse_list),
                                                                    mae_avg=np.mean(mae_list),
                                                                    rmse_avg=np.mean(rmse_list),
                                                                    psnr_avg=np.mean(psnr_list),
                                                                    ssim_avg=np.mean(ssim_list),
                                                                    niqe_avg=np.mean(niqe_list)))

target_enhanced_paths = find_png_files(target_enhanced_folder)[0]
# target_enhanced_paths = target_paths
output_enhanced_paths = find_png_files(result_enhanced_folder)[1]
print(len(target_enhanced_paths), len(output_enhanced_paths))
mse_list, mae_list, rmse_list, psnr_list, ssim_list, niqe_list = get_error_metrics_folder(target_enhanced_paths, output_enhanced_paths)
with open(output_file_path, 'a') as f:
    f.write('After Enhancement:\n   mse: {mse_avg:.4f} | mae: {mae_avg:.4f} | rmse: {rmse_avg:.4f} |'
            ' psnr: {psnr_avg:.4f} | ssim: {ssim_avg:.4f} | niqe: {niqe_avg:.4f}\n'.format(mse_avg=np.mean(mse_list),
                                                                    mae_avg=np.mean(mae_list),
                                                                    rmse_avg=np.mean(rmse_list),
                                                                    psnr_avg=np.mean(psnr_list),
                                                                    ssim_avg=np.mean(ssim_list),
                                                                    niqe_avg=np.mean(niqe_list)))
