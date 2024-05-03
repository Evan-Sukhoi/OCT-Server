import os
import numpy as np
from PIL import Image
from matplotlib import pyplot as plt
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from skimage.metrics import normalized_root_mse as compare_nrmse
from skimage.metrics import structural_similarity as compare_ssim
from skimage.metrics import peak_signal_noise_ratio as compare_psnr

import subprocess

output_file_path = 'output.txt'
original_size = (0, 0)

def find_png_files(directory):
    png_files = []
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.endswith('.png'):
                print(os.path.join(root, file))
                png_files.append(os.path.join(root, file))
    return png_files



def get_error_metrics(im_pd, im_gt):
    im_pd = np.array(im_pd).astype(np.float64)
    if np.max(im_pd) != 0:
        im_pd = im_pd / np.max(im_pd)
    im_gt = np.array(im_gt).astype(np.float64)
    if np.max(im_gt) != 0:
        im_gt = im_gt / np.max(im_gt)
    im_pd = im_pd[0]
    im_gt = im_gt[0]
    size = im_pd.shape[0]
    mse, mae, rmse, psnr, ssim = 0, 0, 0, 0, 0
    for i in range(size):
        pd = im_pd[i]
        gt = im_gt[i]
        # print(pd.shape, gt.shape)
        assert (pd.flatten().shape == gt.flatten().shape)
        mse_pred = mean_squared_error(y_true=gt.flatten(), y_pred=pd.flatten())
        mae_pred = mean_absolute_error(y_true=gt.flatten(), y_pred=pd.flatten())
        rmse_pred = compare_nrmse(image_true=gt, image_test=pd)
        psnr_pred = compare_psnr(image_true=gt, image_test=pd)
        ssim_pred = compare_ssim(gt, pd, win_size=3, data_range=255)
        mse += mse_pred
        mae += mae_pred
        rmse += rmse_pred
        psnr += psnr_pred
        ssim += ssim_pred
    # print(
    #     'mse: {mse_pred:.4f} | mae: {mae_pred:.4f} | rmse: {rmse_pred:.4f} |'
    #     ' psnr: {psnr_pred:.4f} | ssim: {ssim_pred:.4f}'.format(mse_pred=mse,
    #                                                             mae_pred=mae,
    #                                                             rmse_pred=rmse,
    #                                                             psnr_pred=psnr,
    #                                                             ssim_pred=ssim))
    return mse, mae, rmse, psnr, ssim


def get_error_metrics_folder(result_folder, target_folder, is_enhance=False):
    global original_size
    mse_list = []
    mae_list = []
    rmse_list = []
    psnr_list = []
    ssim_list = []
    for img_name in os.listdir(result_folder):
        if img_name.endswith('.png'):
            img_path_result = os.path.join(result_folder, img_name)
            img_path_target = os.path.join(target_folder, img_name)
            im_pd = Image.open(img_path_result)
            im_gt = Image.open(img_path_target)
            if is_enhance:
                im_pd = im_pd.resize(original_size, Image.BICUBIC)
                im_gt = im_gt.resize(original_size, Image.BICUBIC)
            else:
                original_size = im_pd.size
            mse, mae, rmse, psnr, ssim = get_error_metrics(im_pd, im_gt)
            mse_list.append(mse)
            mae_list.append(mae)
            rmse_list.append(rmse)
            psnr_list.append(psnr)
            ssim_list.append(ssim)
    print(
        'mse: {mse_avg:.4f} | mae: {mae_avg:.4f} | rmse: {rmse_avg:.4f} |'
        ' psnr: {psnr_avg:.4f} | ssim: {ssim_avg:.4f}'.format(mse_avg=np.mean(mse_list),
                                                                mae_avg=np.mean(mae_list),
                                                                rmse_avg=np.mean(rmse_list),
                                                                psnr_avg=np.mean(psnr_list),
                                                                ssim_avg=np.mean(ssim_list)))
    return mse_list, mae_list, rmse_list, psnr_list, ssim_list


def plot_error_metrics(mse_list, mae_list, rmse_list, psnr_list, ssim_list):
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.plot(mse_list, label='MSE')
    plt.plot(mae_list, label='MAE')
    plt.plot(rmse_list, label='RMSE')
    plt.plot(psnr_list, label='PSNR')
    plt.plot(ssim_list, label='SSIM')
    plt.legend()
    plt.xlabel('Epoch')
    plt.ylabel('Error')
    plt.title('Error Metrics')
    plt.subplot(1, 2, 2)
    plt.plot(mse_list, label='MSE')
    plt.plot(mae_list, label='MAE')
    plt.plot(rmse_list, label='RMSE')
    plt.plot(psnr_list, label='PSNR')
    plt.plot(ssim_list, label='SSIM')
    plt.legend()
    plt.xlabel('Epoch')
    plt.ylabel('Error')
    plt.title('Error Metrics')
    plt.show()


def crop_and_save_image(image_path, save_path, top_left, bottom_right):
    if not os.path.exists(image_path):
        print(f"图像文件不存在 {image_path}")
    img = Image.open(image_path)

    cropped_img = img.crop((top_left[0], top_left[1], bottom_right[0], bottom_right[1]))

    if not os.path.exists(os.path.dirname(save_path)):
        os.makedirs(os.path.dirname(save_path))

    cropped_img.save(save_path)
    # print(f"裁剪后的图像已保存至 {save_path}")


top_left_corner = (80, 188)
bottom_right_corner = (187, 295)

top_left_corner_2 = (209, 188)
bottom_right_corner_2 = (316, 295)

source_folder = "D:\\Projects\\Real-ESRGAN\\realesrgan\\evaluation\\sources"
result_folder = "D:\\Projects\\Real-ESRGAN\\realesrgan\\evaluation\\results"
target_folder = "D:\\Projects\\Real-ESRGAN\\realesrgan\\evaluation\\targets"

# for i in tqdm(range(0, 7)):
#     for k in range(0, 2):
#         for j in range(0, 128):
#             image_file_path = "epoch299_train\\iter" + str(i*4) + "\\batch" + str(k) + "\\imgno" + str(j) + ".png"

#             save_file_path = "epoch299_train_cropped_result\\iter" + str(i*4) + "\\batch"  + str(k) + "\\imgno" + str(j) + ".png"
#             crop_and_save_image(image_file_path, save_file_path, top_left_corner, bottom_right_corner)

#             save_file_path = "epoch299_train_cropped_target\\iter" + str(i*4) + "\\batch"  + str(k) + "\\imgno" + str(j) + ".png"
#             crop_and_save_image(image_file_path, save_file_path, top_left_corner_2, bottom_right_corner_2)

for image_name in os.listdir(source_folder):
    if image_name.endswith('.png'):
        image_file_path = os.path.join(source_folder, image_name)
        save_result_file_path = os.path.join(result_folder, image_name)
        save_target_file_path = os.path.join(target_folder, image_name)
        crop_and_save_image(image_file_path, save_result_file_path, top_left_corner, bottom_right_corner)
        crop_and_save_image(image_file_path, save_target_file_path, top_left_corner_2, bottom_right_corner_2)

# write into output.txt
mse_list, mae_list, rmse_list, psnr_list, ssim_list = get_error_metrics_folder(result_folder, target_folder)
with open(output_file_path, 'w') as f:
    f.write('Before Enhancement:\n   mse: {mse_avg:.4f} | mae: {mae_avg:.4f} | rmse: {rmse_avg:.4f} |'
            ' psnr: {psnr_avg:.4f} | ssim: {ssim_avg:.4f}\n'.format(mse_avg=np.mean(mse_list),
                                                                    mae_avg=np.mean(mae_list),
                                                                    rmse_avg=np.mean(rmse_list),
                                                                    psnr_avg=np.mean(psnr_list),
                                                                    ssim_avg=np.mean(ssim_list)))

result_enhanced_folder = "D:\\Projects\\Real-ESRGAN\\realesrgan\\evaluation\\results_enhanced"
target_enhanced_folder = "D:\\Projects\\Real-ESRGAN\\realesrgan\\evaluation\\targets_enhanced"

# deblur images
# for image_name in os.listdir(result_folder):
#     if image_name.endswith('.png'):
#         result_file_path = os.path.join(result_folder, image_name)
#         target_file_path = os.path.join(target_folder, image_name)
#     command_1 = 'python inference_realesrgan.py --model_path D:\\Projects\\Real-ESRGAN\experiments\\train_RealESRNetx4plus_1000k_B12G4\\models\\net_g_795000.pth -i ' + result_file_path +' -o ' + result_enhanced_folder + ' --fp32 -s 4'
#     command_2 = 'python inference_realesrgan.py --model_path D:\\Projects\\Real-ESRGAN\experiments\\train_RealESRNetx4plus_1000k_B12G4\\models\\net_g_795000.pth -i ' + target_file_path +' -o ' + target_enhanced_folder + ' --fp32 -s 4'
#     subprocess.call(command_1, shell=True)
#     subprocess.call(command_2, shell=True)

mse_list, mae_list, rmse_list, psnr_list, ssim_list = get_error_metrics_folder(result_enhanced_folder, target_enhanced_folder, is_enhance=True)
with open(output_file_path, 'a') as f:
    f.write('After Enhancement:\n   mse: {mse_avg:.4f} | mae: {mae_avg:.4f} | rmse: {rmse_avg:.4f} |'
            ' psnr: {psnr_avg:.4f} | ssim: {ssim_avg:.4f}\n'.format(mse_avg=np.mean(mse_list),
                                                                    mae_avg=np.mean(mae_list),
                                                                    rmse_avg=np.mean(rmse_list),
                                                                    psnr_avg=np.mean(psnr_list),
                                                                    ssim_avg=np.mean(ssim_list)))
