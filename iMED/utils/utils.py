import numpy as np
import torch
import random
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from skimage.metrics import normalized_root_mse as compare_nrmse
from skimage.metrics import structural_similarity as compare_ssim
from skimage.metrics import peak_signal_noise_ratio as compare_psnr

# 设置随机数种子，让实验保持一致
def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


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
        assert (pd.flatten().shape == gt.flatten().shape)
        mse_pred = mean_squared_error(y_true=gt.flatten(), y_pred=pd.flatten())
        mae_pred = mean_absolute_error(y_true=gt.flatten(), y_pred=pd.flatten())
        rmse_pred = compare_nrmse(image_true=gt, image_test=pd)
        psnr_pred = compare_psnr(image_true=gt, image_test=pd)
        ssim_pred = compare_ssim(gt, pd)
        mse += mse_pred
        mae += mae_pred
        rmse += rmse_pred
        psnr += psnr_pred
        ssim += ssim_pred
    return mse, mae, rmse, psnr, ssim


class AverageMeter:
    def __init__(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

