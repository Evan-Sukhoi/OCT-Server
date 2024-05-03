import os
import numpy as np
from PIL import Image
from matplotlib import pyplot as plt
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from skimage.metrics import normalized_root_mse as compare_nrmse
from skimage.metrics import structural_similarity as compare_ssim
from skimage.metrics import peak_signal_noise_ratio as compare_psnr


def generate_data_path(path: str):
    names = path.split("\\")
    tif_path, volume_path = None, None
    for filename in os.listdir(path):
        suffix = os.path.splitext(filename)[-1]
        if suffix == '.tif':
            tif_path = os.path.join(path, filename)
        elif suffix == '.fda_OCT_pngs':
            images_path = os.path.join(path, filename)
            volume = np.zeros((128, 128, 128), dtype=np.int64)
            numbers = []
            for i in range(128):
                numbers.append(i)
            for i in range(len(numbers)):
                oct_path = "octscan_%i.png" % (numbers[i] + 1)
                oct_path = os.path.join(images_path, oct_path)
                img = Image.open(oct_path)
                img = img.crop((0, 30, 512, 542))
                img = img.resize((128, 128))
                img = np.array(img, dtype=np.float32)
                volume[i, :, :] = img
            volume_path = os.path.join(path, "volume.npy")
            np.save(volume_path, volume)
    assert tif_path is not None
    assert volume_path is not None
    return tif_path, volume_path, names[-1]


def clean_data_npy(path: str):
    for filename in os.listdir(path):
        if filename == 'volume.npy':
            filename = os.path.join(path, filename)
            os.remove(filename)


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
    # print(
    #     'mse: {mse_pred:.4f} | mae: {mae_pred:.4f} | rmse: {rmse_pred:.4f} |'
    #     ' psnr: {psnr_pred:.4f} | ssim: {ssim_pred:.4f}'.format(mse_pred=mse,
    #                                                             mae_pred=mae,
    #                                                             rmse_pred=rmse,
    #                                                             psnr_pred=psnr,
    #                                                             ssim_pred=ssim))
    return mse, mae, rmse, psnr, ssim


def generate_file_path(**kwargs):
    file_path = os.getcwd()
    for (key, value) in kwargs.items():
        tmp_path = "{}{}".format(str(key), str(value))
        file_path = os.path.join(file_path, tmp_path)
        if not str(value).endswith(".png") and not os.path.exists(file_path):
            os.mkdir(file_path)
    return file_path


def save_comparison_images(output, target, mode, **kwargs):
    output = np.array(output).astype(np.float64)
    target = np.array(target).astype(np.float64)
    for batch in range(len(output)):
        output_batch = output[batch]
        target_batch = target[batch]
        seq = range(output_batch.shape[0])
        for idx in seq:
            pd = output_batch[idx, :, :]
            gt = target_batch[idx, :, :]
            f = plt.figure()
            f.add_subplot(1, 4, 1)
            plt.imshow(pd, interpolation='none', cmap='gray')
            plt.title("Output")
            plt.axis("off")
            f.add_subplot(1, 4, 2)
            plt.imshow(gt, interpolation='none', cmap='gray')
            plt.title("Target")
            plt.axis("off")
            f.add_subplot(1, 4, 3)
            plt.imshow(gt - pd, interpolation='none', cmap='gray')
            plt.title("Target - Output")
            plt.axis("off")
            f.add_subplot(1, 4, 4)
            plt.imshow(pd - gt, interpolation='none', cmap='gray')
            plt.title("Output - Target")
            plt.axis("off")
            file_path = generate_file_path(result='', mode=mode, **kwargs, batch=batch, img="no{}.png".format(idx))
            f.savefig(file_path)
            plt.close()
    # print("Save difference images in %s" % mode)


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
