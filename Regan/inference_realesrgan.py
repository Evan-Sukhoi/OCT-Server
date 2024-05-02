import argparse
import cv2
import glob
import os
from basicsr.archs.rrdbnet_arch import RRDBNet
from basicsr.utils.download_util import load_file_from_url

from realesrgan import RealESRGANer
from realesrgan.archs.srvgg_arch import SRVGGNetCompact


def deblur(model_path, input_folder, output_path):
    print('Deblurring...')
    model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=4)
    netscale = 4

    # use dni to control the denoise strength
    dni_weight = None
    tile = 0
    tile_pad = 10
    pre_pad = 0
    fp32 = True
    gpu_id = 0
    outscale = 4
    suffix = 'out'

    # restorer
    upsampler = RealESRGANer(
        scale=netscale,
        model_path=model_path,
        dni_weight=dni_weight,
        model=model,
        tile=tile,
        tile_pad=tile_pad,
        pre_pad=pre_pad,
        half=not fp32,
        gpu_id=gpu_id)

    os.makedirs(output_path, exist_ok=True)

    for idx, path in enumerate(sorted(glob.glob(os.path.join(input_folder, '*')))):
        imgname, extension = os.path.splitext(os.path.basename(path))
        print('Testing', idx, imgname)

        img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
        if len(img.shape) == 3 and img.shape[2] == 4:
            img_mode = 'RGBA'
        else:
            img_mode = None

        try:
            output, _ = upsampler.enhance(img, outscale=outscale)
        except RuntimeError as error:
            print('Error', error)
            print('If you encounter CUDA out of memory, try to set --tile with a smaller number.')
        else:
            extension = extension[1:]
            if img_mode == 'RGBA':  # RGBA images should be saved in png format
                extension = 'png'
            if suffix == '':
                save_path = os.path.join(output_path, f'{imgname}.{extension}')
            else:
                save_path = os.path.join(output_path, f'{imgname}_{suffix}.{extension}')
            cv2.imwrite(save_path, output)

