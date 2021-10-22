from argparse import FileType
import cv2
import numpy as np
import math
import os
import glob
import torch
import numpy
from torch.autograd import Variable
from skimage.metrics import structural_similarity as sk_cpt_ssim
from skimage.metrics.simple_metrics import peak_signal_noise_ratio
from skimage import io, transform

import os
import subprocess


def get_image_list(fileTxt, dataRoot):
    """

    Args:
        fileTxt ([type]): test set file
        dataRoot ([type]): generated test set

    Returns:
        [type]: [description]
    """
    filelist = []
    fileroot = []

    with open(fileTxt) as f:
        for line in f:
            line = line.strip()
            line0 = line.split()[0]
            line0 = os.path.join(dataRoot, "{line0}".format(line0=line0))
            filelist.append(line0)
            line1 = line.split()[0]
            fileroot.append(line1)

    return filelist, fileroot


if __name__ == '__main__':

    testFile = 'filelists/test.txt'
    name2 = '10_19_vgg_with_weight_0.1_26W_random_audio'
    generatedData = '/data_8T/xjw/DeepFake/generated_result/Frames/{}/'.format(
        name2)
    originalData = 'lrs2_preprocessed'
    filelist, file1 = get_image_list(testFile, generatedData)

    test_psnr = 0
    test_ssim = 0
    num = 0
    csattn_gan_PSNR = 0
    csattn_gan_SSIM = 0

    for i, root in enumerate(file1):
        gt = os.path.join(generatedData, "{}".format(i))
        if not os.path.exists(gt): continue

        img_path = os.listdir(gt)
        for img in img_path:
            if not img == "audio.wav":
                csattn_gan_img_path = os.path.join(
                    generatedData, "{root}/{img}".format(root=i, img=img))
                original_img_path = os.path.join(
                    originalData, "{root}/{img}".format(root=root, img=img))
                csattn_gan_img = cv2.imread(csattn_gan_img_path)
                original_img = cv2.imread(original_img_path)
                if original_img is None or csattn_gan_img is None:
                    print(csattn_gan_img_path)
                    continue

                csattn_gan_img = transform.resize(csattn_gan_img, (96, 96))
                original_img = transform.resize(original_img, (96, 96))
                PSNR = peak_signal_noise_ratio(csattn_gan_img, original_img)
                SSIM = sk_cpt_ssim(csattn_gan_img,
                                   original_img,
                                   multichannel=True)
                test_psnr += PSNR
                test_ssim += SSIM
        test_psnr = test_psnr / (len(img_path) - 1)
        test_ssim = test_ssim / (len(img_path) - 1)

        print("each video_psnr:", test_psnr)
        print("each video_ssim:", test_ssim)
        num += 1
        csattn_gan_PSNR += test_psnr
        csattn_gan_SSIM += test_ssim

    csattn_gan_PSNR /= num
    csattn_gan_SSIM /= num
    print(num)
    print(csattn_gan_PSNR)
    print(csattn_gan_SSIM)
