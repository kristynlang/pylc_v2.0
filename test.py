# Mountain Legacy Project: Semantic Segmentation
# Author: Spencer Rose
# Date: November 2019
#
# REFERENCES:
# Fully-convolutional networks (baseline)
# Long, Jonathan, Evan Shelhamer, and Trevor Darrell.
# "Fully convolutional networks for semantic segmentation." In Proceedings of
# the IEEE conference on computer vision and pattern recognition, pp. 3431-3440.
# 2015  https://people.eecs.berkeley.edu/~jonlong/long_shelhamer_fcn.pdf
# U-Net

import os, gc, math, datetime
import torch
import numpy as np
import h5py
from config import get_config
import utils.utils as utils
from utils.dbwrapper import MLPDataset, load_data
from models.base import Model
from models.unet import UNet
from tqdm import tqdm, trange
from params import params


def test(config, model):
    ''' Test trained model '''
    # Load test image data subimages [NCWH]
    img_data = torch.as_tensor(utils.get_image(config.img_path, config.in_channels), dtype=torch.float32)
    print('\tTest image {} / Shape: {}'.format(config.img_path, img_data.shape))
    img_data = img_data.unfold(0, params.patch_size, params.patch_size).unfold(1, params.patch_size, params.patch_size)
    img_data = torch.reshape(img_data, (img_data.shape[0]*img_data.shape[1], config.in_channels, params.patch_size, params.patch_size) )
    print('\tImage Patches {} / Shape: {}'.format(img_data.shape[0], img_data.shape))

    # Load mask data subimages [NCWH]
    mask_data = torch.as_tensor(utils.get_image(config.mask_path, 3), dtype=torch.int64)
    print('\tMask Image {} / Shape: {}'.format(config.mask_path, mask_data.shape))
    mask_data = mask_data.unfold(0, params.patch_size, params.patch_size).unfold(1, params.patch_size, params.patch_size)
    mask_data = torch.reshape(mask_data, (mask_data.shape[0]*mask_data.shape[1], 3, params.patch_size, params.patch_size) )
    mask_data = utils.class_encode(mask_data)
    print('\tMask Patches {} / Shape: {}'.format(mask_data.shape[0], mask_data.shape))

    assert img_data.shape[0]==mask_data.shape[0], 'Image dimensions must match mask dimensions.'

    n_samples = int(img_data.shape[0] * config.clip)

    print('\n\tNumber of samples: {} (clip: {})'.format(n_samples, config.clip))

    assert img_data.shape[0] == mask_data.shape[0], 'Different number of image and mask samples generated.'

    print('\nApplying model ... ')

    model.net.eval()
    with torch.no_grad():
        for i in trange(n_samples):
            x = img_data[i].unsqueeze(0).float()
            y = mask_data[i].unsqueeze(0).long()

            if model.n_classes == 4:
                y = utils.merge_classes(y)

            model.eval(x, y, test=True)
            model.log()

            model.iter += 1
            if model.iter == int(config.clip * n_samples):
                model.save(test=True)
                exit()

    print("\n[Test] Dice Best: %4.4f\n" % (model.loss.best_dice))

def init_capture(config):
    ''' initialize parameters for capture type '''
    if config.capture == 'historic':
        config.n_classes = 11
        config.in_channels = 1
    elif config.capture == 'historic_merged':
        config.n_classes = 4
        config.in_channels = 1
    elif config.capture == 'repeat':
        config.n_classes = 11
        config.in_channels = 3
    elif config.capture == 'repeat_merged':
        config.n_classes = 4
        config.in_channels = 3
    return config

def main(config):
    # initialize config parameters based on capture type
    config = init_capture(config)

    print("\nTraining {} model / Mode: {} / Capture Type: {} ... ".format(config.model, config.mode, config.capture))
    print('\tInput channels: {} / Classes: {}'.format(config.in_channels, config.n_classes))

    model = Model(config)

    if config.mode == params.NORMAL:
        test(config, model)
    else:
        raise ValueError("Unknown run mode \"{}\"".format(config.mode))


if __name__ == "__main__":

    ''' Parse model configuration '''
    config, unparsed, parser = get_config(params.TEST)

    # If we have unparsed arguments, or help request print usage and exit
    if len(unparsed) > 0 or config.h:
        parser.print_usage()
        exit()

    main(config)


#
# test.py ends here
