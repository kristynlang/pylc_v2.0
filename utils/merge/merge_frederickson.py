# ===========================
# Data Handling Utilities
# ===========================

import os, sys, glob, math, json
import numpy as np
import torch
import h5py
import random
from random import shuffle
import cv2
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter

# ===========================
# Parameters
# ===========================

root = '/Users/kristynlang/Desktop/kristyn'

# merged scheme categories
categories_lcc_a = {
    '#000000': 'Not categorized',
    '#ffa500': 'Broadleaf/Mixedwood',
    '#228b22': 'Coniferous',
    '#7cfc00': 'Herbaceous/Shrub',
    '#8b4513': 'Sand/Gravel/Rock',
    '#5f9ea0': 'Wetland',
    '#0000ff': 'Water',
    '#2dbdff': 'Snow/Ice',
    '#ff0004': 'Regenerating Area',
}

labels_lcc_a = [
    'Not categorized',
    'Broadleaf/Mixedwood',
    'Coniferous',
    'Herbaceous/Shrub',
    'Sand/Gravel/Rock',
    'Wetland',
    'Water',
    'Snow/Ice',
    'Regenerating Area',
]

palette_lcc_a = np.array(
[[0, 0, 0],
 [255, 165, 0],
 [34, 139, 34],
 [124, 252, 0],
 [139, 69, 19],
 [95, 158, 160],
 [0, 0, 255],
 [45, 189, 255],
 [255, 0, 4],
 ])

#fortin categories and colors

mask_categories = {
'#000000':'Not categorized',
'#ffa500':'Broadleaf/mixedwood forest',
'#228b22':'Coniferous forest',
'#7cfc00':'Herbaceous/shrub',
'#8b4513':'Barren rock',
'#d35d0e':'Infrastructure',
'#5f9ea0':'Wetland',
'#0000ff':'Water',
'#2dbdff':'Snow/Ice',
'#ff0004':'Regenerating Area',
}
        
category_labels = [
'Not categorized',
'Broadleaf/mixedwood forest',
'Coniferous forest',
'Herbaceous/shrub',
'Barren grouond',
'Infrastructure',
'Wetland',
'Water',
'Snow/Ice',
'Burn']

palette_original = np.array(
[[0, 0, 0],
[255, 165, 0],
[34, 139, 34],
[124, 252, 0],
[139, 69, 19],
[211, 93, 14],
[95, 158, 160],
[0, 0, 255],
[45, 189, 255],
[255, 0, 4],
])

# merged classes
categories_merged = [
    np.array([0]),
    np.array([1]),
    np.array([2]),
    np.array([3]),
    np.array([4,5]),
    np.array([7]),
    np.array([8]),
    np.array([9]),
    np.array([10])
]


# Merged classes
palette_merged = palette_lcc_a

mask_categories_merged = categories_lcc_a

category_labels_merged = labels_lcc_a

# size of the output feature map
output_size = 324

# size of the tiles to extract and save in the database, must be >= to input size
input_size = 512
patch_size = 512

# patch stride: smaller than input_size for overlapping tiles
stride_size = 162

# number of pixels to pad *after* resize to image with by mirroring (edge's of
# patches tend not to be analyzed well, so padding allows them to appear more centered
# in the patch)
pad_size = (input_size - output_size)//2

# Calculate crop sizes
crop_left = pad_size
crop_right = pad_size + output_size
crop_up = pad_size
crop_down = pad_size + output_size


# ===========================
# Image Processing: Utilities
# ===========================
# Convert RGBA to Hex
def RGB2HEX(color):
    return "#{:02x}{:02x}{:02x}".format(int(color[0]), int(color[1]), int(color[2]))

hex_palette = [RGB2HEX(i) for i in palette_original]
hex_palette_merged = [RGB2HEX(i) for i in palette_merged]


# Read image and reverse channel order
# Loads image as 8 bit (regardless of original depth):
def get_image(image_path, img_ch=3):
    assert img_ch == 3 or img_ch == 1, 'Invalid input channel number.'
    image = None
    if img_ch == 1:
        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    elif img_ch == 3:
        image = cv2.imread(image_path, cv2.IMREAD_COLOR)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return image


# Replace class index values with palette colours
# Input format: NWH (class encoded)
def colourize(img_data, palette):
    n = img_data.shape[0]
    w = img_data.shape[1]
    h = img_data.shape[2]
    # make 3-channel (RGB) image
    img_data = np.moveaxis(np.stack((img_data,)*3, axis=1), 1, -1).reshape(n*w*h, 3)
    # map categories to palette colours
    for i, colour in enumerate(palette):
        bool = img_data == np.array([i,i,i])
        bool = np.all(bool, axis=1)
        img_data[bool] = palette[i]
    return img_data.reshape(n, w, h, 3)


# Merge segmentation classes
def merge_classes(data):

    # get merged classes boolean
    uncat_idx = np.isin(data, categories_merged[0])
    bmix_idx = np.isin(data, categories_merged[1])
    con_idx = np.isin(data, categories_merged[2])
    herbshrub_idx = np.isin(data, categories_merged[3])
    rock_idx = np.isin(data, categories_merged[4])
    wetland_idx = np.isin(data, categories_merged[5])
    water_idx = np.isin(data, categories_merged[6])
    snowice_idx = np.isin(data, categories_merged[7])
    reg_idx = np.isin(data, categories_merged[8])
    
    # merge classes
    data[uncat_idx] = 0
    data[bmix_idx] = 1
    data[con_idx] = 2
    data[herbshrub_idx] = 3
    data[rock_idx] = 4
    data[wetland_idx] = 5
    data[water_idx] = 6
    data[snowice_idx] = 7
    data[reg_idx] = 8

    return data

# Convert RBG mask array to class-index encoded values
# input form: NCWH with RGB-value encoding, where C = RGB (3)
# Palette paramters in form [CC'], where C = number of classes, C' = 3 (RGB)
# output form: NCWH with one-hot encoded classes, where C = number of classes
def class_encode(input, palette=palette_original):
    assert input.shape[1] == 3
    input = input.to(torch.float32).mean(dim=1)
    palette = torch.from_numpy(palette).to(torch.float32).mean(dim=1)
    # map mask colours to segmentation classes
    for idx, c in enumerate(palette):
        bool = input == c
        input[bool] = idx
    return input.to(torch.uint8)


# Parameter Grid Search: Data augmentation
def grid_search(px_dist, px_count, dset_px_dist, dset_px_count, dset_probs, n_classes):
    profile_data = []
    eps = 1e-8

    # Initialize oversample filter and class prior probabilities
    oversample_filter = np.clip(1/n_classes - dset_probs, a_min=0, a_max=1.)
    probs = px_dist/px_count
    probs_weighted = np.multiply(np.multiply(probs, 1/dset_probs), oversample_filter)
    scores = np.sqrt(np.sum(probs_weighted, axis=1))

    # Initialize augmentation parameters
    rate_coefs = np.arange(1, 21, 1)
    thresholds = np.arange(0, 3., 0.1)
    aug_n_samples_max = 16000
    #delta_px = []
    jsd = []

    # initialize balanced model distribution
    balanced_px_prob = np.empty(n_classes)
    balanced_px_prob.fill(1/n_classes)

    # Grid search
    for i, rate_coef, in enumerate(rate_coefs):
        for j, threshold in enumerate(thresholds):
            assert rate_coef >= 1, 'Rate coefficient must be >= one.'
            oversample = scores > threshold
            rates = np.multiply(oversample, rate_coef * scores).astype(int)
            # limit to max number of augmented images
            if np.sum(rates) < aug_n_samples_max:
                aug_px_dist = np.multiply(np.expand_dims(rates, axis=1), px_dist)
                full_px_dist = px_dist + aug_px_dist
                full_px_probs = np.sum(full_px_dist, axis=0)/np.sum(full_px_dist)
                #delta_px += [np.sum(np.multiply(full_px_probs, oversample_filter))]
                jsd += [JSD(full_px_probs, balanced_px_prob)]
                profile_data += [{
                    'probs': full_px_probs,
                    'threshold' : threshold,
                    'rate_coef': rate_coef,
                    'rates': rates,
                    'n_samples': int(np.sum(full_px_dist)/px_count),
                    'aug_n_samples': np.sum(rates)
                }]

    # Get parameters that minimize Jensen-Shannon Divergence metric
    if len(jsd) == 0:
        print('No augmentation optimization found')
        return
    optim_idx = np.argmin(np.asarray(jsd))
    return profile_data[optim_idx]

# Print colour-coded categories
def plot_legend(palette, title, categories=mask_categories):

    cell_width = 260
    cell_height = 30
    swatch_width = 48
    margin = 12
    topmargin = 40

    n = len(palette)
    ncols = 3
    nrows = n // ncols + int(n % ncols > 0)

    width = cell_width * 4 + 2 * margin
    height = cell_height * nrows + margin + topmargin
    dpi = 72

    fig, ax = plt.subplots(figsize=(width / dpi, height / dpi), dpi=dpi)
    fig.subplots_adjust(margin/width, margin/height,
                        (width-margin)/width, (height-topmargin)/height)
    ax.set_xlim(0, cell_width * 4)
    ax.set_ylim(cell_height * (nrows-0.5), -cell_height/2.)
    ax.yaxis.set_visible(False)
    ax.xaxis.set_visible(False)
    ax.set_axis_off()
    ax.set_title(title, fontsize=24, loc="left", pad=10)

    for i, colour in enumerate(palette):
        row = i % nrows
        col = i // nrows
        y = row * cell_height
        label = ''

        swatch_start_x = cell_width * col
        swatch_end_x = cell_width * col + swatch_width
        text_pos_x = cell_width * col + swatch_width + 7

        ax.text(text_pos_x, y, categories[colour] + label, fontsize=14,
                horizontalalignment='left',
                verticalalignment='center')

        ax.hlines(y, swatch_start_x, swatch_end_x,
                  color=colour, linewidth=18)

    plt.show()



# Plot image/mask samples
def plot_samples(img_data, mask_data, n_rows=20, n_cols=5, offset=0, title=None, palette=hex_palette, categories=mask_categories):

    plot_legend(palette, title, categories=categories)

    k = offset
    fig, axes = plt.subplots(
            n_rows,
            n_cols,
            sharex='col',
            sharey='row',
            figsize=(40, 100),
            subplot_kw={'xticks': [], 'yticks': []})
    plt.rcParams.update({'font.size': 18})

    # Plot sample subimages
    for i in range(0, n_rows):
        for j in range(n_cols):
            # Get sample patch & mask
            img = img_data[k].astype(int)
            mask = mask_data[k]

            # show original subimage
            if img_data.ndim == 3:
                axes[i,j].imshow(img, cmap='gray')
            else:
                axes[i,j].imshow(img)
            # overlay subimage seg-mask
            axes[i,j].imshow(mask, alpha=0.35)
            axes[i,j].set_title('Sample #{}'.format(k + 1))
            k += 1
    plt.show()


# Plot dataset pixel distribution profile
def plot_profile(
    n_classes,
    probs,
    n_samples,
    category_labels=category_labels,
    palette=hex_palette,
    title=None,
    overlay=None,
    label_1=None,
    label_2=None):

    # Calculate JSD and weights
    balanced_px_prob = np.empty(n_classes)
    balanced_px_prob.fill(1/n_classes)
    jsd = JSD(probs, balanced_px_prob)
    weights = eval_weights(probs)

    print(title)
    print('\n{:20s}{:>15s}{:>15s}\n'.format('Class', 'Prob', 'Weight', ))
    for i, p in enumerate(probs):
        print('{:20s}{:15f}{:15f}'.format(category_labels[i], p, weights[i]))
    print('\nTotal samples: {}'.format(n_samples))
    print('Sample Size: {} x {} = {} pixels'.format(patch_size, patch_size, patch_size*patch_size))
    print('\nJSD: {:6f}'.format(jsd))

    # Set figure size
    fig_size = plt.rcParams["figure.figsize"]
    fig_size[0] = 18
    fig_size[1] = 6
    plt.rcParams["figure.figsize"] = fig_size
    plt.rcParams.update({'font.size': 9})

    x = np.arange(n_classes)
    if overlay is not None:
        width = 0.3  # the width of the bars
        plt.bar(x - width/2, overlay, width=width, color=palette, alpha=0.5, label=label_2)
        plt.bar(x + width/2, probs, width=width, color=palette, alpha=0.9, label=label_1)
        plt.legend()
    else:
        plt.bar(x, probs, width=0.7, color=palette, alpha=0.9)
    plt.xticks(x, category_labels)
    plt.ylabel('Proportion of pixels')
    plt.title(title)
    plt.axhline(1/n_classes, color='k', linestyle='dashed', linewidth=1)
    plt.show()



# Plot sample pixel distribution profile
def plot_grid_profiles(profile_data, n_rows=25, n_cols=5, offset=0):

    thresholds = profile_data['threshold']
    rate_coefs = profile_data['rate_coef']
    k = offset
    width = 0.5  # the width of the bars
    fig, axes = plt.subplots(n_rows, n_cols, sharex='col', sharey='row', figsize=(40, 300),
                             subplot_kw={'xticks': [], 'yticks': []})
    plt.rcParams.update({'font.size': 14})

    # Plot sample class profiles
    for i in range(0, n_rows):
        for j in range(n_cols):
            # plot profile histogram
            threshold = thresholds[k] if isinstance(thresholds, list) else thresholds
            rate_coef = rate_coefs[k] if isinstance(rate_coefs, list) else rate_coefs
            title = 'Sample #{} [Thresh: {} \ Rate Coef: {}]'.format(k, threshold, rate_coef)
            axes[i,j].axhline(1/n_classes, color='k', linestyle='dashed', linewidth=1)
            axes[i,j].bar(
                range(n_classes),
                profile_data['probs'][k],
                color=hex_palette,
                alpha=0.7
            )
            if profile_data['mse'] is not None:
                title += "\nMSE: {}".format(profile_data['mse'][k])
            axes[i,j].set_title(title)
            k += 1
    plt.show()