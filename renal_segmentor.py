# -*- coding: utf-8 -*-
"""
Created on Fri Jul 12 11:25:34 2019

@author: Alex Daniel
"""

# %% Import Packages
import os
import nibabel as nib
import numpy as np
from skimage.transform import resize
from sklearn import preprocessing
from keras import backend as K
from keras.models import load_model


# %% Define Functions

def dice_coef(y_true, y_pred):
    smooth = 1.0
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)


def dice_coef_loss(y_true, y_pred):
    loss = 1 - dice_coef(y_true, y_pred)
    return loss


def pre_process_img(raw_data):
    data = np.flipud(raw_data)
    for n in np.arange(data.shape[2]):
        data[:, :, n] = preprocessing.normalize(data[:, :, n])
    data = resize(data, (256, 256, data.shape[2]), preserve_range=True)
    data = data.reshape((data.shape[0], data.shape[1], data.shape[2], 1))
    data = np.rot90(data, -1)
    data = np.swapaxes(data, 0, 2)
    data = np.swapaxes(data, 1, 2)
    return data


def un_process_mask(mask, base_img):
    mask = mask.reshape(mask.shape[0], mask.shape[1], mask.shape[2])
    mask = np.swapaxes(mask, 0, 2)
    mask = np.swapaxes(mask, 0, 1)
    mask = resize(mask, (base_img.shape[0], base_img.shape[1], base_img.shape[2]))
    mask = np.rot90(mask)
    mask = np.flipud(mask)
    return mask


def split_path(full_path):
    directory = os.path.dirname(full_path)
    base = os.path.splitext(os.path.basename(full_path))[0]
    extension = os.path.splitext(os.path.basename(full_path))[1]
    if extension == '.gz' and base[-4:] == '.nii':
        extension = '.nii.gz'
        base = base[:-4]
    return directory, base, extension


# %% Import data

img_path = './test_data/test_sub_01.PAR'
directory, base, extension = split_path(img_path)
binary = True
raw = True
img = nib.load(img_path, scaling='fp')
data = img.get_data()
data = pre_process_img(data)

# %% Predict mask

if 'model' not in locals():
    model = load_model('.\models\All46_norm_0.93008.model',
                       custom_objects={'dice_coef_loss': dice_coef_loss, 'dice_coef': dice_coef})
batch_size = 2 ** 3

prediction = model.predict(data, batch_size=batch_size)
mask = un_process_mask(prediction, img)
if binary:
    mask = (mask > 0.5) * 1

# %% Output mask

output_path = directory + '/' + base + '_mask.nii.gz'
mask_img = nib.Nifti1Image(mask, img.affine)
nib.save(mask_img, output_path)

if raw:
    nib.save(img, directory + '/' + base + '.nii.gz')
