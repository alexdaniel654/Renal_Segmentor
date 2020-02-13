# -*- coding: utf-8 -*-
"""
Created on Fri Jul 12 11:25:34 2019

@author: Alex Daniel
"""

# Import Packages
import os
import nibabel as nib
import numpy as np
from skimage.transform import resize
from tensorflow.keras import backend as K
from tensorflow.keras.models import load_model
import tensorflow as tf
from gooey import Gooey, GooeyParser

# Define Classes


class RawData:
    def __init__(self, path):
        self.path = path
        self.__split_path__()
        self.img = nib.Nifti1Image
        self.data = np.array
        self.affine = np.array

    def __split_path__(self):
        self.directory = os.path.dirname(self.path)
        self.base = os.path.splitext(os.path.basename(self.path))[0]
        self.extension = os.path.splitext(os.path.basename(self.path))[1]
        if self.extension == '.gz' and self.base[-4:] == '.nii':
            self.extension = '.nii.gz'
            self.base = self.base[:-4]

    def load(self):
        if self.extension == '.PAR':
            self.img = nib.load(self.path, scaling='fp')
        else:
            self.img = nib.load(self.path)
        self.data = self.img.get_fdata()
        self.affine = self.img.affine

# Define Functions


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
    data = np.copy(raw_data)
    data = np.flip(data, 1)
    data = np.swapaxes(data, 0, 2)
    data = np.swapaxes(data, 1, 2)
    for n in range(data.shape[0]):
        data[n, :, :] = rescale(data[n, :, :])
    data = resize(data, (data.shape[0], 256, 256))
    data = data.reshape((data.shape[0], data.shape[1], data.shape[2], 1))
    return data


def un_pre_process(raw_data, base_img):
    data = np.copy(raw_data)
    data = np.squeeze(data)
    data = np.swapaxes(data, 0, 2)
    data = np.swapaxes(data, 0, 1)
    data = resize(data, (base_img.shape[0], base_img.shape[1], base_img.shape[2]))
    data = np.flip(data, 1)
    return data


def rescale(data):
    black = np.mean(data) - 0.5 * np.std(data)
    if black < data.min():
        black = data.min()
    white = np.mean(data) + 4 * np.std(data)
    if white > data.max():
        white = data.max()
    data = np.clip(data, black, white)-black
    data = data/(white-black)
    return data


def predict_mask(data):
    print('Loading model')
    model = load_model(resource_path('./models/renal_segmentor.model'),
                       custom_objects={'dice_coef_loss': dice_coef_loss, 'dice_coef': dice_coef})

    print('Making prediction')
    batch_size = 2 ** 3
    prediction = model.predict(data, batch_size=batch_size)
    return prediction


def get_parser():
    # Make argparser
    parser = GooeyParser(prog='Renal Segmentor', description='Segment renal MRI images.')
    parser.add_argument('input', default='test_data/test_sub_01.PAR',
                        help='The image you wish to segment',
                        widget='FileChooser')
    parser.add_argument('-b', '--binary', action='store_true', default=False, dest='binary',
                        help='The mask output will only be 0 or 1.')
    parser.add_argument('-r', '--raw', action='store_true', default=False, dest='raw',
                        help='Output the raw data used for the segmentation.')
    parser.add_argument('-output', default=False,
                        help='The name you wish to give your output mask.')
    return parser


def resource_path(relative_path):
    """ Get absolute path to resource, works for dev and for PyInstaller """
    try:
        # PyInstaller creates a temp folder and stores path in _MEIPASS
        base_path = sys._MEIPASS
    except Exception:
        base_path = os.path.abspath(".")

    return os.path.join(base_path, relative_path)


@Gooey(program_name='Renal Segmentor',
       image_dir=resource_path('./icons'))
def main():
    tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

    parser = get_parser()
    args = parser.parse_args()

    # Import data
    print('Loading data')
    raw_data = RawData(args.input)
    raw_data.load()

    print('Pre-processing')
    data = pre_process_img(raw_data.data)

    # Predict mask
    prediction = predict_mask(data)

    print('Outputting data')
    mask = un_pre_process(prediction, raw_data.img)
    if args.binary:
        mask = (mask > 0.5) * 1

    # Output mask
    if not args.output:
        output_path = raw_data.directory + '/' + raw_data.base + '_mask.nii.gz'
    else:
        output_path = args.output

    mask_img = nib.Nifti1Image(mask, raw_data.affine)
    nib.save(mask_img, output_path)

# TODO Make this so it exports with output data rather than input data
    if args.raw:
        nib.save(raw_data.img, raw_data.directory + '/' + raw_data.base + '.nii.gz')


if __name__ == "__main__":
    main()
