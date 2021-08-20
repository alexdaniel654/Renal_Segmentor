# -*- coding: utf-8 -*-
"""
Created on Fri Jul 12 11:25:34 2019

@author: Alex Daniel
"""

# Import Packages

import nibabel as nib
import numpy as np
import os
import sys
import tensorflow as tf

from gooey import Gooey, GooeyParser
from skimage.measure import label, regionprops
from skimage.transform import resize
from tensorflow.keras import backend as K
from tensorflow.keras.models import load_model

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


def cleanup(mask):
    clean_mask = np.zeros(mask.shape, dtype=np.uint8)
    label_mask = label(mask, connectivity=1)
    props = regionprops(label_mask)
    areas = [region.area for region in props]
    kidney_labels = np.argpartition(areas, -2)[-2:]  # This means there have to be two kidneys in the scan...

    clean_mask[label_mask == props[kidney_labels[0]].label] = 1
    clean_mask[label_mask == props[kidney_labels[1]].label] = 1

    return clean_mask


def get_parser():
    # Make argparser
    parser = GooeyParser(description='Segment renal MRI images.')
    parser.add_argument('input',
                        metavar='Input Data',
                        help='The image you wish to segment.',
                        widget='FileChooser',
                        gooey_options={'wildcard':
                                       'Common Files (*.PAR, *.nii.gz, '
                                       '*.hdr, *.nii)|*.PAR; *.nii.gz; '
                                       '*.hdr; *.nii|'
                                       'Compressed Nifti (*.nii.gz)|*.nii.gz|'
                                       'Nifti (*.nii)|*.nii|'
                                       'Philips (*.PAR)|*.PAR|'
                                       'Analyze (*.hdr/*.img)|*.hdr|'
                                       'All files (*.*)|*.*',
                                       'message': 'Select Input Data'}
                        )
    parser.add_argument('-b', '--binary',
                        metavar='Binary Output',
                        action='store_true',
                        default=False,
                        dest='binary',
                        help='The mask output will only be 0 or 1.'
                        )
    parser.add_argument('-p', '--post_process',
                        metavar='Apply Post Processing',
                        action='store_true',
                        default=True,
                        dest='post_process',
                        help='Remove all but the two largest regions of the '
                             'mask.'
                        )
    parser.add_argument('-r', '--raw',
                        metavar='Output Raw Data',
                        action='store_true',
                        default=False,
                        dest='raw',
                        help='Output the raw data used for the segmentation.'
                        )
    parser.add_argument('-output',
                        metavar='Output file',
                        default=None,
                        help='The name and location of your output mask. ('
                             'Default is to save with input data)',
                        widget='FileSaver',
                        gooey_options={'wildcard':
                                       'Compressed Nifti (*.nii.gz)|*.nii.gz|'
                                       'Nifti (*.nii)|*.nii|'
                                       'Analyze (*.hdr/*.img)|*.hdr|'
                                       'All files (*.*)|*.*',
                                       'message': "Select Output"}
                        )
    return parser


def resource_path(relative_path):
    """ Get absolute path to resource, works for dev and for PyInstaller """
    try:
        # PyInstaller creates a temp folder and stores path in _MEIPASS
        base_path = sys._MEIPASS
    except Exception:
        base_path = os.path.abspath(".")

    return os.path.join(base_path, relative_path)


# This chunk means if arguments are passed to the script/executable then it
# runs via the command line and if no arguments are passed, it runs the GUI
if len(sys.argv) >= 2:
    if not '--ignore-gooey' in sys.argv:
        sys.argv.append('--ignore-gooey')


@Gooey(program_name='Renal Segmentor',
       image_dir=resource_path('./images'))
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

    if args.post_process:
        cleaned_mask = cleanup((mask > 0.05) * 1)
        mask[cleaned_mask < 0.5] = 0.0

    if args.binary:
        mask = (mask > 0.5) * 1

    # Output mask
    if not args.output:
        output_path = raw_data.directory + '/' + raw_data.base + '_mask.nii.gz'
    else:
        output_path = args.output

    if os.path.splitext(os.path.basename(output_path))[1] == '':
        output_path += '.nii.gz'

    mask_img = nib.Nifti1Image(mask, raw_data.affine)
    nib.save(mask_img, output_path)

    if args.raw:
        nib.save(raw_data.img, os.path.dirname(output_path) + '/' + raw_data.base + '.nii.gz')


if __name__ == "__main__":
    main()
