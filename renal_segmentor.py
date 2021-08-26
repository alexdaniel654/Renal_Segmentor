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
from nibabel.processing import conform
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
        self.mask = np.array
        self.affine = np.array
        self.shape = tuple
        self.zoom = tuple

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
        self.shape = self.img.shape
        self.zoom = self.img.header.get_zooms()
        self.orientation = nib.orientations.aff2axcodes(self.affine)

    def get_mask(self, weights_path='./models/renal_segmentor.model'):
        img = conform(self.img, out_shape=(240, 240, self.shape[-1]),
                      voxel_size=(1.458, 1.458, self.zoom[-1] * 0.998),
                      orientation='LIP')
        data = img.get_fdata()
        data = np.flip(data, 1)
        data = np.swapaxes(data, 0, 2)
        data = np.swapaxes(data, 1, 2)
        data = self._rescale(data)
        data = resize(data, (data.shape[0], 256, 256))
        data = data.reshape((data.shape[0], data.shape[1], data.shape[2], 1))
        model = load_model(resource_path(weights_path),
                           custom_objects={'dice_coef_loss':
                                           self._dice_coef_loss,
                                           'dice_coef': self._dice_coef})
        batch_size = 2 ** 3
        mask = model.predict(data, batch_size=batch_size)
        mask = np.squeeze(mask)
        mask = np.swapaxes(mask, 0, 2)
        mask = np.swapaxes(mask, 0, 1)
        mask = np.flip(mask, 1)
        mask = resize(mask, (240, 240, self.shape[-1]))
        mask_img = nib.Nifti1Image(mask, img.affine)
        mask_img = conform(mask_img, out_shape=self.shape,
                           voxel_size=self.zoom, orientation=self.orientation)
        self.mask = mask_img.get_fdata()
        return self.mask

    @staticmethod
    def _rescale(data):
        black = np.mean(data) - 0.5 * np.std(data)
        if black < data.min():
            black = data.min()
        white = np.mean(data) + 4 * np.std(data)
        if white > data.max():
            white = data.max()
        data = np.clip(data, black, white) - black
        data = data / (white - black)
        return data

    @staticmethod
    def _dice_coef(y_true, y_pred):
        smooth = 1.0
        y_true_f = K.flatten(y_true)
        y_pred_f = K.flatten(y_pred)
        intersection = K.sum(y_true_f * y_pred_f)
        return (2. * intersection + smooth) / (
                    K.sum(y_true_f) + K.sum(y_pred_f) + smooth)

    def _dice_coef_loss(self, y_true, y_pred):
        loss = 1 - self._dice_coef(y_true, y_pred)
        return loss


# Define Functions


def cleanup(mask):
    clean_mask = np.zeros(mask.shape, dtype=np.uint8)
    label_mask = label(mask, connectivity=1)
    props = regionprops(label_mask)
    areas = [region.area for region in props]

    # This means there have to be two kidneys in the scan...
    kidney_labels = np.argpartition(areas, -2)[-2:]

    clean_mask[label_mask == props[kidney_labels[0]].label] = 1
    clean_mask[label_mask == props[kidney_labels[1]].label] = 1

    return clean_mask


def get_parser():
    # Make argparser
    parser = GooeyParser(description='Segment renal MRI images.')
    parser.add_argument('input',
                        metavar='Input Data',
                        help='The image you wish to segment.',
                        nargs='*',
                        widget='MultiFileChooser',
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
                        metavar='Output Directory',
                        default=None,
                        help='The location to save outputs. (Default is '
                             'to save with input data)',
                        widget='DirChooser',
                        gooey_options={'full_width': True,
                                       'message': "Select Output Directory"}
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
       image_dir=resource_path('./images'),
       default_size=(610, 620),
       progress_regex=r"^Processed (?P<current>\d+) of (?P<total>\d+) files$",
       progress_expr="current / total * 100",
       timing_options={'show_time_remaining': True,
                       'hide_time_remaining_on_complete': True},
       menu=[{
           'name': 'File',
           'items': [{
                'type': 'AboutDialog',
                'menuTitle': 'About',
                'name': 'Renal Segmentor',
                'description': 'Automatically segment the kidneys from MRI '
                               'data.',
                'version': '1.1.0',
                'copyright': '2021',
                'website': 'https://github.com/alexdaniel654/Renal_Segmentor',
                'developer': 'https://www.researchgate.net/profile/'
                             'Alexander-Daniel-2',
                'license': 'GPLv3'
           }, {
               'type': 'Link',
               'menuTitle': 'Check for New Versions',
               'url': 'https://github.com/alexdaniel654/Renal_Segmentor'
                      '/releases/latest'
           }]
       }, {
           'name': 'Help',
           'items': [{
               'type': 'Link',
               'menuTitle': 'Documentation',
               'url': 'https://github.com/alexdaniel654/'
                      'Renal_Segmentor#renal-segmentor'
           }, {
               'type': 'Link',
               'menuTitle': 'Report an Issue',
               'url': 'https://github.com/alexdaniel654/Renal_Segmentor/issues'
           }]
       }]
       )
def main():
    tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

    parser = get_parser()
    args = parser.parse_args()

    # Import data
    inputs = args.input
    for n, file in enumerate(inputs):
        raw_data = RawData(file)
        raw_data.load()

        mask = raw_data.get_mask()

        if args.post_process:
            cleaned_mask = cleanup((mask > 0.05) * 1)
            mask[cleaned_mask < 0.5] = 0.0

        if args.binary:
            mask = (mask > 0.5) * 1

        # Output mask
        if not args.output:
            out_dir = raw_data.directory
        else:
            out_dir = args.output
        mask_fname = os.path.join(out_dir, raw_data.base + '_mask.nii.gz')

        mask_img = nib.Nifti1Image(mask, raw_data.affine)
        nib.save(mask_img, mask_fname)

        if args.raw:
            nib.save(raw_data.img, os.path.join(out_dir, raw_data.base +
                                                '.nii.gz'))

        print(f'Processed {n + 1} of {len(inputs)} files')


if __name__ == "__main__":
    main()
