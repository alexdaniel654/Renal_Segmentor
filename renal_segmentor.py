# -*- coding: utf-8 -*-
"""
Created on Fri Jul 12 11:25:34 2019

@author: Alex Daniel
"""

# Import Packages

import nibabel as nib
import os
import pandas as pd
import sys
import tensorflow as tf

from gooey import Gooey, GooeyParser
from segment import Tkv
from segment.data import fetch

# Define Functions


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
                        default=False,
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
    parser.add_argument('-v', '--volumes',
                        metavar='Export Kidney Volumes',
                        action='store_true',
                        default=False,
                        dest='volumes',
                        help='Export the total, left and right kidney '
                             'volumes to a csv.'
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
       default_size=(610, 640),
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
                'version': '1.2.0',
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
    volumes = pd.DataFrame(index=args.input, columns=['tkv', 'lkv', 'rkv'])
    n = 0
    for path, row in volumes.iterrows():
        n += 1
        segmentation = Tkv(path)

        mask = segmentation.get_mask(post_process=args.post_process)

        volumes.loc[path, 'tkv'] = segmentation.tkv
        volumes.loc[path, 'lkv'] = segmentation.lkv
        volumes.loc[path, 'rkv'] = segmentation.rkv

        if args.binary:
            mask = (mask > 0.5) * 1

        # Output mask
        if not args.output:
            out_dir = segmentation.directory
        else:
            out_dir = args.output

        mask_fname = os.path.join(out_dir, segmentation.base + '_mask.nii.gz')

        mask_img = nib.Nifti1Image(mask, segmentation.affine)
        nib.save(mask_img, mask_fname)

        if args.raw:
            nib.save(segmentation._img, os.path.join(out_dir, segmentation.base
                                                     + '.nii.gz'))

        print(f'Processed {n} of {len(volumes)} files')

    if args.volumes:
        volumes.to_csv(os.path.join(out_dir, 'volumes.csv'),
                       index_label='File',
                       header=['TKV (ml)', 'LKV (ml)', 'RKV (ml)'])


if __name__ == "__main__":
    main()
