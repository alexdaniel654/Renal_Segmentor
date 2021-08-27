import numpy as np
import numpy.testing as npt
import nibabel as nib
import os
import pandas as pd
import pytest
import renal_segmentor as rs
import shutil
import sys

from gooey.python_bindings import argparse_to_json
from unittest.mock import patch

# Constants

SUB_01_IMG = nib.load('./test_data/test_sub_01.PAR', scaling='fp')
SUB_02_IMG = nib.load('./test_data/test_sub_02.PAR', scaling='fp')
SUB_01_DATA = SUB_01_IMG.get_fdata()
SUB_02_DATA = SUB_02_IMG.get_fdata()

if os.path.exists('test_output'):
    shutil.rmtree('test_output')

# Helpers


def image_stats(data):
    mean = np.nanmean(data)
    std = np.nanstd(data)
    max = np.nanmax(data)
    min = np.nanmin(data)
    dim = data.shape[-1]
    row_sum = np.sum(data, axis=0).reshape(-1)[0]
    return [mean, std, max, min, dim, row_sum]


def same_image(test, gold_image_stats):
    if type(test) != list and len(test) != 6:
        test = image_stats(test)
    npt.assert_allclose(test, gold_image_stats, rtol=1e-3, atol=1e-9)


# Test Cases

class TestRawData:
    @pytest.mark.parametrize('path, expected', [
        ('./test_data/test_sub_01.PAR',
         [15586.417648035316, 12314.402055019415, 95094.74873995132, 0.0, 13,
          1851954.3729744065]),
        ('./test_data/test_sub_01.hdr',
         [15586.417648035316, 12314.402055019415, 95094.74873995132, 0.0, 13,
          1851954.3729744065]),
        ('./test_data/test_sub_01.nii',
         [15586.417648035316, 12314.402055019415, 95094.74873995132, 0.0, 13,
          1851954.3729744065]),
        ('./test_data/test_sub_01.nii.gz',
         [15586.417648035316, 12314.402055019415, 95094.74873995132, 0.0, 13,
          1851954.3729744065]),
        ('./test_data/test_sub_02.PAR',
         [9.148690e+03, 1.237672e+04, 1.338610e+05, 0.000000e+00, 1.700000e+01,
          1.591041e+06]),
        ('./test_data/test_sub_02.hdr',
         [9.148663e+03, 1.237674e+04, 1.338610e+05, 0.000000e+00, 1.700000e+01,
          1.591044e+06]),
        ('./test_data/test_sub_02.nii',
         [9.148663e+03, 1.237674e+04, 1.338610e+05, 0.000000e+00, 1.700000e+01,
          1.591044e+06]),
        ('./test_data/test_sub_02.nii.gz',
         [9.148663e+03, 1.237674e+04, 1.338610e+05, 0.000000e+00, 1.700000e+01,
          1.591044e+06]),
    ])
    def test_load(self, path, expected):
        raw_data = rs.RawData(path)
        raw_data.load()
        stats = image_stats(raw_data.data)
        same_image(stats, expected)

    @pytest.mark.parametrize('data, expected', [
        (SUB_01_DATA, [0.14769913868900825, 0.18441673990363622, 1.0, 0.0, 13,
                       6.659772184770346]),
        (SUB_02_DATA, [0.126408, 0.198033, 1.0, 0.0, 17.0, 20.621155])
    ])
    def test_rescale(self, data, expected):
        rsdata = rs.RawData._rescale(data)
        same_image(rsdata, expected)

    @pytest.mark.parametrize('path, expected', [
        ('foo.PAR', ['', 'foo', '.PAR']),
        ('foo.nii.gz', ['', 'foo', '.nii.gz']),
        ('./foo/bar.PAR', ['./foo', 'bar', '.PAR']),
        ('./foo/bar.nii.gz', ['./foo', 'bar', '.nii.gz'])
    ])
    def test_split_path(self, path, expected):
        raw_data = rs.RawData(path)
        assert raw_data.directory == expected[0]
        assert raw_data.base == expected[1]
        assert raw_data.extension == expected[2]

    @pytest.mark.parametrize('path, expected, expected_cleaned', [
        ('./test_data/test_sub_01.PAR',
         [0.040264, 0.19392, 1.0,  0.0, 13.0, 0.0],
         [0.040212, 0.193889, 1.0, 0.0, 13.0, 0.0]),
        ('./test_data/test_sub_01.nii.gz',
         [0.040264, 0.19392, 1.0,  0.0, 13.0, 0.0],
         [0.040212, 0.193889, 1.0, 0.0, 13.0, 0.0]),
        ('./test_data/test_sub_02.PAR',
         [0.018649, 0.13261, 1.0, 0.0, 17.0, 0.0],
         [0.018423, 0.131926, 1.0, 0.0, 17.0, 0.0]),
        ('./test_data/test_sub_02.nii.gz',
         [0.018649, 0.13261, 1.0, 0.0, 17.0, 0.0],
         [0.018423, 0.131926, 1.0, 0.0, 17.0, 0.0]),
    ])
    def test_get_mask(self, path, expected, expected_cleaned):
        raw_data = rs.RawData(path)
        raw_data.load()
        prediction = raw_data.get_mask(post_process=False)
        prediction_cleaned = raw_data.get_mask(post_process=True)
        same_image(prediction, expected)
        same_image(prediction_cleaned, expected_cleaned)


class TestParser:
    @pytest.mark.parametrize('action, widget, expected', [
        (1, 'FileChooser', {'id': 'input',
                            'type': 'FileChooser',
                            'required': True}),
        (2, 'CheckBox', {'id': '-b',
                         'type': 'CheckBox',
                         'required': False}),
        (3, 'CheckBox', {'id': '-p',
                         'type': 'CheckBox',
                         'required': False}),
        (4, 'CheckBox', {'id': '-r',
                         'type': 'CheckBox',
                         'required': False}),
        (5, 'CheckBox', {'id': '-v',
                         'type': 'CheckBox',
                         'required': False}),
        (6, 'DirChooser', {'id': '-output',
                           'type': 'DirChooser',
                           'required': False})
    ])
    def test_parser(self, action, widget, expected):
        parser = rs.get_parser()
        assert len(parser._actions) == 7
        result = argparse_to_json.action_to_json(parser._actions[action],
                                                 widget, {})
        assert expected.items() <= result.items()


class TestCli:
    def test_single_file(self):
        test_args = ['renal_segmentor',
                     './test_data/test_sub_01.PAR',
                     '-output', 'test_output']
        os.makedirs('test_output', exist_ok=True)
        with patch.object(sys, 'argv', test_args):
            rs.main()
            output_files = os.listdir('test_output')
            assert len(output_files) == 1
            assert os.path.exists('test_output/test_sub_01_mask.nii.gz') == 1

            for f in os.listdir('test_output'):
                os.remove(os.path.join('test_output', f))
            shutil.rmtree('test_output')

    def test_multiple_files(self):
        test_args = ['renal_segmentor',
                     './test_data/test_sub_01.PAR',
                     './test_data/test_sub_02.PAR',
                     '-output', 'test_output', '-v', '-r']
        os.makedirs('test_output', exist_ok=True)
        with patch.object(sys, 'argv', test_args):
            rs.main()

            output_files = os.listdir('test_output')
            assert len(output_files) == 5
            assert os.path.exists('test_output/test_sub_01_mask.nii.gz') == 1
            assert os.path.exists('test_output/test_sub_02_mask.nii.gz') == 1
            assert os.path.exists('test_output/test_sub_01.nii.gz') == 1
            assert os.path.exists('test_output/test_sub_02.nii.gz') == 1
            assert os.path.exists('test_output/volumes.csv') == 1

            volumes = pd.read_csv('test_output/volumes.csv')

            npt.assert_allclose(volumes['TKV (ml)'].mean(), 304.663341)
            npt.assert_allclose(volumes['LKV (ml)'].mean(), 159.170390)
            npt.assert_allclose(volumes['RKV (ml)'].mean(), 145.492951)

            mask_sub_01 = nib.load(
                'test_output/test_sub_01_mask.nii.gz').get_fdata()
            mask_sub_02 = nib.load(
                'test_output/test_sub_02_mask.nii.gz').get_fdata()

            same_image(mask_sub_01, [0.040264, 0.19392, 1.0,  0.0, 13.0, 0.0])
            same_image(mask_sub_02, [0.018649, 0.13261, 1.0, 0.0, 17.0, 0.0])

            for f in os.listdir('test_output'):
                os.remove(os.path.join('test_output', f))
            shutil.rmtree('test_output')
