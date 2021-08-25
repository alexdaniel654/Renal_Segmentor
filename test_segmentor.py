import numpy as np
import numpy.testing as npt
import nibabel as nib
import os
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
    npt.assert_allclose(test, gold_image_stats, rtol=1e-2, atol=1e-5)


# Test Cases

# Load data
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
def test_load(path, expected):
    raw_data = rs.RawData(path)
    raw_data.load()
    stats = image_stats(raw_data.data)
    same_image(stats, expected)


# Rescale
@pytest.mark.parametrize('data, expected', [
    (SUB_01_DATA, [0.14769913868900825, 0.18441673990363622, 1.0, 0.0, 13,
                   6.659772184770346]),
    (SUB_02_DATA, [0.126408, 0.198033, 1.0, 0.0, 17.0, 20.621155])
])
def test_rescale(data, expected):
    rsdata = rs.RawData._rescale(data)
    same_image(rsdata, expected)


# Split Path
@pytest.mark.parametrize('path, expected', [
    ('foo.PAR', ['', 'foo', '.PAR']),
    ('foo.nii.gz', ['', 'foo', '.nii.gz']),
    ('./foo/bar.PAR', ['./foo', 'bar', '.PAR']),
    ('./foo/bar.nii.gz', ['./foo', 'bar', '.nii.gz'])
])
def test_split_path(path, expected):
    raw_data = rs.RawData(path)
    assert raw_data.directory == expected[0]
    assert raw_data.base == expected[1]
    assert raw_data.extension == expected[2]


# Get Mask
@pytest.mark.parametrize('path, expected, expected_cleaned', [
    ('./test_data/test_sub_01.PAR',
     [4.022243e-02, 1.939664e-01, 1.004862e+00, -7.716416e-03,
      1.300000e+01, 0.000000e+00],
     [0.014793, 0.120724, 1.0, 0.0, 13.0, 0.0]),
    ('./test_data/test_sub_01.nii.gz',
     [4.020664e-02, 1.939301e-01, 1.004896e+00, -7.716225e-03,
      1.300000e+01, 0.000000e+00],
     [0.015724, 0.124405, 1.0, 0.0, 13.0, 0.0]),
    ('./test_data/test_sub_02.PAR',
     [0.018624, 0.133018, 1.165945, -0.138861, 17.0, 0.0],
     [6.461648e-03, 8.012425e-02, 1.000000e+00, 0.000000e+00, 1.700000e+01,
      0.000000e+00]),
    ('./test_data/test_sub_02.nii.gz',
     [0.018624, 0.133018, 1.165945, -0.138861, 17.0, 0.0],
     [6.461648e-03, 8.012425e-02, 1.000000e+00, 0.000000e+00, 1.700000e+01,
      0.000000e+00]),
])
def test_get_mask(path, expected, expected_cleaned):
    raw_data = rs.RawData(path)
    raw_data.load()
    prediction = raw_data.get_mask()
    prediction_cleaned = rs.cleanup(prediction)
    same_image(prediction, expected)
    same_image(prediction_cleaned, expected_cleaned)


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
    (5, 'DirChooser', {'id': '-output',
                       'type': 'DirChooser',
                       'required': False})
])
def test_parser(action, widget, expected):
    parser = rs.get_parser()
    assert len(parser._actions) == 6
    result = argparse_to_json.action_to_json(parser._actions[action], widget,
                                             {})
    assert expected.items() <= result.items()


def test_segment_cli():
    os.makedirs('test_output', exist_ok=True)

    # One input file
    test_args = ['renal_segmentor',
                 './test_data/test_sub_01.PAR',
                 '-output', 'test_output']
    with patch.object(sys, 'argv', test_args):
        rs.main()
        assert os.path.exists('test_output/test_sub_01_mask.nii.gz') == 1

        for f in os.listdir('test_output'):
            os.remove(os.path.join('test_output', f))

    # Multiple input files
    test_args = ['renal_segmentor',
                 './test_data/test_sub_01.PAR',
                 './test_data/test_sub_02.PAR',
                 '-output', 'test_output']
    with patch.object(sys, 'argv', test_args):
        rs.main()
        assert os.path.exists('test_output/test_sub_01_mask.nii.gz') == 1
        assert os.path.exists('test_output/test_sub_02_mask.nii.gz') == 1

        for f in os.listdir('test_output'):
            os.remove(os.path.join('test_output', f))

    shutil.rmtree('test_output')


