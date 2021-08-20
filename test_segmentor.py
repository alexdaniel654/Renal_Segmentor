import pytest
import numpy as np
import numpy.testing as npt
import nibabel as nib
import renal_segmentor as rs
from gooey.python_bindings import argparse_to_json

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
     [16804.754730733697, 11854.72306071763, 91807.66119103732, 0.0, 13,
      1350029.8990467489]),
    ('./test_data/test_sub_02.hdr',
     [16804.754730733697, 11854.72306071763, 91807.66119103732, 0.0, 13,
      1350029.8990467489]),
    ('./test_data/test_sub_02.nii',
     [16804.754730733697, 11854.72306071763, 91807.66119103732, 0.0, 13,
      1350029.8990467489]),
    ('./test_data/test_sub_02.nii.gz',
     [16804.754730733697, 11854.72306071763, 91807.66119103732, 0.0, 13,
      1350029.8990467489]),
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
    (SUB_02_DATA, [0.14961473411100507, 0.17965318394622729, 1.0, 0.0, 13,
                   5.8241968078437045])
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
     [4.023278e-02, 1.940837e-01, 1.005738e+00, -6.771960e-03,
      1.300000e+01, 0.000000e+00],
     [0.014247, 0.118507, 1.0, 0.0, 13.0, 0.0]),
    ('./test_data/test_sub_02.nii.gz',
     [4.023382e-02, 1.940864e-01, 1.005738e+00, -6.772716e-03,
      1.300000e+01, 0.000000e+00],
     [0.014236, 0.118463, 1.0, 0.0, 13.0, 0.0]),
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
    (5, 'FileSaver', {'id': '-output',
                      'type': 'FileSaver',
                      'required': False})
])
def test_parser(action, widget, expected):
    parser = rs.get_parser()
    assert len(parser._actions) == 6
    result = argparse_to_json.action_to_json(parser._actions[action], widget,
                                             {})
    assert expected.items() <= result.items()
