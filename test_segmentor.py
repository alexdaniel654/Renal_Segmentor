import pytest
import numpy as np
import nibabel as nib
import renal_segmentor as rs
from gooey.python_bindings import argparse_to_json

# Constants

SUB_01_IMG = nib.load('./test_data/test_sub_01.PAR', scaling='fp')
SUB_02_IMG = nib.load('./test_data/test_sub_02.PAR', scaling='fp')
SUB_01_DATA = SUB_01_IMG.get_fdata()
SUB_02_DATA = SUB_02_IMG.get_fdata()


# Fixtures


# @pytest.fixture
# def sub_01_data():
#     img = nib.load('./test_data/test_sub_01.PAR', scaling='fp')
#     data = img.get_fdata()
#     return data
#
#
# @pytest.fixture
# def sub_02_data():
#     img = nib.load('./test_data/test_sub_02.PAR', scaling='fp')
#     data = img.get_fdata()
#     return data

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
    return np.allclose(test, gold_image_stats, rtol=1e-2, atol=1e-5)


# Test Cases
# Rescale


@pytest.mark.parametrize('data, expected', [
    (SUB_01_DATA, [0.14769913868900825, 0.18441673990363622, 1.0, 0.0, 13, 6.659772184770346]),
    (SUB_02_DATA, [0.14961473411100507, 0.17965318394622729, 1.0, 0.0, 13, 5.8241968078437045])
])
def test_rescale(data, expected):
    rsdata = rs.rescale(data)
    assert same_image(rsdata, expected)


# Split Path


@pytest.mark.parametrize('path, expected', [
    ('foo.PAR', ['', 'foo', '.PAR']),
    ('foo.nii.gz', ['', 'foo', '.nii.gz']),
    ('./foo/bar.PAR', ['./foo', 'bar', '.PAR']),
    ('./foo/bar.nii.gz', ['./foo', 'bar', '.nii.gz'])
])
def test_split_path(path, expected):
    raw_data = rs.RawData(path)
    # directory, base, extension = rs.split_path(path)
    assert raw_data.directory == expected[0]
    assert raw_data.base == expected[1]
    assert raw_data.extension == expected[2]


# Pre_process

@pytest.mark.parametrize('data, expected', [
    (SUB_01_DATA, [0.14816738145449654, 0.18183894498589623, 1.0, 0.0, 1, 0.0]),
    (SUB_02_DATA, [0.15022795636734432, 0.17748050040287047, 1.0, 0.0, 1, 0.0])
])
def test_pre_process(data, expected):
    pre_processed = rs.pre_process_img(data)
    assert same_image(pre_processed, expected)


# Un Pre_process


@pytest.mark.parametrize('data, img, expected', [
    (SUB_01_DATA, SUB_01_IMG,
     [15585.19294043287, 10395.261922505786, 73537.98252322787, 0.0, 13, 675689.1686444802]),
    (SUB_02_DATA, SUB_02_IMG,
     [16805.856856592778, 9647.452068107865, 65574.50210122531, 1.0363210171026915, 13, 1904548.5458263867]),
    (rs.pre_process_img(SUB_01_DATA), SUB_01_IMG,
     [0.14817189592712668, 0.18117106283408443, 1.0, 0.0, 13, 3.6336586087168503]),
    (rs.pre_process_img(SUB_02_DATA), SUB_02_IMG,
     [0.15021906059512827, 0.17657091709956615, 1.0, 0.0, 13, 4.267108061292017])
])
def test_un_pre_process(data, img, expected):
    un_pre_processed = rs.un_pre_process(data, img)
    assert same_image(un_pre_processed, expected)


# Prediction

@pytest.mark.parametrize('data, expected', [
    (SUB_01_DATA, [0.04029935, 0.19563328, 1.0, 0.0, 1, 9.1820955e-05]),
    (SUB_02_DATA, [0.040863547, 0.1969865, 1.0, 0.0, 1, 7.7813864e-05])
])
def test_prediction(data, expected):
    pre_processed_data = rs.pre_process_img(data)
    prediction = rs.predict_mask(pre_processed_data)
    assert same_image(prediction, expected)


# Load data

@pytest.mark.parametrize('path, expected', [
    ('./test_data/test_sub_01.PAR',
     [15586.417648035316, 12314.402055019415, 95094.74873995132, 0.0, 13, 1851954.3729744065]),
    ('./test_data/test_sub_01.hdr',
     [15586.417648035316, 12314.402055019415, 95094.74873995132, 0.0, 13, 1851954.3729744065]),
    ('./test_data/test_sub_01.nii',
     [15586.417648035316, 12314.402055019415, 95094.74873995132, 0.0, 13, 1851954.3729744065]),
    ('./test_data/test_sub_01.nii.gz',
     [15586.417648035316, 12314.402055019415, 95094.74873995132, 0.0, 13, 1851954.3729744065]),
    ('./test_data/test_sub_02.PAR',
     [16804.754730733697, 11854.72306071763, 91807.66119103732, 0.0, 13, 1350029.8990467489]),
    ('./test_data/test_sub_02.hdr',
     [16804.754730733697, 11854.72306071763, 91807.66119103732, 0.0, 13, 1350029.8990467489]),
    ('./test_data/test_sub_02.nii',
     [16804.754730733697, 11854.72306071763, 91807.66119103732, 0.0, 13, 1350029.8990467489]),
    ('./test_data/test_sub_02.nii.gz',
     [16804.754730733697, 11854.72306071763, 91807.66119103732, 0.0, 13, 1350029.8990467489]),
])
def test_load(path, expected):
    raw_data = rs.RawData(path)
    raw_data.load()
    stats = image_stats(raw_data.data)
    assert same_image(stats, expected)


@pytest.mark.parametrize('action, widget, expected', [
    (1, 'FileChooser', {'id': 'input',
                        'type': 'FileChooser',
                        'required': True}),
    (2, 'CheckBox', {'id': '-b',
                     'type': 'CheckBox',
                     'required': False}),
    (3, 'CheckBox', {'id': '-r',
                     'type': 'CheckBox',
                     'required': False}),
    (4, 'FileSaver', {'id': '-output',
                      'type': 'FileSaver',
                      'required': False})
])
def test_parser(action, widget, expected):
    parser = rs.get_parser()
    assert len(parser._actions) == 5
    result = argparse_to_json.action_to_json(parser._actions[action], widget, {})
    assert expected.items() <= result.items()
