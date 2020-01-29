import pytest
import numpy as np
import nibabel as nib
import renal_segmentor as rs

# Constants


SUB_01_DATA = nib.load('./test_data/test_sub_01.PAR', scaling='fp').get_fdata()
SUB_02_DATA = nib.load('./test_data/test_sub_02.PAR', scaling='fp').get_fdata()

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
    return [mean, std, max, min]


def same_image(test, gold_image_stats):
    if type(test) != list and len(test) != 4:
        test = image_stats(test)
    return np.allclose(test, gold_image_stats, rtol=1e-2, atol=1e-5)

# Test Cases
# Rescale


@pytest.mark.parametrize('subject, expected', [
    (SUB_01_DATA, [0.14769913868900825, 0.18441673990363622, 1.0, 0.0]),
    (SUB_02_DATA, [0.14961473411100507, 0.17965318394622729, 1.0, 0.0])
])
def test_rescale(subject, expected):
    rsdata = rs.rescale(subject)
    assert same_image(rsdata, expected)
# Split Path


def test_file_simple_extension():
    path = 'foo.PAR'
    directory, base, extension = rs.split_path(path)
    assert directory == ''
    assert base == 'foo'
    assert extension == '.PAR'


def test_file_nifti_extension():
    path = 'foo.nii.gz'
    directory, base, extension = rs.split_path(path)
    assert directory == ''
    assert base == 'foo'
    assert extension == '.nii.gz'


def test_path_simple_extension():
    path = './foo/bar.PAR'
    directory, base, extension = rs.split_path(path)
    assert directory == './foo'
    assert base == 'bar'
    assert extension == '.PAR'


def test_path_nifti_extension():
    path = './foo/bar.nii.gz'
    directory, base, extension = rs.split_path(path)
    assert directory == './foo'
    assert base == 'bar'
    assert extension == '.nii.gz'
