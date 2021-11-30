import nibabel as nib
import os
import pytest
import shutil

from segment import Tkv
from segment.data import fetch
from .utils import image_stats, same_image


class TestRawData:
    @pytest.mark.parametrize('path, expected', [
        (fetch.Sub1('PAR').path,
         [15586.417648035316, 12314.402055019415, 95094.74873995132, 0.0, 13,
          1851954.3729744065]),
        (fetch.Sub1('img').path,
         [15586.417648035316, 12314.402055019415, 95094.74873995132, 0.0, 13,
          1851954.3729744065]),
        (fetch.Sub1('nii').path,
         [15586.417648035316, 12314.402055019415, 95094.74873995132, 0.0, 13,
          1851954.3729744065]),
        (fetch.Sub1('nii.gz').path,
         [15586.417648035316, 12314.402055019415, 95094.74873995132, 0.0, 13,
          1851954.3729744065]),
        (fetch.Sub2('PAR').path,
         [9.148690e+03, 1.237672e+04, 1.338610e+05, 0.000000e+00, 1.700000e+01,
          1.591041e+06]),
        (fetch.Sub2('img').path,
         [9.148663e+03, 1.237674e+04, 1.338610e+05, 0.000000e+00, 1.700000e+01,
          1.591044e+06]),
        (fetch.Sub2('nii').path,
         [9.148663e+03, 1.237674e+04, 1.338610e+05, 0.000000e+00, 1.700000e+01,
          1.591044e+06]),
        (fetch.Sub2('nii.gz').path,
         [9.148663e+03, 1.237674e+04, 1.338610e+05, 0.000000e+00, 1.700000e+01,
          1.591044e+06]),
    ])
    def test_load(self, path, expected):
        raw_data = Tkv(path)
        stats = image_stats(raw_data.data)
        same_image(stats, expected)

    @pytest.mark.parametrize('data, expected', [
        (fetch.Sub1('PAR').data,
         [0.14769913868900825, 0.18441673990363622, 1.0, 0.0, 13,
          6.659772184770346]),
        (fetch.Sub2('PAR').data,
         [0.126408, 0.198033, 1.0, 0.0, 17.0, 20.621155])
    ])
    def test_rescale(self, data, expected):
        rsdata = Tkv._rescale(data)
        same_image(rsdata, expected)

    @pytest.mark.parametrize('path, expected', [
        ('foo.PAR', ['', 'foo', '.PAR']),
        ('foo.nii.gz', ['', 'foo', '.nii.gz']),
        ('./foo/bar.PAR', ['./foo', 'bar', '.PAR']),
        ('./foo/bar.nii.gz', ['./foo', 'bar', '.nii.gz'])
    ])
    def test_split_path(self, path, expected):
        directory, base, extension = Tkv._split_path(path)
        assert directory == expected[0]
        assert base == expected[1]
        assert extension == expected[2]

    @pytest.mark.parametrize('path, expected, expected_cleaned', [
        (fetch.Sub1('PAR').path,
         [0.040264, 0.19392, 1.0,  0.0, 13.0, 0.0],
         [0.040212, 0.193889, 1.0, 0.0, 13.0, 0.0]),
        (fetch.Sub1('nii.gz').path,
         [0.040264, 0.19392, 1.0,  0.0, 13.0, 0.0],
         [0.040212, 0.193889, 1.0, 0.0, 13.0, 0.0]),
        (fetch.Sub2('PAR').path,
         [0.018649, 0.13261, 1.0, 0.0, 17.0, 0.0],
         [0.018423, 0.131926, 1.0, 0.0, 17.0, 0.0]),
        (fetch.Sub2('nii.gz').path,
         [0.018649, 0.13261, 1.0, 0.0, 17.0, 0.0],
         [0.018423, 0.131926, 1.0, 0.0, 17.0, 0.0]),
    ])
    def test_get_mask(self, path, expected, expected_cleaned):
        raw_data = Tkv(path)
        prediction = raw_data.get_mask(post_process=False)
        prediction_cleaned = raw_data.get_mask(post_process=True)
        same_image(prediction, expected)
        same_image(prediction_cleaned, expected_cleaned)

    @pytest.mark.parametrize('path, expected', [
        (fetch.Sub1('PAR').path,
         [0.040212, 0.193889, 1.0, 0.0, 13.0, 0.0]),
        (fetch.Sub2('PAR').path,
         [0.018423, 0.131926, 1.0, 0.0, 17.0, 0.0])
    ])
    def test_mask_to_nifti(self, path, expected):
        directory = os.path.dirname(path)
        base = os.path.splitext(os.path.basename(path))[0]
        raw_data = Tkv(path)
        prediction = raw_data.get_mask()

        # Default name
        raw_data.mask_to_nifti()
        output_files = os.listdir(directory)
        expected_output = base + '_mask.nii.gz'
        assert expected_output in output_files
        saved_data = nib.load(path[:-4] + '_mask.nii.gz').get_fdata()
        same_image(saved_data, expected)
        os.remove(path[:-4] + '_mask.nii.gz')

        # Custom name
        raw_data.mask_to_nifti(os.path.join(directory,
                                            base + '_automated_mask.nii.gz'))
        output_files = os.listdir(directory)
        assert base + '_automated_mask.nii.gz' in output_files
        saved_data = nib.load(
            os.path.join(directory,
                         base + '_automated_mask.nii.gz')).get_fdata()
        same_image(saved_data, expected)
        os.remove(os.path.join(directory, base + '_automated_mask.nii.gz'))

    @pytest.mark.parametrize('path, expected', [
        (fetch.Sub1('PAR').path,
         [1.558641e+04, 1.231441e+04, 9.509475e+04, 0.000000e+00,
         1.300000e+01, 1.851950e+06]),
        (fetch.Sub2('PAR').path,
         [9.148663e+03, 1.237674e+04, 1.338610e+05, 0.000000e+00,
          1.700000e+01, 1.591044e+06])
    ])
    def test_image_to_nifti(self, path, expected):
        directory = os.path.dirname(path)
        base = os.path.splitext(os.path.basename(path))[0]
        raw_data = Tkv(path)
        # Custom name (avoids overwriting existing data)
        raw_data.data_to_nifti(os.path.join(directory,
                                            base + '_raw_data.nii.gz'))
        output_files = os.listdir(directory)
        assert base + '_raw_data.nii.gz' in output_files
        saved_data = nib.load(
            os.path.join(directory,
                         base + '_raw_data.nii.gz')).get_fdata()
        same_image(saved_data, expected)
        os.remove(os.path.join(directory, base + '_raw_data.nii.gz'))
