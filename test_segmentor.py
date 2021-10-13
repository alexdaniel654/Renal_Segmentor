import numpy.testing as npt
import nibabel as nib
import os
import pandas as pd
import pytest
import renal_segmentor as rs
import shutil
import sys

from gooey.python_bindings import argparse_to_json
from segment.data import fetch
from segment.tests.utils import same_image
from unittest.mock import patch


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
                     fetch.Sub1('PAR').path,
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
                     fetch.Sub1('PAR').path,
                     fetch.Sub2('PAR').path,
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
