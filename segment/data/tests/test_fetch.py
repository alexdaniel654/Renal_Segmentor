import numpy as np
import os
import pytest

from nibabel.parrec import PARRECImage
from nibabel.nifti1 import Nifti1Image
from nibabel.analyze import AnalyzeImage
from segment.data import fetch


class TestSub1:

    def test_sub1_par(self):
        sub1 = fetch.Sub1('PAR')
        assert isinstance(sub1.img, PARRECImage)
        assert isinstance(sub1.path, str)
        assert sub1.path.endswith('.PAR')
        assert isinstance(sub1.data, np.ndarray)
        assert len(np.shape(sub1.data)) == 3

    def test_sub1_nii(self):
        sub1 = fetch.Sub1('nii')
        assert isinstance(sub1.img, Nifti1Image)
        assert isinstance(sub1.path, str)
        assert sub1.path.endswith('.nii')
        assert isinstance(sub1.data, np.ndarray)
        assert len(np.shape(sub1.data)) == 3

    def test_sub1_niigz(self):
        sub1 = fetch.Sub1('nii.gz')
        assert isinstance(sub1.img, Nifti1Image)
        assert isinstance(sub1.path, str)
        assert sub1.path.endswith('.nii.gz')
        assert isinstance(sub1.data, np.ndarray)
        assert len(np.shape(sub1.data)) == 3

    def test_sub1_imghdr(self):
        sub1 = fetch.Sub1('img')
        assert isinstance(sub1.img, AnalyzeImage)
        assert isinstance(sub1.path, str)
        assert sub1.path.endswith('.img')
        assert isinstance(sub1.data, np.ndarray)
        assert len(np.shape(sub1.data)) == 3


class TestSub2:

    def test_sub2_par(self):
        sub2 = fetch.Sub2('PAR')
        assert isinstance(sub2.img, PARRECImage)
        assert isinstance(sub2.path, str)
        assert sub2.path.endswith('.PAR')
        assert isinstance(sub2.data, np.ndarray)
        assert len(np.shape(sub2.data)) == 3

    def test_sub2_nii(self):
        sub2 = fetch.Sub2('nii')
        assert isinstance(sub2.img, Nifti1Image)
        assert isinstance(sub2.path, str)
        assert sub2.path.endswith('.nii')
        assert isinstance(sub2.data, np.ndarray)
        assert len(np.shape(sub2.data)) == 3

    def test_sub2_niigz(self):
        sub2 = fetch.Sub2('nii.gz')
        assert isinstance(sub2.img, Nifti1Image)
        assert isinstance(sub2.path, str)
        assert sub2.path.endswith('.nii.gz')
        assert isinstance(sub2.data, np.ndarray)
        assert len(np.shape(sub2.data)) == 3

    def test_sub2_imghdr(self):
        sub2 = fetch.Sub2('img')
        assert isinstance(sub2.img, AnalyzeImage)
        assert isinstance(sub2.path, str)
        assert sub2.path.endswith('.img')
        assert isinstance(sub2.data, np.ndarray)
        assert len(np.shape(sub2.data)) == 3


class TestWeights:

    def test_get_filt_md5(self):
        with open('test.txt', 'w') as f:
            f.write('test file')
        assert fetch.Weights._get_file_md5('test.txt') == \
               'f20d9f2072bbeb6691c0f9c5099b01f3'
        os.remove('test.txt')

    def test_weights(self):
        weights = fetch.Weights()
        assert isinstance(weights.path, str)
        assert weights.path.endswith('.model')
        assert isinstance(weights.dir, str)
        assert os.path.isfile(weights.path)

        # Remove downloaded weights and replace with another file to verify
        # hash discrepancy warning
        os.remove(weights.path)
        with open(weights.path, 'w') as f:
            f.write('simulated partial download or deprecated weights.')

        with pytest.warns(UserWarning):
            weights = fetch.Weights()

        # Remove incorrect weights
        os.remove(weights.path)
