import h5py
from hashlib import md5
import nibabel as nib
import os
import warnings
import wget

DIR_DATA = os.path.join(os.path.dirname(os.path.realpath(__file__)),
                        'test_data')

# Set up directory to cache weights
if 'SEGMENT_HOME' in os.environ:
    segment_home = os.environ['SEGMENT_HOME']
else:
    segment_home = os.path.join(os.path.expanduser('~'), '.segment')
    os.makedirs(segment_home, exist_ok=True)


class Sub1:
    def __init__(self, file_format):
        if file_format == 'PAR':
            self.path = os.path.join(DIR_DATA, 'test_sub_01.' + file_format)
            self.img = nib.load(self.path, scaling='fp')
        else:
            self.path = os.path.join(DIR_DATA, 'test_sub_01.' + file_format)
            self.img = nib.load(self.path)
        self.data = self.img.get_fdata()


class Sub2:
    def __init__(self, file_format):
        if file_format == 'PAR':
            self.path = os.path.join(DIR_DATA, 'test_sub_02.' + file_format)
            self.img = nib.load(self.path, scaling='fp')
        else:
            self.path = os.path.join(DIR_DATA, 'test_sub_02.' + file_format)
            self.img = nib.load(self.path)
        self.data = self.img.get_fdata()


class Weights:
    def __init__(self):
        expected_weights_hash = 'cbd72abc0dfda034b0a83d22c9a8c924'
        self.target_file = os.path.join(segment_home, 'renal_segmentor.h5')
        already_exists = os.path.isfile(self.target_file)
        if not already_exists:
            print('Downloading model weights')
            wget.download('https://zenodo.org/record/17831252/files'
                          '/whole_kidney_cnn_v2_2_0.h5?download=1',
                          self.target_file)
            self._keras_compliance()
        self._check_hash(expected_weights_hash)
        self.path = self.target_file
        self.dir = segment_home

    def _check_hash(self, expected_hash):
        test_hash = self._get_file_md5(self.target_file)
        if test_hash != expected_hash:
            warnings.warn(f'The weigths in {segment_home} do not match the '
                          f'weights expected by this version of '
                          f'renal_segmentor. This could mean a new version '
                          f'of the weights is available and therefore you '
                          f'should also update renal_segmentor or that there '
                          f'was an error when downloading the weights')
            
    def _keras_compliance(self):
        f = h5py.File(self.target_file, 'r+')
        model_config_str = f.attrs.get('model_config')
        if model_config_str.find('"groups": 1, ') != -1:
            model_config_str = model_config_str.replace('"groups": 1, ', '')
            f.attrs.modify('model_config', model_config_str)
            f.flush()
            model_config_str = f.attrs.get("model_config")
            assert model_config_str.find('"groups": 1, ') == -1
        f.close()

    @staticmethod
    def _get_file_md5(filename):
        """Compute the md5 checksum of a file"""
        md5_data = md5()
        with open(filename, 'rb') as f:
            for chunk in iter(lambda: f.read(128 * md5_data.block_size), b''):
                md5_data.update(chunk)
        return md5_data.hexdigest()
