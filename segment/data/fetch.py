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
        expected_weights_hash = 'e9f60f9fe6ad9eced8b055bf6792a1d1'
        self.target_file = os.path.join(segment_home, 'renal_segmentor.model')
        already_exists = os.path.isfile(self.target_file)
        if not already_exists:
            print('Downloading model weights')
            wget.download('https://zenodo.org/record/4894406/files'
                          '/whole_kidney_cnn.model?download=1',
                          self.target_file)
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

    @staticmethod
    def _get_file_md5(filename):
        """Compute the md5 checksum of a file"""
        md5_data = md5()
        with open(filename, 'rb') as f:
            for chunk in iter(lambda: f.read(128 * md5_data.block_size), b''):
                md5_data.update(chunk)
        return md5_data.hexdigest()
