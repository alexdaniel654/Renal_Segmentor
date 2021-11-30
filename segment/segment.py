import nibabel as nib
import numpy as np
import os

from nibabel.processing import conform
from segment.data import fetch
from skimage.measure import label, regionprops
from skimage.transform import resize
from tensorflow.keras import backend as K
from tensorflow.keras.models import load_model

# Define Classes


class Tkv:
    def __init__(self, path):
        self.path = path
        self.directory, self.base, self.extension = self._split_path(self.path)
        self._img = nib.Nifti1Image
        self.data = np.array
        self.mask = np.array
        self._mask_img = nib.Nifti1Image
        self.affine = np.array
        self.shape = tuple
        self.zoom = tuple
        self.orientation = None
        self.tkv = np.nan
        self.lkv = np.nan
        self.rkv = np.nan
        self._load_data()

    @staticmethod
    def _split_path(path):
        directory = os.path.dirname(path)
        base = os.path.splitext(os.path.basename(path))[0]
        extension = os.path.splitext(os.path.basename(path))[1]
        if extension == '.gz' and base[-4:] == '.nii':
            extension = '.nii.gz'
            base = base[:-4]
        return directory, base, extension

    def _load_data(self):
        if self.extension == '.PAR':
            self._img = nib.load(self.path, scaling='fp')
        else:
            self._img = nib.load(self.path)
        self.data = self._img.get_fdata()
        self.affine = self._img.affine
        self.shape = self._img.shape
        self.zoom = self._img.header.get_zooms()
        self.orientation = nib.orientations.aff2axcodes(self.affine)

    def get_mask(self, weights_path=None, post_process=True, inplace=False):
        if weights_path is None:
            weights_path = fetch.Weights().path
        img = conform(self._img, out_shape=(240, 240, self.shape[-1]),
                      voxel_size=(1.458, 1.458, self.zoom[-1] * 0.998),
                      orientation='LIP')
        data = img.get_fdata()
        data = np.flip(data, 1)
        data = np.swapaxes(data, 0, 2)
        data = np.swapaxes(data, 1, 2)
        data = self._rescale(data)
        data = resize(data, (data.shape[0], 256, 256))
        data = data.reshape((data.shape[0], data.shape[1], data.shape[2], 1))
        model = load_model(weights_path,
                           custom_objects={'dice_coef_loss':
                                           self._dice_coef_loss,
                                           'dice_coef': self._dice_coef})
        batch_size = 2 ** 3
        mask = model.predict(data, batch_size=batch_size)
        mask = np.squeeze(mask)
        mask = np.swapaxes(mask, 0, 2)
        mask = np.swapaxes(mask, 0, 1)
        mask = np.flip(mask, 1)
        mask = resize(mask, (240, 240, self.shape[-1]))
        if post_process:
            cleaned_mask = self._cleanup(mask > 0.05)
            mask[cleaned_mask < 0.5] = 0.0
        mask_img = nib.Nifti1Image(mask, img.affine)
        self._mask_img = conform(mask_img,
                                 out_shape=self.shape,
                                 voxel_size=self.zoom,
                                 orientation=self.orientation)
        self.mask = self._rescale(self._mask_img.get_fdata(), 0, 1)
        self._mask_img = nib.Nifti1Image(self.mask, self._mask_img.affine)
        self.tkv = (np.sum(self.mask > 0.5) *
                    np.prod(self.zoom))/1000
        self.lkv = (np.sum(self.mask[120:] > 0.5) *
                    np.prod(self.zoom))/1000
        self.rkv = (np.sum(self.mask[:120] > 0.5) *
                    np.prod(self.zoom)) / 1000

        if not inplace:
            return self.mask

    def mask_to_nifti(self, path=None):
        if path is None:
            path = os.path.join(self.directory, self.base + '_mask.nii.gz')

        # Generate the mask if that hasn't already been done
        if type(self._mask_img) is type:
            self.get_mask(inplace=True)

        nib.save(self._mask_img, path)

    def data_to_nifti(self, path=None):
        if path is None:
            path = os.path.join(self.path, self.base + '.nii.gz')
        nib.save(self._img, path)

    @staticmethod
    def _rescale(data, black=None, white=None):
        if black is None:
            black = np.mean(data) - 0.5 * np.std(data)
            if black < data.min():
                black = data.min()
        if white is None:
            white = np.mean(data) + 4 * np.std(data)
            if white > data.max():
                white = data.max()
        data = np.clip(data, black, white) - black
        data = data / (white - black)
        return data

    @staticmethod
    def _cleanup(mask):
        clean_mask = np.zeros(mask.shape, dtype=np.uint8)
        label_mask = label(mask > 0.5, connectivity=1)
        props = regionprops(label_mask)
        areas = [region.area for region in props]

        # This means there have to be two kidneys in the scan...
        kidney_labels = np.argpartition(areas, -2)[-2:]

        clean_mask[label_mask == props[kidney_labels[0]].label] = 1
        clean_mask[label_mask == props[kidney_labels[1]].label] = 1

        return clean_mask

    @staticmethod
    def _dice_coef(y_true, y_pred):
        smooth = 1.0
        y_true_f = K.flatten(y_true)
        y_pred_f = K.flatten(y_pred)
        intersection = K.sum(y_true_f * y_pred_f)
        return (2. * intersection + smooth) / (
                    K.sum(y_true_f) + K.sum(y_pred_f) + smooth)

    def _dice_coef_loss(self, y_true, y_pred):
        loss = 1 - self._dice_coef(y_true, y_pred)
        return loss
