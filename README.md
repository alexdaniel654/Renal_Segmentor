# Renal Segmentor
[![Python CI](https://github.com/alexdaniel654/Renal_Segmentor/actions/workflows/python_ci.yml/badge.svg?branch=master)](https://github.com/alexdaniel654/Renal_Segmentor/actions/workflows/python_ci.yml)
[![codecov](https://codecov.io/gh/alexdaniel654/Renal_Segmentor/branch/master/graph/badge.svg?token=6oSiDfrFpJ)](https://codecov.io/gh/alexdaniel654/Renal_Segmentor)
[![License: GPL v3](https://img.shields.io/badge/License-GPLv3-blue.svg)](https://www.gnu.org/licenses/gpl-3.0)
[![DOI](https://zenodo.org/badge/236753300.svg)](https://zenodo.org/badge/latestdoi/236753300)

An application and Python package to segment kidneys from renal MRI data using a convolutional neural network (CNN).

<h2 align="center"><img src="https://raw.githubusercontent.com/alexdaniel654/Renal_Segmentor/master/images/banner.png" height="128"></h2>

## Using the segmentor

The easiest way to make use of the segmentor is to download the windows executable, this allows you to mask data with a stand-alone application. The executable can either be run as a GUI or a command line application allowing it to be integrated into bash scripts.

Alternatively, the methods used by the segmentor are available as a Python package and can be integrated into existing Python pipelines. Instructions for making use of the segmentor via each method are given below.

### As a Graphical User Interface (GUI)

1. Download the [latest release](https://github.com/alexdaniel654/Renal_Segmentor/releases/latest/download/renal_segmentor.exe)
2. Double click `renal_segmentor.exe`. The GUI takes quite a long time to load (~30 sec) and doesn't have a splash screen so be patient.
3. Once the GUI has loaded, click `Browse` and select all the raw data you want to segment. You can select multiple files at once. Supported file types are `.PAR/.REC`, `.nii`, `.nii.gz` and `.img/.hdr`, other file types supported by [nibable](https://nipy.org/nibabel/api.html#file-formats) may work but are untested.
4. If you want the mask to be just 0s and 1s tick the `binary` check box, if you want the CNNs probability that the voxel is a kidney, leave it unchecked.
5. Post-processing can be applied, this discards all but the two largest connected regions in the mask in theory retaining only the two kidneys and removing any erroneous unconnected regions. It should be noted that because post-processing keeps only two regions, care should be taken if using this application with transplant patients as they may have three kidneys.
6. Tick the `raw` checkbox if you want the raw image data to be saved as a `.nii.gz` file in the same location as the mask (can be useful if you're using `.PAR/REC` as your input).
7. If you would like a `.csv` file containing the Total Kidney Volume (TKV), Left Kidney Volume (LKV) and Right Kidney Volume (RKV) for each image that was segmented tick the `Export Kidney Volumes` box.
8. The masks are output from the segmentor as `.nii.gz` with `_mask` added to the original file name e.g. the mask of `sub_01.PAR` is `sub_01_mask.nii.gz`. By default, the mask is output to the same place as the raw data, if you would like the masks to be output to a different directory click `Browse` under `Output Directory` and select the folder you wish the masks to go to.
9. Click start.
10. The application will run and a few seconds later a box will appear saying the program completed successfully. The first time you run a segmentation will take a little longer as the latest algorithm weights are downloaded from the internet at this point.
11. If you want to segment some more data click the `edit` button on the bottom of the finished screen, if you're done, click `close`.

<h2 align="center"><img src="https://raw.githubusercontent.com/alexdaniel654/Renal_Segmentor/master/images/screenshot.png" height="512"></h2>

### Via a Command Line Interface (CLI)
1. Download the [latest release](https://github.com/alexdaniel654/Renal_Segmentor/releases/latest/download/renal_segmentor.exe)
2. Run the `renal_segmentor.exe -h` to generate a list of available parameters. The application runs via a command line if any input arguments are specified, if not, it opens as a GUI.

### As a Python package
1. Activate the python environment you want to install the package on and run `pip install renalsegmentor`.
2. The example code snippet below will generate a mask of `T2w.nii.gz` as a numpy array and print the TKV to the terminal.

```python
from segment import Tkv
segmentation = Tkv('T2w.nii.gz')
mask = segmentation.get_mask()
print(f'Total Kidney Volume = {segmentation.tkv:.2f} ml')
```

## Citing Renal Segmentor
If you have made use of renal segmentor for your work, please cite Daniel AJ, _et al_. Automated renal segmentation in healthy and chronic kidney disease subjects using a convolutional neural network. Magnetic Resonance in Medicine 2021;86:1125–1136 doi: [https://doi.org/10.1002/mrm.28768](https://doi.org/10.1002/mrm.28768). Alternatively if you wish to cite a specific software version, each release has an individual DOI on Zenodo, the DOI for the latest release can be [found here](https://doi.org/10.5281/zenodo.4068850).

## How it works

The methods used in this software are outlined in Daniel AJ, _et al_. Automated renal segmentation in healthy and chronic kidney disease subjects using a convolutional neural network. Magnetic Resonance in Medicine 2021;86:1125–1136 doi: [https://doi.org/10.1002/mrm.28768](https://doi.org/10.1002/mrm.28768).

The dataset used to train this network is freely available on Zenodo, doi: [https://doi.org/10.5281/zenodo.5153567](https://doi.org/10.5281/zenodo.5153567).

## Contributing

Feel free to open a pull request if you have a feature you want to develop or [drop me an email](mailto:alexander.daniel@nottingham.ac.uk) to discuss things further.
