# Renal Segmentor
[![Build Status](https://travis-ci.com/alexdaniel654/Renal_Segmentor.svg?token=fiWxYk2SzMsVfjbp9BPV&branch=master)](https://travis-ci.com/alexdaniel654/Renal_Segmentor)
[![codecov](https://codecov.io/gh/alexdaniel654/Renal_Segmentor/branch/master/graph/badge.svg?token=6oSiDfrFpJ)](https://codecov.io/gh/alexdaniel654/Renal_Segmentor)

An application to segment kidneys from renal MRI data using a convolutional neural network (CNN).

## Using the segmentor from an executable

1. Double click `renal_segmentor.exe`. The GUI still needs some work so takes quite a long time to load (~30 sec) and doesn't have a splash screen so be patient.
2. Once the GUI has loaded, click `Browse` and select the raw data you want to segment. Supported file types are `.PAR/.REC`, `.nii`, `.nii.gz` and `.img/.hdr`, other file types supported by [nibable](https://nipy.org/nibabel/api.html#file-formats) may work but are untested.
3. If you want the mask to be just 0s and 1s tick the `binary` check box, if you want the CNNs probability that the voxel is a kidney, leave it unchecked.
4. Tick the `raw` checkbox if you want the raw data to be saved as a `.nii.gz` file in the same location as the mask.
5. By default the segmentor outputs the mask as a `.nii.gz` file in the same folder as the raw data e.g. if the raw checkbox was ticked, after running the programme the folder with raw data `sub_01.PAR` would also have `sub_01.nii.gz` and `sub_01_mask.nii.gz` in it. If you want the mask to be output somewhere different click `Browse` and navigate to the folder you want to save the data in then give the mask a name.
6. Click start.
7. The application will run and hopefully a few seconds later a box will appear saying the program completed successfully. 
8. If you want to segment some different data click the `edit` button on the bottom of the finished screen, if you're done, click `close`.