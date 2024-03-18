import tifffile
import BounTI
import skimage
import numpy as np

volume = BounTI.volume_import(r"{YOUR LOCATION!!!}\Lizard-16bit.tif")
volume_label, seed = BounTI.segmentation(volume,37000,12000,100,20,seed_dilation=False)
labeled_volume = skimage.util.img_as_uint(volume_label.astype(np.uint16))
tifffile.imwrite(r"{YOUR LOCATION!!!}\Lizard-16bit-Segmented.tif", labeled_volume)
print(volume_label)



