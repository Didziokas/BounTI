import nibabel as nib
import numpy as np
import os
from scipy import ndimage as ndi
import warnings
from skimage.morphology import ball
import skimage
import tifffile
import gc

def volume_import(volume_path, dtype = np.uint16):
    file = os.path.join(volume_path)
    volume = tifffile.imread(file)
    volume_array = np.array(volume, dtype=dtype)
    return volume_array

def get_largest(label, segments):
    labels, _ = ndi.label(label)
    assert (labels.max() != 0)
    number = 0
    try:
        bincount = np.bincount(labels.flat)[1:]
        bincount_sorted = np.sort(bincount)[::-1]
        largest = labels-labels
        m=0
        for i in range(segments):
            index = int(np.where(bincount == bincount_sorted[i])[0][m]) + 1
            ilargest = labels == index
            largest += np.where(ilargest, i + 1, 0)
        if i == segments-1:
            number = segments
    except:
        warnings.warn(f"Number of segments should be reduced to {i}")
        if number == 0:
            number = i
    return largest,number

def grow(labels, number):
    grownlabels = np.copy(labels)
    for i in range(number):
        filtered = np.where(labels==i+1,1,0)
        grown = ndi.binary_dilation(np.copy(filtered), structure=ball(2)).astype(np.uint16)
        grownlabels = np.where(np.copy(grown), i + 1, np.copy(grownlabels))
        del grown
        del filtered
    return grownlabels


def bbox2_3D(img):
    r = np.any(img, axis=(1, 2))
    c = np.any(img, axis=(0, 2))
    z = np.any(img, axis=(0, 1))

    rmin, rmax = np.where(r)[0][[0, -1]]
    cmin, cmax = np.where(c)[0][[0, -1]]
    zmin, zmax = np.where(z)[0][[0, -1]]

    return rmin, rmax, cmin, cmax, zmin, zmax

def segmentation(volume_array, initial_threshold, target_threshold, segments, iterations, label = False, label_preserve = False, seed_dilation = False):


    if label == False:
        volume_label = volume_array > initial_threshold
    else:
        volume_label = label


    if label_preserve == False:
        seed, number = get_largest(volume_label,segments)
    else:
        seed = volume_label
        number = segments


    if seed_dilation == True:
        formed_seed = grow(seed, number)
    else:
        formed_seed = seed

    labeled_volume = np.copy(formed_seed)

    for i in range(iterations+1):
        volume_label = volume_array > initial_threshold - (i * (initial_threshold - target_threshold) / iterations)
        volume_label = np.where(labeled_volume != 0, False, volume_label)
        for j in range(number):
            try:
                rmin, rmax, cmin, cmax, zmin, zmax = bbox2_3D(labeled_volume == j + 1)
            except:
                rmin, rmax, cmin, cmax, zmin, zmax = -1, 1000000, -1, 1000000, -1, 1000000
            maximum = labeled_volume.shape
            rmin = max(0, rmin - int((rmax - rmin) * 0.1))
            rmax = min(int((rmax - rmin) * 0.1) + rmax, maximum[0])
            cmin = max(0, cmin - int((cmax - cmin) * 0.1))
            cmax = min(int((cmax - cmin) * 0.1) + cmax, maximum[1])
            zmin = max(0, zmin - int((zmax - zmin) * 0.1))
            zmax = min(int((zmax - zmin) * 0.1) + zmax, maximum[2])
            temp_label = np.copy(volume_label)
            reduced_labeled_volume = labeled_volume[rmin:rmax, cmin:cmax, zmin:zmax]
            temp_label[rmin:rmax, cmin:cmax, zmin:zmax] = np.copy(volume_label)[rmin:rmax, cmin:cmax,
                                                          zmin:zmax] + (
                                                                  reduced_labeled_volume == j + 1)
            pos = np.where(reduced_labeled_volume == j + 1)
            labeled_volume[rmin:rmax, cmin:cmax, zmin:zmax] = np.where(reduced_labeled_volume == j + 1, 0,
                                                                       reduced_labeled_volume)
            labeled_temp, _ = ndi.label(np.copy(temp_label[rmin:rmax, cmin:cmax, zmin:zmax]))
            try:
                index = int(labeled_temp[pos[0][0], pos[1][0], pos[2][0]])
            except:
                index = 1
            try:
                relabelled = np.copy(labeled_temp) == index
                labeled_volume[rmin:rmax, cmin:cmax, zmin:zmax] = np.where(np.copy(relabelled), j + 1,
                                                                           labeled_volume[rmin:rmax, cmin:cmax,
                                                                           zmin:zmax])
                del temp_label
                del pos
                del labeled_temp
                del relabelled
                gc.collect()
            except:
                print(f"missing {j}")
    return labeled_volume, formed_seed


