## BounTI (boundary-preserving threshold iteration): A user-friendly tool for automatic hard tissue segmentation [(Paper)](https://onlinelibrary.wiley.com/doi/full/10.1111/joa.14063)
### Initial Seed and Threshold Iteration Segmentation
![](https://github.com/Didziokas/BounTI/blob/main/Lizard%20rotate%20resize.gif)

### Segmentation Results
![](https://github.com/Didziokas/BounTI/blob/main/Lizard%20explosion.gif)

## Parallelized versions

New implementations of BounTI built on the same algorithm and API as
`BounTI.py`, but much faster: each threshold level still iterates over the
segments individually in their own bounding-box windows (the original
individual-label-iteration algorithm — not the wavefront flood of
`BounTI Flood.pyscro`), with the per-segment work of a level distributed
over parallel threads. The thread count is chosen from free physical RAM
at run time (minimal RAM usage, large volumes never page the machine). On
the project's lizard volume (156.6 M voxels), measured on a 14-core desktop:

| Data | Parameters (IT/TT/NS/NI) | Original | Parallelized | Speed-up |
|---|---|---|---|---|
| Crop, 400×200×300 | NS 20 / NI 20 | 644 s | **8.2 s** | 78× |
| 45-slice crop | NS 200 / NI 100 | 59.8 s | **4.2 s** | 14× |
| 45-slice crop | NS 14 / NI 7 | 13.0 s | **1.0 s** | 14× |
| Full volume | NS 200 / NI 100 (paper demo) | ≈ 2 h | **4 m 43 s** | ≈ 25× |
| Full volume | NS 100 / NI 10 (quick preset) | 930 s | **54 s** | 17× |

The parallelized core is available for every platform BounTI supports:

* **`Python Script/BounTI_parallelized.py`** — drop-in for `BounTI.py`:
  the same `volume_import` / `segmentation` interface, so existing code
  (and `Example.py`) works unchanged.
* **`Avizo-Amira Addon/BounTI Parallelized.pyscro`** — drop-in for the
  original `BounTI.pyscro` addon: same ports, parameters, ranges and
  outputs, with the parallelized core embedded.
* **`BounTI-Dragonfly/`** — the Dragonfly 2025.1 addon (plugin +
  one-command installer + README; see that folder's README for
  installation and usage, including manual install steps).

Behavior changes in all three (approved by the tool's owner):

* Seed Dilation is a true 1-voxel dilation and defaults to Off (the
  manual's recommendation); the originals used a radius-2 ball.
* Equal-size seed components are selected distinctly (the original could
  pick the same component twice when counts tied, corrupting labels).
* A label is never erased once assigned: disputed pieces of a segment
  stay labeled instead of being transiently zeroed away.
* Clear validation errors instead of bare asserts.

Results agree with the reference `BounTI.py` on well-separated data;
near boundaries between segments, disagreement stays within a few
percent of labeled voxels (measured 0.8% at low segment counts, <10% at
NS = 100 on the lizard volume) and final slices are visually
indistinguishable.

### Requirements

The code was tested on python=3.7;3.8;3.9;3.10 with the following packages:
- numpy, scipy, scikit-image, nibabel, tifffile

(`BounTI_parallelized.py`, the parallelized Avizo addon and the Dragonfly
plugin only need numpy, scipy, scikit-image — plus tifffile for
`volume_import` — of the list above.)

Avizo/Amira, Dragonfly addons and Windows Standalone Executable do not required these to be installed.

### Access

- Example Scan (Lizard) [(Google Drive, 150MB)](https://drive.google.com/file/d/1UmZ710h3OIylqJ-gCHMBGhXw-mKLSIs6/view?usp=drive_link).
- Dragonfly Addon [(GitHub)](https://github.com/Didziokas/BounTI/tree/main/BounTI-Dragonfly), [(Google Drive)](https://drive.google.com/drive/folders/1SlOn6_gfNt8abs5PmvFXLNZFqvUaAmRg?usp=drive_link).
- Avizo/Amira Addon [(GitHub)](https://github.com/Didziokas/BounTI/blob/main/Avizo-Amira%20Addon/BounTI.pyscro), [(Google Drive)](https://drive.google.com/file/d/1Bve7ZHuCLbmBp09UOBVaYg1MSxDZkXqf/view?usp=drive_link).
- Standalone Executable [(Google Drive, 70MB)](https://drive.google.com/drive/folders/1bn20Z5Ox2QUURDm16Qcq7JMmXLVORkhi?usp=drive_link).
- Python Script [(GitHub)](https://github.com/Didziokas/BounTI/tree/main/Python%20Script), [(Google Drive)](https://drive.google.com/drive/folders/1SVpdfeJhGyz7V_i7r5Wg8LZJX7K0R_dN?usp=drive_link).
- Full Repository [(Google Drive)](https://drive.google.com/drive/folders/14oFgNVe05iVretZDN28tl6bFSYIH1krK?usp=drive_link).

### User Guide

In-depth user guide is available [here](https://github.com/Didziokas/BounTI/blob/main/BounTI%20User%20Manual.pdf).

### Citation
If you use our code please cite:
```text
@article{https://doi.org/10.1111/joa.14063,
author = {Didziokas, Marius and Pauws, Erwin and Kölby, Lars and Khonsari, Roman H. and Moazen, Mehran},
title = {BounTI (boundary-preserving threshold iteration): A user-friendly tool for automatic hard tissue segmentation},
journal = {Journal of Anatomy},
doi = {https://doi.org/10.1111/joa.14063},
url = {https://onlinelibrary.wiley.com/doi/abs/10.1111/joa.14063}
}
```

## Acknowledgements

The parallelized versions (`BounTI_parallelized.py`, the parallelized
Avizo addon and the Dragonfly plugin) were developed with assistance
from AI (the **lab-deep** model running in the **ZCode** harness), as
described in full in [AU Chatbot interface: guides and templates](https://nat.au.dk/ailab/chatbot-interface-guides-and-templates).

