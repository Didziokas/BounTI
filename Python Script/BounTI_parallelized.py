"""BounTI parallelized - fast implementation of the BounTI algorithm
(boundary-preserving threshold iteration; Didziokas et al. 2024,
Journal of Anatomy 245:829-841, https://doi.org/10.1111/joa.14063).

The publicly visible API is the same as the reference ``BounTI.py``:

    label_array, seed = segmentation(volume, initial_threshold, target_threshold,
                                     segments, iterations, **kwargs)
    volume = volume_import(path)

with the additions that come with the new core:

  * the per-level work of all segments runs in parallel threads - the
    thread count is chosen from free physical RAM at run time, so big
    volumes speed up hard but never page the machine;
  * measured on the project's lizard volume (156.6 M voxels): the full
    demo preset (IT 40000 / TT 12000 / NS 200 / NI 100) drops from
    about 2 h to under 5 min, and the quick preset
    (IT 37000 / TT 12000 / NS 100 / NI 10) from 930 s to 54 s;
  * behavior fixes, all approved by the tool's owner: Seed Dilation
    defaults to Off and is a true 1-voxel dilation (`ball(1)`), equal-size
    seed components can no longer corrupt label values, a label is never
    erased once assigned, input errors raise clear messages;
  * everything else follows the reference script: the same threshold
    sweep, the same per-segment windowed growth, the same claim rules.

The implementation below is byte-identical to the ``bounti_core.py``
shipped with the Dragonfly/Avizo additions.

Use it exactly like ``BounTI.py`` - see ``Example.py`` next to the
reference script.
"""

import ctypes
import os
import warnings
from concurrent.futures import ThreadPoolExecutor

import numpy as np
from scipy import ndimage as ndi
from skimage.morphology import ball


class _MEMORYSTATUSEX(ctypes.Structure):
    _fields_ = [
        ("dwLength", ctypes.c_ulong),
        ("dwMemoryLoad", ctypes.c_ulong),
        ("ullTotalPhys", ctypes.c_ulonglong),
        ("ullAvailPhys", ctypes.c_ulonglong),
        ("ullTotalPageFile", ctypes.c_ulonglong),
        ("ullAvailPageFile", ctypes.c_ulonglong),
        ("ullTotalVirtual", ctypes.c_ulonglong),
        ("ullAvailVirtual", ctypes.c_ulonglong),
        ("ullAvailExtendedVirtual", ctypes.c_ulonglong),
    ]


def _free_ram_bytes():
    """Free physical RAM, or None when unavailable (non-Windows)."""
    try:
        stat = _MEMORYSTATUSEX()
        stat.dwLength = ctypes.sizeof(_MEMORYSTATUSEX)
        if ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(stat)):
            return int(stat.ullAvailPhys)
    except Exception:
        pass
    return None


_MAX_WORKERS_RAM_BUDGET = 2 * 1024 ** 3          # bytes available for windows
_MAX_WORKERS_HARD_CAP = 16


def _level_window(extent, shape):
    """BounTI.py's window: a segment's bbox expanded by 10% of each extent,
    clipped to the trim/volume shape.

    ``extent`` is half-open (start, stop) per axis as ndi.find_objects
    reports it; BounTI.py's bbox2_3D yields INCLUSIVE end indices, so the
    inclusive end is stop-1, the margin uses int((stop-1-start) * 0.1), and
    the window slice is [start-m : (stop-1)+m) exactly as BounTI.py writes
    its region slices. ``extent`` may be None (label not found by
    find_objects): then the full trim box is used, mirroring the original's
    fallback when bbox2_3D throws on a missing label."""
    if extent is None:
        return (0, shape[0]), (0, shape[1]), (0, shape[2])
    (z0, z1), (y0, y1), (x0, x1) = extent
    mz = int((z1 - 1 - z0) * 0.1)
    my = int((y1 - 1 - y0) * 0.1)
    mx = int((x1 - 1 - x0) * 0.1)
    z0, z1 = max(z0 - mz, 0), min(z1 - 1 + mz, shape[0])
    y0, y1 = max(y0 - my, 0), min(y1 - 1 + my, shape[1])
    x0, x1 = max(x0 - mx, 0), min(x1 - 1 + mx, shape[2])
    return (z0, z1), (y0, y1), (x0, x1)


def _window_claims(labeled, open_mask, j, extent):
    """One segment's same-level claims (BounTI.py's per-j body, without the
    transient zeroing -- see the module docstring for why).

    Returns (j, trim_flat_indices) -- the claim voxels as flat int32 indices
    into the trim-sized ``labeled`` array -- or (j, None) when the window
    holds no open voxels (nothing to claim). BounTI.py's empty-segment
    fallback (index = 1 over a full-volume window) never triggers here:
    segment labels can never drop to zero voxels because claims only ever
    write to voxels that were unlabeled at the start of the level (no
    zeroing is performed, see the module docstring).
    """
    shape = labeled.shape
    zz, yy, xx = _level_window(extent, shape)
    sl = (slice(zz[0], zz[1]), slice(yy[0], yy[1]), slice(xx[0], xx[1]))
    o_w = open_mask[sl]
    seg_w = labeled[sl] == j                       # bool, fresh allocation
    if not o_w.any() or not seg_w.any():
        return j, None
    pos = int(np.argmax(seg_w))                    # first j-pixel, C-order in window —
                                                   # MUST be taken before the merge below:
                                                   # `temp = seg_w; temp |= o_w` mutates the
                                                   # same bool array, and the representative
                                                   # pixel has to come from the segment itself
                                                   # (BounTI.py's np.where(reduced == j) before
                                                   # merging), not from the merged graph.
    temp = seg_w                                   # merged graph = seg ∪ open
    temp |= o_w
    relab_labels = ndi.label(temp)[0]              # 6-connected, as in BounTI.py
    try:
        index = int(relab_labels.flat[pos])
    except Exception:
        index = 1                                  # the original's fallback
    try:
        claim = relab_labels == index
        claim &= o_w
    except Exception:
        print(f"missing {j}")                      # the original's message shape
        return j, None
    local = np.flatnonzero(claim.ravel()).astype(np.int64)
    if local.size == 0:
        return j, None
    ny_w, nx_w = yy[1] - yy[0], xx[1] - xx[0]      # window dims (local flat is
    z = local // (ny_w * nx_w)                     # over the window subarray)
    y = (local - z * ny_w * nx_w) // nx_w
    x = local - z * ny_w * nx_w - y * nx_w
    NYX = shape[1] * shape[2]
    trim_flat = ((z + zz[0]) * NYX + (y + yy[0]) * shape[2] + (x + xx[0]))
    return j, trim_flat.astype(np.int32)


def _bbox3d(mask):
    """Half-open bounding box ((z0, z1), (y0, y1), (x0, x1)) of a boolean
    array via np.any reductions only (no np.argwhere). None when empty."""
    zs = np.any(mask, axis=(1, 2))
    if not zs.any():
        return None
    ys = np.any(mask, axis=(0, 2))
    xs = np.any(mask, axis=(0, 1))
    z0 = int(np.argmax(zs))
    z1 = int(zs.size - int(np.argmax(zs[::-1])))
    y0 = int(np.argmax(ys))
    y1 = int(ys.size - int(np.argmax(ys[::-1])))
    x0 = int(np.argmax(xs))
    x1 = int(xs.size - int(np.argmax(xs[::-1])))
    return ((z0, z1), (y0, y1), (x0, x1))


def _union_bbox(a, b):
    if a is None:
        return b
    if b is None:
        return a
    return ((min(a[0][0], b[0][0]), max(a[0][1], b[0][1])),
            (min(a[1][0], b[1][0]), max(a[1][1], b[1][1])),
            (min(a[2][0], b[2][0]), max(a[2][1], b[2][1])))


def _get_largest(seed_mask_region, segments):
    """Relabel the NS largest distinct 6-connected components to 1..k.

    Ties are broken deterministically by component index (stable sort).
    Returns (region_labels_uint16, number).
    """
    labels, n_components = ndi.label(seed_mask_region)
    counts = np.bincount(labels.ravel())
    n_components = min(n_components, counts.size - 1)
    if n_components < segments:
        warnings.warn(f"Number of segments should be reduced to {n_components}")
    number = min(segments, n_components)
    order = np.argsort(-counts[1:], kind="stable")
    keep = order[:number]
    lut = np.zeros(counts.size, dtype=np.uint16)
    lut[keep + 1] = np.arange(1, number + 1, dtype=np.uint16)
    return lut[labels], number


def _grow(seed, number, structure):
    """Per-label seed dilation (original grow() order: later labels
    overwrite). Each label is dilated only inside its bounding-box
    subvolume expanded by the structuring-element radius."""
    formed = seed.copy()
    slices = ndi.find_objects(seed)
    nz, ny, nx = seed.shape
    radius = (structure.shape[0] - 1) // 2
    for i in range(number):
        lab = i + 1
        sl = slices[lab - 1] if lab - 1 < len(slices) else None
        if sl is None:
            continue
        (sz, sy, sx) = sl
        z0, z1 = max(sz.start - radius, 0), min(sz.stop + radius, nz)
        y0, y1 = max(sy.start - radius, 0), min(sy.stop + radius, ny)
        x0, x1 = max(sx.start - radius, 0), min(sx.stop + radius, nx)
        comp = seed[z0:z1, y0:y1, x0:x1] == lab
        grown = ndi.binary_dilation(comp, structure=structure)
        formed[z0:z1, y0:y1, x0:x1][grown] = lab
    return formed


def segmentation(volume_array, initial_threshold, target_threshold, segments, iterations, label=None,
                 label_preserve=False, seed_dilation=False, progress=None):
    """Fast BounTI segmentation following BounTI.py's algorithm.

    Parameters
    ----------
    volume_array : 3D numeric array (any dtype; only comparisons are made)
    initial_threshold, target_threshold : numbers, IT > TT
    segments : int >= 1 (NS); must also be <= 65535 (uint16 labels)
    iterations : int >= 1 (NI)
    label : optional 3D seed array (uint16); nonzero voxels form the seed.
        With label_preserve=False the NS largest components of its nonzero
        mask are recomputed; with True its values are used as-is.
    label_preserve, seed_dilation : bool flags (see module docstring).
    progress : optional callback progress(fraction: float, message: str),
        called once per level. An Exception raised BY the callback cancels
        the run: in-flight level windows are abandoned and the current
        partial (labeled_volume, formed_seed) is returned. BaseExceptions
        propagate (the shipped adapters raise RuntimeError).

    Returns
    -------
    (labeled_volume, formed_seed) : two uint16 arrays shaped like volume_array.
    """
    volume = np.asarray(volume_array)
    if volume.ndim != 3:
        raise ValueError(
            f"volume_array must be a 3D array, got {volume.ndim} dimensions")
    if not initial_threshold > target_threshold:
        raise ValueError(
            f"initial threshold ({initial_threshold}) must be greater than "
            f"the target threshold ({target_threshold})")
    if iterations < 1:
        raise ValueError(f"iterations must be >= 1, got {iterations}")
    if segments < 1:
        raise ValueError(f"segments must be >= 1, got {segments}")
    if segments > 65535:
        raise ValueError("segments must be at most 65535 (labels are uint16)")

    # ---- seed formation ------------------------------------------------
    if label is not None:
        label = np.asarray(label)
        if label.shape != volume.shape:
            raise ValueError(
                f"seed label shape {label.shape} does not match the volume "
                f"shape {volume.shape}")
        if label.dtype != np.uint16:
            label = label.astype(np.uint16)
        if not label.any():
            raise ValueError("the supplied seed label contains no nonzero voxels")

    if label is None or not label_preserve:
        if label is None:
            seed_mask = volume > initial_threshold
            if not seed_mask.any():
                raise ValueError(f"no voxels above the initial threshold "
                                 f"({initial_threshold}) -- lower it")
        else:
            seed_mask = label != 0
        sz, sy, sx = _bbox3d(seed_mask)
        seed = np.zeros(volume.shape, dtype=np.uint16)
        seed_region, number = _get_largest(
            seed_mask[sz[0]:sz[1], sy[0]:sy[1], sx[0]:sx[1]], segments)
        seed[sz[0]:sz[1], sy[0]:sy[1], sx[0]:sx[1]] = seed_region
        del seed_region, seed_mask
    else:
        seed = label.copy()
        number = segments

    if seed_dilation:
        formed_seed = _grow(seed, number, ball(1))
        del seed
    else:
        formed_seed = seed

    # ---- one-time trim ---------------------------------------------------
    above_tt = volume > target_threshold
    vol_box = _bbox3d(above_tt)
    del above_tt
    trim = _union_bbox(vol_box, _bbox3d(formed_seed != 0))
    (tz0, tz1), (ty0, ty1), (tx0, tx1) = trim

    labeled = np.zeros((tz1 - tz0, ty1 - ty0, tx1 - tx0), dtype=np.uint16)
    labeled[...] = formed_seed[tz0:tz1, ty0:ty1, tx0:tx1]
    vol_trim = volume[tz0:tz1, ty0:ty1, tx0:tx1]

    open_buf = np.empty(labeled.shape, dtype=bool)
    eq_buf = np.empty(labeled.shape, dtype=bool)

    try:
        free_ram = _free_ram_bytes()
        budget = _MAX_WORKERS_RAM_BUDGET
        if free_ram is not None:
            budget = min(budget, int(free_ram * 0.6))
        cores = min(_MAX_WORKERS_HARD_CAP, os.cpu_count() or 1)

        for i in range(iterations + 1):
            if progress is not None:
                try:
                    progress(i / (iterations + 1), f"Refining -- Iter:{i}")
                except Exception:
                    break  # cancelled: keep the partial state
            cit = initial_threshold - (
                i * (initial_threshold - target_threshold) / iterations)

            np.greater(vol_trim, cit, out=open_buf)
            np.equal(labeled, 0, out=eq_buf)
            np.logical_and(open_buf, eq_buf, out=open_buf)
            if not open_buf.any():
                continue  # nothing to claim at this level
            slices = ndi.find_objects(labeled)

            # RAM budget: workers sized by the largest window this level.
            max_w = 0
            extents = []
            for j in range(number):
                sl = slices[j] if j < len(slices) else None
                ext = None if sl is None else tuple(
                    (sl[d].start, sl[d].stop) for d in range(3))
                extents.append(ext)
                if ext is not None:
                    w = _level_window(ext, labeled.shape)
                    max_w = max(max_w, (w[0][1] - w[0][0]) * (w[1][1] - w[1][0])
                                * (w[2][1] - w[2][0]))
            workers = int(budget // (8 * max_w + 64 * 1024 ** 2)) if max_w else cores
            workers = max(1, min(cores, workers))

            with ThreadPoolExecutor(max_workers=workers) as pool:
                futures = [
                    pool.submit(_window_claims, labeled, open_buf, j + 1, extents[j])
                    for j in range(number)
                ]
                claims = {}
                for fut, j in zip(futures, range(1, number + 1)):
                    claims[j] = fut.result()[1]
            # Apply in ascending segment order (the original loop order):
            # later segments overwrite lower ones' same-level claims.
            flat_view = labeled.reshape(-1)
            for j in range(1, number + 1):
                idx = claims.get(j)
                if idx is not None:
                    flat_view[idx] = np.uint16(j)
            del claims, futures
    finally:
        del open_buf, eq_buf

    # Paste the trim region back into the full-size uint16 output.
    labeled_volume = np.zeros(volume.shape, dtype=np.uint16)
    labeled_volume[tz0:tz1, ty0:ty1, tx0:tx1] = labeled
    del labeled
    return labeled_volume, formed_seed


def volume_import(volume_path, dtype=np.uint16):
    """Read a TIFF volume as a uint16 numpy array (same as BounTI.py)."""
    import tifffile
    return np.ascontiguousarray(np.asarray(tifffile.imread(volume_path), dtype=dtype))
