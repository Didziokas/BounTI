# BounTI for Dragonfly

Dragonfly plugin for **BounTI** — *Boundary-preserving Threshold Iteration*,
a segmentation tool for micro-CT data (Didziokas et al. 2024, *Journal of
Anatomy* 245:829–841;
[paper](https://doi.org/10.1111/joa.14063).

The tool starts from the anatomical components that are well separated at a
very high grey value (the **Initial Threshold**) and iteratively grows them
into the lower-value bone voxels — stepping the threshold down in `NI+1`
discrete steps to the **Target Threshold** — while preventing each component
from spreading across bridges of shared grey level into neighbouring bones.

This plugin is a fast, parallelized implementation of the same algorithm,
aligned with the standalone `BounTI.py` in the original repository, packaged
as a per-user Dragonfly addon. Built and tested on **Dragonfly 2025.1
(Build 2053)**, Windows 11 ×64. It only uses what Dragonfly already ships
(Python 3.10, numpy, scipy, scikit-image) — no extra packages, no admin
rights, and nothing under `C:\Program Files\Dragonfly` is touched.

## Performance

Measured on the lizard skull test dataset (`Lizard-16bit.tif`,
1007 × 305 × 510 uint16, 156.6 million voxels) on a 14-core desktop:

| Data | Parameters (IT/TT/NS/NI) | Original algorithm | This version | Speed-up |
|---|---|---|---|---|
| Full volume | 40000 / 12000 / 200 / 100 (paper demo) | ≈ 2 h (est.) | **4 m 43 s** | ≈ 25× |
| Full volume | 34000 / 21500 / 14 / 7 | — | **24.7 s** | — |
| Full volume | 37000 / 12000 / 100 / 10 (quick preset) | 930 s (standalone `BounTI.py` reference) | **54 s** | 17× |
| Crop, 400×200×300 | NS 20 / NI 20 (direct crop benchmark) | 644 s | **8.2 s** | 78× |
| 45-slice crop | 40000 / 12000 / 200 / 100 | 59.8 s | **4.2 s** | 14× |
| 45-slice crop | 34000 / 21500 / 14 / 7 | 13.0 s | **1.0 s** | 14× |

The speed-up comes from eliminating redundant full-volume work and running
the per-level work of all segments in parallel threads. The thread count is
chosen **from free physical RAM** at run time, so it parallelizes hard while
keeping large volumes from paging.

`IT` = Initial Threshold, `TT` = Target Threshold, `NS` = Number of
Segments, `NI` = Number of Iterations. The "—" row simply means no
comparable original-algorithm measurement exists for that configuration;
the crop rows show the expected scaling.

## Installing

1. Download this folder.
2. Run the installer **while Dragonfly is closed**:
   * right-click `Install.ps1` → **Run with PowerShell**, or from a
     terminal:
     `powershell -ExecutionPolicy Bypass -File Install.ps1`
3. Restart/open Dragonfly.

The installer copies the `BounTI` folder into the per-user plugins folder
of **every Dragonfly data folder it finds on the account** — under
`%LOCALAPPDATA%\Comet\Dragonfly<version>\`,
`%LOCALAPPDATA%\ORS\Dragonfly<version>\` or `%LOCALAPPDATA%\Dragonfly<version>\`
(the location differs between Dragonfly releases) — so
`...\Plugins\BounTI\__init__.py` exists. Nothing under
`C:\Program Files\Dragonfly` is touched.

#### Manual install (without the installer)

1. Open File Explorer, paste `%LOCALAPPDATA%` into the address bar, and
   find your Dragonfly data folder — it is named `Dragonfly<version>`
   directly there, or inside `Comet\` or `ORS\` (e.g.
   `Comet\Dragonfly2025.1`). If you cannot find it, open Dragonfly once;
   opening it will have created the folder.
2. Open that folder's `pythonUserExtensions\Plugins\` subfolder
   (create the `Plugins` folder if it does not exist).
3. Disable/unblock if needed: if you downloaded the files as a ZIP,
   right-click the ZIP > Properties > tick "Unblock" before extracting.
4. Copy the **`BounTI` folder** (the whole folder with the four `.py`
   files) into `Plugins\` — so that
   `...\pythonUserExtensions\Plugins\BounTI\__init__.py` exists.
5. Restart Dragonfly. The plugin loads at startup.

To remove it manually, delete that `BounTI` folder and restart Dragonfly.

**Uninstall:** `powershell -ExecutionPolicy Bypass -File Install.ps1 -Uninstall`
— or just delete the `BounTI` folder from the Plugins directory.

## Using it

Load an image, then open the BounTI panel:

* **Utilities > Plugins > Run BounTI**, or
* right-click a channel in the scene and pick **Segment with BounTI...**

The dockable panel appears in the *Segmentation Tools* tab. Fields:

* **Channel** — the image to segment. 16-bit unsigned data is expected
  (Dragonfly import produces this); 8-bit data is widened to 16-bit with
  values unchanged (they already sit in 0–255), and float or out-of-range
  data is converted — linearly remapped to the 0–65535 range — with a
  warning, per the BounTI manual's conversion rules.
* **Seed MultiROI** — *optional.* An existing MultiROI whose labels are
  used as the starting points instead of the automatically selected seed
  components.
* **Initial Threshold** — high grey value that isolates your target;
  place it at, or just right of, the start of the bone peak in the
  histogram. If bones merge together, raise it; if bones vanish, lower it.
* **Target Threshold** — low grey value giving the desired bone
  definition; usually just right of the soft-tissue peak.
* **Number of Segments** — set slightly above the number of anatomical
  components you expect.
* **Number of Iterations** — the discrete steps from Initial to Target
  Threshold. Start around 20; raise to 100–200 if adjacent segments
  visually need finer boundaries (each iteration is a full threshold
  step, and with this implementation each is fast — a 100-level run on a
  full skull scan is minutes, so iterating generously is now cheap).
* **Seed Dilation** — dilates every seed by one voxel before the sweep.
  Default **Off**; the BounTI manual notes it is "usually not required".
* **Label Preservation** — only enabled with a Seed MultiROI: keep its
  label values as-is instead of re-selecting the largest components.
* **Save Seed** — also publish the seed as a MultiROI named
  `<channel>_BounTI_Seed`.

Press **Apply** (a volume over 1 GB asks for confirmation first). A
progress dialog appears with a working **Cancel** — cancelling stops at
the end of the level in progress and publishes the partial result. The
output is a MultiROI named **`<channel>_BounTI`** with one label per
segment bound to the channel in real space and rendered in all open
views immediately; each segment is individually editable, hideable and
exportable like any Dragonfly MultiROI. (Label names follow the label
values — with the automatic seed that is "Segment 1", "Segment 2", …
1..N; with Label Preservation the names follow the MultiROI's own label
values.)

### How the algorithm works here

1. **Seed.** From the voxels above the *Initial Threshold*, take the *N*
   largest 6-connected components — if fewer exist, the warning
   "Number of segments should be reduced to {k}" is printed in Dragonfly's
   Python console and the run proceeds with all *k*. (With Seed Dilation
   each seed is grown by one voxel first.) With a Seed MultiROI the
   starting points come from it: its non-zero voxels form one region and
   the *N* largest 6-connected components of it are re-selected and
   numbered 1..N — or, with Label Preservation, each non-zero label
   region of the MultiROI is used as-is, keeping its label values.
2. **Sweep.** Compute the current-iteration threshold
   `CIT(i) = IT − i·(IT − TT)/NI` for `i = 0..NI` and work down the
   levels. Before each level, the not-yet-labeled voxels above `CIT(i)`
   are gathered once, and everything is limited to the bounding box of
   the active region.
3. **Grow.** For each segment, inside that segment's own bounding box
   extended by a 10 % margin (the standalone algorithm's windowing), the
   segment claims the connected run of newly-opened voxels that touches
   its current footprint. Segments are processed in label order, so on
   voxels two segments both reach, the higher-numbered segment wins —
   matching the standalone version. Once assigned, a label never moves:
   a voxel labeled in an earlier level is never re-taken or zeroed.
4. **Publish.** The labeled uint16 array is written into the Dragonfly
   object model as a MultiROI whose labels are renamed to
   "Segment #", and it is published live in the open scenes.

### Differences vs the original Avizo *BounTI Flood*

This implementation follows the reference `BounTI.py` logic instead of the
original addon's wavefront-flood internals. On well-separated data the
results coincide; the intended differences (all fixes approved by the
tool's owner) are:

* **Growth is windowed per segment** (its bounding box + 10 % margin)
  instead of a global wavefront: a segment only spreads into regions it
  actually touches, so labels no longer creep through every shared
  boundary in runaway fashion — and stray debris fragments far from the
  seed no longer merge into segments.
* **A label is never erased once assigned.** The original's transient
  zeroing during sub-passes could permanently delete pieces of segments
  in dispute zones; here those voxels simply stay labeled by whichever
  segment claimed them first.
* **Equal-size seeds no longer corrupt labels.** The original could pick
  the same component twice when two components tied in size, producing
  stray duplicate label values; this version always selects *N distinct*
  components with deterministic tie-breaking.
* **Seed Dilation is a true 1-voxel dilation** (`ball(1)`, the 6-connected
  cross). The original used `ball(2)` — a radius-2 ball up to 5 voxels
  across — although the paper and manual describe a 1-voxel dilation.
* **Seed Dilation defaults to Off** (the manual's recommendation); the
  original defaulted it to On.
* **Clear validation errors** instead of bare asserts: Initial must be
  greater than Target Threshold, counts must be at least 1, etc.
* **Number of Segments accepts up to 1000** in the Dragonfly panel
  (the Avizo addon clamps at 100), so large designs like the paper's
  NS = 200 lizard dataset are possible.

## Files

| File | What it is |
|---|---|
| `BounTI/__init__.py` | Plugin entry point (auto-imported by Dragonfly at startup) |
| `BounTI/BounTI.py` | Plugin class: menu registration, Apply, progress + Cancel, data conversion, MultiROI publishing |
| `BounTI/mainform_bounti.py` | The Qt dock panel (fields, tooltips mirroring the BounTI manual) |
| `BounTI/bounti_core.py` | The segmentation core (numpy/scipy only) — the same code embedded verbatim in the Avizo "BounTI Flood Fast" version |
| `Install.ps1` | One-command install / uninstall for every Dragonfly version on this user account |

## Troubleshooting

* **Panel doesn't show up** — plugins are only picked up at startup:
  install with Dragonfly closed and restart it. The panel is under the
  *Segmentation Tools* tab, header "BounTI".
* **"Initial Threshold (…) must be greater than Target Threshold (…)"** —
  validation, not a bug: swap the two values.
* **"Number of segments should be reduced to k" warning** (in Dragonfly's
  Python console/log) — there are fewer distinct components above the
  Initial Threshold than requested segments; the run uses all *k* of
  them. Raise the Initial Threshold to separate more components, or set
  Number of Segments near the real count.
* **Small isolated fragments stay unclaimed** where the old flood merged
  them into a segment arbitrarily — intended behaviour of the windowed
  growth (see the differences section above). Raising Number of
  Iterations refines boundaries between adjacent real segments.
* **Very slow run** — 100+ iterations over a multi-GB volume is simply the
  heaviest configuration; start with NI ≈ 20, judge the result, and only
  then iterate finer. Cancel from the progress dialog anytime — you keep
  the partial result.

## Compatibility

Tested on **Dragonfly 2025.1 Build 2053** (Windows 11 x64, per-user
install) with the lizard skull sample volume
(`Lizard-16bit.tif`). Requires no extra packages: it only uses what
Dragonfly already ships (numpy, scipy, scikit-image).

## Acknowledgements

Development of this plugin was assisted by AI (the **lab-deep** model
running in the **ZCode** harness), as described in full in
[AU - Chatbot interface: guides and templates](https://nat.au.dk/ailab/chatbot-interface-guides-and-templates).

## Citing

If this tool is used toward published research, please cite:

> Didziokas et al. *BounTI: boundary-preserving threshold iteration — a
> user-friendly tool for micro-CT segmentation*. **Journal of Anatomy**
> 245:829–841 (2024). https://doi.org/10.1111/joa.14063
