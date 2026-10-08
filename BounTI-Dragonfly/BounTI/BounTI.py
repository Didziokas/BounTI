"""
BounTI: boundary-preserving threshold iteration, as a Dragonfly plugin.

Seeds are formed from the largest components above the initial threshold (or
from an optional seed MultiROI), then the segments flood-grow while the
working threshold sweeps from the initial threshold (IT) down to the target
threshold (TT), so boundaries already separated at IT are preserved. The
algorithm itself lives in ``bounti_core`` (framework-free numpy/scipy/skimage,
shared verbatim with the Avizo 'BounTI Flood Fast' addon).

UI parity with the Avizo addon: Initial/Target Threshold, Number of
Segments/Iterations, Seed Dilation / Label Preservation / Save Seed, and an
optional manual-seed MultiROI. Output is published as a MultiROI titled
'<channel>_BounTI' (plus '<channel>_BounTI_Seed' when Save Seed is checked).

:author: DragonBounTI project
"""

__version__ = '1.0.0'

import os

# Same multiple-OpenMP-runtime workaround the app's bundled sklearn applies
# (Python_env/Lib/site-packages/sklearn/__init__.py): the ORS libraries run on
# libomp140 while numpy/scipy/skimage load libiomp5md, and letting both
# initialize avoids a hard 'OMP: Error #15' process abort.
os.environ.setdefault('KMP_DUPLICATE_LIB_OK', 'True')

import numpy as np

import ORSModel
from ORSModel import Channel, OrsSelectedObjects, orsObj, Progress
from OrsLibraries.workingcontext import WorkingContext
from OrsHelpers.datasethelper import DatasetHelper
from ORSServiceClass.actionAndMenu.menu import Menu
from ORSServiceClass.decorators.infrastructure import action, interest, menuItem
from ORSServiceClass.messagebox.orsMessageBox import OrsTMessageBox
from ORSServiceClass.OrsPlugin.orsPlugin import OrsPlugin
from ORSServiceClass.OrsPlugin.uidescriptor import UIDescriptor

from . import bounti_core


class BounTI(OrsPlugin):

    # Plugin definition
    multiple = False
    savable = True
    keepAlive = False
    canBeGenericallyOpened = True
    showInToolbar = False

    # UIs
    # NOTE: tab='Segmentation Tools' is pinned by the feature; it is not a
    # shipped tab value (the common one is 'Dataset Tools'), so the dock/tab
    # placement is untested outside the live application.
    UIDescriptors = [UIDescriptor(name='MainForm',
                                  title='BounTI',
                                  dock='Right',
                                  tab='Segmentation Tools',
                                  modal=False,
                                  collapsible=True,
                                  movable=True,
                                  floatable=True)]

    @classmethod
    def getMainFormClass(cls):
        # The form module is mainform_bounti (the shipped mainform_* pattern):
        # on Windows' case-insensitive filesystem a form module named
        # 'bounti.py' would collide with the plugin module 'BounTI.py'.
        from .mainform_bounti import BounTIForm
        return BounTIForm

    @classmethod
    def startupDefault(cls, openMainForm=True):
        instance = super().startupDefault(openMainForm=openMainForm)
        if instance is not None:
            instance.selectedObjectsChanged()

        return instance

    @interest(OrsSelectedObjects)
    def selectedObjectsChanged(self):
        if self.getMainForm() is None:
            return
        cList = WorkingContext.getEntitiesOfClassAsObjects(self, OrsSelectedObjects, Channel.getClassNameStatic())
        if cList is not None and len(cList) == 1 and isinstance(cList[0], Channel):
            c = cList[0]
            self.getMainForm().refreshUI(c)

    @classmethod
    @menuItem('Utilities')
    def runBounTI(cls):
        aMenuItem = Menu(title='Run BounTI',
                         id_='BounTI',
                         section='090_Plugins',
                         action=cls.getActionStringForStartupDefault())
        return aMenuItem

    @classmethod
    @menuItem()
    def contextualMenu(cls, context):
        aMenuItem = None
        if context is not None:
            # Only propose if list is a 1 channel list
            evaluated = eval(context)
            if evaluated is not None and len(evaluated) == 1:
                structuredGrid = orsObj(evaluated[0])
                if structuredGrid is not None and isinstance(structuredGrid, Channel):
                    aMenuItem = Menu(title='Segment with BounTI...',
                                     id_='BounTI',
                                     section='tools',
                                     action=cls.getActionStringForStartupDefault(),
                                     enabled=True)

        return aMenuItem

    @action(title='Apply')
    def apply(self):
        """Runs the BounTI segmentation on the channel selected in the main
        form and publishes the result as a MultiROI.

        :return: the MultiROI holding the segmentation (None when validation
                 fails or the user cancels)
        :rtype: ORSModel.ors.MultiROI
        """
        mainForm = self.getMainForm()
        if mainForm is None:
            return None

        # ---- UI fields ------------------------------------------------------
        channel = mainForm.getChannel()
        initialThreshold = mainForm.getInitialThreshold()
        targetThreshold = mainForm.getTargetThreshold()
        numberOfSegments = mainForm.getNumberOfSegments()
        numberOfIterations = mainForm.getNumberOfIterations()
        seedDilation = mainForm.getSeedDilation()
        labelPreservation = mainForm.getLabelPreservation()
        saveSeed = mainForm.getSaveSeed()
        seedMultiROI = mainForm.getSeedMultiROI()

        # ---- validation (report and return, never crash) --------------------
        if channel is None:
            OrsTMessageBox.message(None, 'Select a channel to segment.', 'BounTI',
                                   OrsTMessageBox.Critical, OrsTMessageBox.Ok)
            return None
        if not initialThreshold > targetThreshold:
            OrsTMessageBox.message(None,
                                   'Initial Threshold ({}) must be greater than Target Threshold ({}).'.format(
                                       initialThreshold, targetThreshold),
                                   'BounTI', OrsTMessageBox.Critical, OrsTMessageBox.Ok)
            return None
        if numberOfSegments < 1 or numberOfIterations < 1:
            OrsTMessageBox.message(None,
                                   'Number of Segments and Number of Iterations must be at least 1.',
                                   'BounTI', OrsTMessageBox.Critical, OrsTMessageBox.Ok)
            return None

        # ---- input data -----------------------------------------------------
        # Channel data as a [z, y, x] numpy array. The array is a writable
        # view into the channel; it is only read here (and inside the core,
        # which makes no writes to the volume), never written.
        volume = self._convertToUint16(channel.getNDArray(0))

        # Very large volumes: warn and let the user bail out
        if volume.nbytes > 1024 ** 3:
            answer = OrsTMessageBox.message(
                None,
                'The volume is {:.1f} GB. BounTI on a volume this large can '
                'take a long time. Continue?'.format(volume.nbytes / (1024 ** 3)),
                'BounTI', OrsTMessageBox.Question, OrsTMessageBox.Yes | OrsTMessageBox.No)
            if answer != OrsTMessageBox.Yes:
                return None

        # Optional manual seed from a MultiROI: read its label values as a
        # [z, y, x] array (getAsNDArray is flat, z-major; reshaped here).
        seed = None
        if seedMultiROI is not None:
            arr_handle = seedMultiROI.getAsArray(0, None)
            seed = arr_handle.getAsNDArray()
            arr_handle.deleteObject()
            seed = seed.reshape(seedMultiROI.getZSize(), seedMultiROI.getYSize(), seedMultiROI.getXSize())

        # ---- segmentation ---------------------------------------------------
        IProgress = Progress()
        try:
            IProgress.startProgressWithCaption('BounTI', numberOfIterations + 1, True)
            # Draw the dialog at least once before the (slowest) seed phase.
            self._pumpQtEvents()

            def progressAdapter(fraction, message):
                # The Apply handler runs on the UI thread, so the progress
                # dialog would never repaint (it shows blank and Cancel is
                # dead) without pumping the Qt event loop here.
                self._pumpQtEvents()
                # getIsCancelled is a SINGLE call (ors.pyi: def getIsCancelled(
                # self, logging=False) -> bool); calling it twice would raise
                # TypeError on every step, which the core would read as
                # cancellation and abort immediately.
                IProgress.setExtraText(message)
                IProgress.updateProgress(int(fraction * (numberOfIterations + 1)))
                if IProgress.getIsCancelled():
                    raise RuntimeError('BounTI run cancelled by the user')

            try:
                # An exception raised by progressAdapter cancels the run and
                # the core returns its partial results (like the Avizo addon).
                labeled, formedSeed = bounti_core.segmentation(
                    volume, initialThreshold, targetThreshold,
                    numberOfSegments, numberOfIterations,
                    label=seed,
                    label_preserve=labelPreservation,
                    seed_dilation=seedDilation,
                    progress=progressAdapter)
            except ValueError as exc:
                # Core validation failures (e.g. no voxels above the initial
                # threshold, or a seed whose shape does not match the channel)
                OrsTMessageBox.message(None, str(exc), 'BounTI',
                                       OrsTMessageBox.Critical, OrsTMessageBox.Ok)
                return None
        finally:
            IProgress.closeProgress()
            IProgress.deleteObject()

        # ---- outputs --------------------------------------------------------
        multiROI = self._publishLabeledArrayAsMultiROI(labeled, channel, '_BounTI')
        if multiROI is None:
            OrsTMessageBox.message(None, 'The MultiROI could not be created from the BounTI result.',
                                   'BounTI', OrsTMessageBox.Critical, OrsTMessageBox.Ok)
            return None

        if saveSeed:
            self._publishLabeledArrayAsMultiROI(formedSeed, channel, '_BounTI_Seed')

        return multiROI

    @staticmethod
    def _pumpQtEvents():
        """Pumps pending Qt events while BounTI runs on the UI thread.

        Lets the ORS progress dialog repaint and delivers clicks on its
        Cancel button, which getIsCancelled then reports to the core.
        """
        try:
            from PyQt6.QtWidgets import QApplication
            app = QApplication.instance()
            if app is not None:
                app.processEvents()
        except Exception:
            pass

    @staticmethod
    def _convertToUint16(volume):
        """Converts channel data to uint16, per the BounTI user manual.

        uint16 data is used as-is; 8-bit data is widened with values
        unchanged; float data, and integer data with values outside 0-65535,
        are linearly remapped from [min, max] to [0, 65535] (with a warning
        message stating the applied conversion).

        :param volume: channel data as a numpy array (never modified)
        :return: a uint16 array (the input itself when already uint16)
        :rtype: numpy.ndarray
        """
        if volume.dtype == np.uint16:
            return volume
        if volume.dtype == np.uint8:
            # 8-bit data: values unchanged, widened to uint16
            return volume.astype(np.uint16)

        vmin = float(volume.min())
        vmax = float(volume.max())
        if not np.issubdtype(volume.dtype, np.floating) and 0 <= vmin and vmax <= 65535:
            # Integer data already inside 0-65535: values unchanged
            return volume.astype(np.uint16)

        # Float data, or integer data outside 0-65535: linear [min, max] ->
        # [0, 65535]. Computed in float32 to keep the temporary small.
        converted = volume.astype(np.float32)
        converted -= vmin
        if vmax > vmin:
            converted *= 65535.0 / (vmax - vmin)
        else:
            converted.fill(0)
        OrsTMessageBox.message(None,
                               'Input data converted for BounTI: values linearly '
                               'remapped from [{:g}, {:g}] to [0, 65535].'.format(vmin, vmax),
                               'BounTI', OrsTMessageBox.Warning, OrsTMessageBox.Ok)
        return np.rint(converted).astype(np.uint16)

    @classmethod
    def _publishLabeledArrayAsMultiROI(cls, labeledArray, channel, titleSuffix):
        """Publishes a uint16 label array as a MultiROI beside ``channel``.

        Shipped write path (nnUNetInferer.publishSegmentationNDArrayAsMultiROI
        + DatasetHelper.createMultiROIFromDataset): the label values go into a
        temporary channel, DatasetHelper turns it into a MultiROI (each voxel
        value becomes the label of that value; preferences and default colors
        are applied inside the helper), then the title, label count and label
        names are set and the MultiROI is published.

        Labels stay uint16/USHORT: they are NOT cast to uint8 (the number of
        segments goes up to 1000, which does not fit a byte).

        :param labeledArray: uint16 [z, y, x] label array
        :param channel: the segmented channel (shape and title source)
        :param titleSuffix: appended to the channel title (e.g. '_BounTI')
        :return: the published MultiROI, or None on failure
        :rtype: ORSModel.ors.MultiROI
        """
        tmpChannel = Channel()
        tmpChannel.copyShapeFromStructuredGrid(channel)
        ORSModel.createChannelFromNumpyArray(labeledArray, channelGUID=tmpChannel.getGUID(), ZAxis=True)

        multiROI = DatasetHelper.createMultiROIFromDataset(tmpChannel, None)
        tmpChannel.deleteObject()
        if multiROI is None:
            return None

        multiROI.setTitle(channel.getTitle() + titleSuffix)

        # getAsMultiROIInArea maps each voxel value to the label of that
        # value, so the label count must cover the highest value (reducing the
        # count destroys labels). For the usual run the values are exactly
        # 1..k and the sequence below is setLabelCount(k) + 'Segment 1..k'.
        labelValues = [int(v) for v in np.unique(labeledArray) if v != 0]
        if labelValues:
            multiROI.setLabelCount(labelValues[-1])
            for labelValue in labelValues:
                multiROI.setLabelName(labelValue, 'Segment {}'.format(labelValue))

        multiROI.publish()
        return multiROI
