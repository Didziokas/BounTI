"""Main form of the BounTI plugin.

The widget tree is built in code with PyQt6 widgets/layouts (the shipped
plugins' Ui_* modules are plain objects constructing the same tree inside
setupUi; in-code construction is the equivalent pattern, without a
.ui/Designer/pyuic workflow).

Named mainform_bounti (the shipped mainform_* form-module pattern): on
Windows' case-insensitive filesystem a form module named 'bounti.py' would
collide with the plugin module 'BounTI.py'.
"""

from PyQt6.QtCore import pyqtSlot
from PyQt6.QtWidgets import (QCheckBox, QComboBox, QFormLayout, QHBoxLayout,
                             QPushButton, QSpinBox, QVBoxLayout)

from ORSModel import Channel, MultiROI, orsObj
from OrsLibraries.workingcontext import WorkingContext
from ORSServiceClass.ORSWidget.orsobjectclasscombobox.orsobjectclasscombobox import OrsObjectClassComboBox
from ORSServiceClass.windowclasses.orsabstractwindow import OrsAbstractWindow


class BounTIForm(OrsAbstractWindow):

    def __init__(self, implementation, parent=None):
        super().__init__(implementation, parent)

        # ---- widgets (defaults and ranges match the Avizo addon) -----------
        self.comboBoxChannel = OrsObjectClassComboBox(self)
        self.comboBoxChannel.setObjectName('comboBoxChannel')
        self.comboBoxChannel.setToolTip('Channel to segment')

        self.comboBoxSeed = QComboBox(self)
        self.comboBoxSeed.setObjectName('comboBoxSeed')
        self.comboBoxSeed.setToolTip(
            'Optional MultiROI whose labels seed the segmentation; with '
            "'(none)' the seed is formed from the components above the "
            'Initial Threshold')

        self.initialThreshold_spinbox = QSpinBox(self)
        self.initialThreshold_spinbox.setObjectName('initialThreshold_spinbox')
        self.initialThreshold_spinbox.setRange(0, 100000)
        self.initialThreshold_spinbox.setValue(34000)
        self.initialThreshold_spinbox.setToolTip(
            'High grey value giving adequate separation of anatomical '
            'elements — place at/just right of the bone peak start; if '
            'segments merge, raise it; if bones vanish, lower it')

        self.targetThreshold_spinbox = QSpinBox(self)
        self.targetThreshold_spinbox.setObjectName('targetThreshold_spinbox')
        self.targetThreshold_spinbox.setRange(0, 100000)
        self.targetThreshold_spinbox.setValue(21500)
        self.targetThreshold_spinbox.setToolTip(
            'Low grey value giving the desired bone definition — usually '
            'just right of the soft-tissue peak')

        self.segments_spinbox = QSpinBox(self)
        self.segments_spinbox.setObjectName('segments_spinbox')
        # Deviation from the Avizo addon (clamp 1-100): widened to 1-1000 so
        # that cases like the paper's NS=200 lizard are possible.
        self.segments_spinbox.setRange(1, 1000)
        self.segments_spinbox.setValue(14)
        self.segments_spinbox.setToolTip(
            'Slightly above the number of anatomical components you expect')

        self.iterations_spinbox = QSpinBox(self)
        self.iterations_spinbox.setObjectName('iterations_spinbox')
        self.iterations_spinbox.setRange(1, 1000)
        self.iterations_spinbox.setValue(7)
        self.iterations_spinbox.setToolTip(
            'Start at 20; raise to 100-200 for finer boundaries between '
            'adjacent segments')

        self.seedDilation_checkbox = QCheckBox('Seed Dilation', self)
        self.seedDilation_checkbox.setObjectName('seedDilation_checkbox')
        self.seedDilation_checkbox.setChecked(False)
        self.seedDilation_checkbox.setToolTip(
            'Dilate each seed label by one voxel before the threshold sweep '
            '(default OFF per the BounTI user manual)')

        self.labelPreservation_checkbox = QCheckBox('Label Preservation', self)
        self.labelPreservation_checkbox.setObjectName('labelPreservation_checkbox')
        self.labelPreservation_checkbox.setChecked(False)
        self.labelPreservation_checkbox.setToolTip(
            'Use the seed MultiROI label values as-is instead of re-selecting '
            'the largest components (only with a seed MultiROI selected)')

        self.saveSeed_checkbox = QCheckBox('Save Seed', self)
        self.saveSeed_checkbox.setObjectName('saveSeed_checkbox')
        self.saveSeed_checkbox.setChecked(False)
        self.saveSeed_checkbox.setToolTip(
            'Also create a MultiROI with the seed used for the segmentation')

        self.apply_pushButton = QPushButton('Apply', self)
        self.apply_pushButton.setObjectName('apply_pushButton')

        # ---- layout ---------------------------------------------------------
        formLayout = QFormLayout()
        formLayout.addRow('Channel', self.comboBoxChannel)
        formLayout.addRow('Seed MultiROI', self.comboBoxSeed)
        formLayout.addRow('Initial Threshold', self.initialThreshold_spinbox)
        formLayout.addRow('Target Threshold', self.targetThreshold_spinbox)
        formLayout.addRow('Number of Segments', self.segments_spinbox)
        formLayout.addRow('Number of Iterations', self.iterations_spinbox)

        buttonLayout = QHBoxLayout()
        buttonLayout.addStretch()
        buttonLayout.addWidget(self.apply_pushButton)
        buttonLayout.addStretch()

        mainLayout = QVBoxLayout(self)
        mainLayout.addLayout(formLayout)
        mainLayout.addWidget(self.seedDilation_checkbox)
        mainLayout.addWidget(self.labelPreservation_checkbox)
        mainLayout.addWidget(self.saveSeed_checkbox)
        mainLayout.addLayout(buttonLayout)

        WorkingContext.registerOrsWidget('BounTI', implementation, 'MainForm', self)

        # ---- connections ---------------------------------------------------
        self.comboBoxChannel.setManagedClass([Channel])
        self.comboBoxChannel.setImplementation(implementation)
        self.comboBoxChannel.currentObjectChanged.connect(self.on_comboBoxChannel_currentObjectChanged)
        self.comboBoxSeed.currentIndexChanged.connect(self.on_comboBoxSeed_currentIndexChanged)
        self.apply_pushButton.clicked.connect(self.on_apply_pushButton_clicked)

        self.channel = None
        self.refreshSeedComboBox()

    @pyqtSlot(object)
    def on_comboBoxChannel_currentObjectChanged(self, obj):
        if obj is not None:
            self.channel = obj

    @pyqtSlot(int)
    def on_comboBoxSeed_currentIndexChanged(self):
        self.updateLabelPreservationEnabled()

    @pyqtSlot()
    def on_apply_pushButton_clicked(self):
        self.getImplementation().apply()

    def refreshUI(self, channel):
        if channel is None:
            return

        self.channel = channel
        self.comboBoxChannel.blockSignals(True)
        self.comboBoxChannel.selectObject(channel)
        self.comboBoxChannel.updateComboBox()
        self.comboBoxChannel.blockSignals(False)
        self.refreshSeedComboBox()

    def refreshSeedComboBox(self):
        """Rebuilds the seed list: '(none)' plus every MultiROI available in
        the current context (the current selection is kept when it still
        exists)."""
        currentGuid = self.comboBoxSeed.currentData()
        self.comboBoxSeed.blockSignals(True)
        self.comboBoxSeed.clear()
        self.comboBoxSeed.addItem('(none)')
        for aMultiROI in MultiROI.getAllInstances():
            if aMultiROI.getIsAvailableInContext():
                self.comboBoxSeed.addItem(aMultiROI.getTitle(), aMultiROI.getGUID())
        if currentGuid is not None:
            index = self.comboBoxSeed.findData(currentGuid)
            if index >= 0:
                self.comboBoxSeed.setCurrentIndex(index)
        self.comboBoxSeed.blockSignals(False)
        self.updateLabelPreservationEnabled()

    def updateLabelPreservationEnabled(self):
        # Label Preservation only applies to a supplied seed (parity with the
        # Avizo addon, where the parameter only appears with a label input)
        self.labelPreservation_checkbox.setEnabled(self.comboBoxSeed.currentIndex() > 0)

    # ---- values read by the plugin's Apply action ---------------------------
    def getChannel(self):
        return self.channel

    def getInitialThreshold(self):
        return self.initialThreshold_spinbox.value()

    def getTargetThreshold(self):
        return self.targetThreshold_spinbox.value()

    def getNumberOfSegments(self):
        return self.segments_spinbox.value()

    def getNumberOfIterations(self):
        return self.iterations_spinbox.value()

    def getSeedDilation(self):
        return self.seedDilation_checkbox.isChecked()

    def getLabelPreservation(self):
        return self.labelPreservation_checkbox.isChecked()

    def getSaveSeed(self):
        return self.saveSeed_checkbox.isChecked()

    def getSeedMultiROI(self):
        if self.comboBoxSeed.currentIndex() <= 0:
            return None
        return orsObj(self.comboBoxSeed.currentData())
