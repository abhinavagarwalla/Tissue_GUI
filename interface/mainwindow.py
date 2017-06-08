from PyQt5 import QtCore, QtWidgets
from PyQt5.QtGui import QPixmap
from PyQt5.QtWidgets import QFileDialog

from dl_interface.model_config import *
from dl_interface.model_test import Test
from interface.image_ops import DisplayImage
from interface.image_slide import ImageClass
from dl_interface.gen_patch import PatchGenerator
from dl_interface.convert_dataset import TFRecordConverter
from dl_interface.model_train import Train

class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(1306, 772)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Preferred)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(MainWindow.sizePolicy().hasHeightForWidth())
        MainWindow.setSizePolicy(sizePolicy)
        MainWindow.setMouseTracking(True)
        self.centralWidget = QtWidgets.QWidget(MainWindow)
        self.centralWidget.setMouseTracking(True)
        self.centralWidget.setObjectName("centralWidget")
        self.horizontalLayout_4 = QtWidgets.QHBoxLayout(self.centralWidget)
        self.horizontalLayout_4.setContentsMargins(11, 11, 11, 11)
        self.horizontalLayout_4.setSpacing(6)
        self.horizontalLayout_4.setObjectName("horizontalLayout_4")
        self.tabs = QtWidgets.QTabWidget(self.centralWidget)
        self.tabs.setMouseTracking(True)
        self.tabs.setMovable(False)
        self.tabs.setObjectName("tabs")
        self.vis = QtWidgets.QWidget()
        self.vis.setMinimumSize(QtCore.QSize(1282, 675))
        self.vis.setObjectName("vis")
        self.horizontalLayout_5 = QtWidgets.QHBoxLayout(self.vis)
        self.horizontalLayout_5.setContentsMargins(11, 11, 11, 11)
        self.horizontalLayout_5.setSpacing(6)
        self.horizontalLayout_5.setObjectName("horizontalLayout_5")
        self.verticalLayout_4 = QtWidgets.QVBoxLayout()
        self.verticalLayout_4.setSizeConstraint(QtWidgets.QLayout.SetDefaultConstraint)
        self.verticalLayout_4.setContentsMargins(5, 5, 5, 5)
        self.verticalLayout_4.setSpacing(6)
        self.verticalLayout_4.setObjectName("verticalLayout_4")
        self.verticalLayout = QtWidgets.QVBoxLayout()
        self.verticalLayout.setContentsMargins(11, 11, 11, 11)
        self.verticalLayout.setSpacing(6)
        self.verticalLayout.setObjectName("verticalLayout")
        self.load_image = QtWidgets.QPushButton(self.vis)
        self.load_image.setObjectName("load_image")
        self.verticalLayout.addWidget(self.load_image)
        self.save_image = QtWidgets.QPushButton(self.vis)
        self.save_image.setObjectName("save_image")
        self.verticalLayout.addWidget(self.save_image)
        self.info = QtWidgets.QLabel(self.vis)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Preferred)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.info.sizePolicy().hasHeightForWidth())
        self.info.setSizePolicy(sizePolicy)
        self.info.setMouseTracking(True)
        self.info.setObjectName("info")
        self.verticalLayout.addWidget(self.info)
        self.verticalLayout_4.addLayout(self.verticalLayout)
        self.zoomSlider = QtWidgets.QSlider(self.vis)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.zoomSlider.sizePolicy().hasHeightForWidth())
        self.zoomSlider.setSizePolicy(sizePolicy)
        self.zoomSlider.setPageStep(1)
        self.zoomSlider.setOrientation(QtCore.Qt.Horizontal)
        self.zoomSlider.setTickPosition(QtWidgets.QSlider.TicksAbove)
        self.zoomSlider.setTickInterval(1)
        self.zoomSlider.setObjectName("zoomSlider")
        self.verticalLayout_4.addWidget(self.zoomSlider)
        self.overlay_group = QtWidgets.QGroupBox(self.vis)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.overlay_group.sizePolicy().hasHeightForWidth())
        self.overlay_group.setSizePolicy(sizePolicy)
        self.overlay_group.setObjectName("overlay_group")
        self.gridLayout_2 = QtWidgets.QGridLayout(self.overlay_group)
        self.gridLayout_2.setContentsMargins(11, 11, 11, 11)
        self.gridLayout_2.setSpacing(6)
        self.gridLayout_2.setObjectName("gridLayout_2")
        self.class_0 = QtWidgets.QCheckBox(self.overlay_group)
        self.class_0.setObjectName("class_0")
        self.gridLayout_2.addWidget(self.class_0, 0, 0, 1, 1)
        self.class_1 = QtWidgets.QCheckBox(self.overlay_group)
        self.class_1.setObjectName("class_1")
        self.gridLayout_2.addWidget(self.class_1, 0, 1, 1, 1)
        self.class_2 = QtWidgets.QCheckBox(self.overlay_group)
        self.class_2.setObjectName("class_2")
        self.gridLayout_2.addWidget(self.class_2, 1, 0, 1, 1)
        self.class_3 = QtWidgets.QCheckBox(self.overlay_group)
        self.class_3.setObjectName("class_3")
        self.gridLayout_2.addWidget(self.class_3, 1, 1, 1, 1)
        self.class_4 = QtWidgets.QCheckBox(self.overlay_group)
        self.class_4.setObjectName("class_4")
        self.gridLayout_2.addWidget(self.class_4, 2, 0, 1, 1)
        self.class_5 = QtWidgets.QCheckBox(self.overlay_group)
        self.class_5.setObjectName("class_5")
        self.gridLayout_2.addWidget(self.class_5, 2, 1, 1, 1)
        self.verticalLayout_4.addWidget(self.overlay_group)
        self.gridLayout = QtWidgets.QGridLayout()
        self.gridLayout.setContentsMargins(5, 5, 5, 5)
        self.gridLayout.setSpacing(6)
        self.gridLayout.setObjectName("gridLayout")
        self.current_level = QtWidgets.QLineEdit(self.vis)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.current_level.sizePolicy().hasHeightForWidth())
        self.current_level.setSizePolicy(sizePolicy)
        self.current_level.setReadOnly(True)
        self.current_level.setObjectName("current_level")
        self.gridLayout.addWidget(self.current_level, 1, 1, 1, 1)
        self.current_level_label = QtWidgets.QLabel(self.vis)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.current_level_label.sizePolicy().hasHeightForWidth())
        self.current_level_label.setSizePolicy(sizePolicy)
        self.current_level_label.setObjectName("current_level_label")
        self.gridLayout.addWidget(self.current_level_label, 1, 0, 1, 1)
        self.curX = QtWidgets.QLineEdit(self.vis)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.curX.sizePolicy().hasHeightForWidth())
        self.curX.setSizePolicy(sizePolicy)
        self.curX.setObjectName("curX")
        self.gridLayout.addWidget(self.curX, 0, 0, 1, 1)
        self.curY = QtWidgets.QLineEdit(self.vis)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.curY.sizePolicy().hasHeightForWidth())
        self.curY.setSizePolicy(sizePolicy)
        self.curY.setObjectName("curY")
        self.gridLayout.addWidget(self.curY, 0, 1, 1, 1)
        self.verticalLayout_4.addLayout(self.gridLayout)
        self.horizontalLayout_5.addLayout(self.verticalLayout_4)
        self.verticalLayout_3 = QtWidgets.QVBoxLayout()
        self.verticalLayout_3.setContentsMargins(5, 5, 5, 5)
        self.verticalLayout_3.setSpacing(6)
        self.verticalLayout_3.setObjectName("verticalLayout_3")
        self.horizontalLayout_3 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_3.setContentsMargins(5, 5, 5, 5)
        self.horizontalLayout_3.setSpacing(6)
        self.horizontalLayout_3.setObjectName("horizontalLayout_3")
        self.overlay_method = QtWidgets.QComboBox(self.vis)
        self.overlay_method.setObjectName("overlay_method")
        self.overlay_method.addItem("")
        self.overlay_method.addItem("")
        self.overlay_method.addItem("")
        self.overlay_method.addItem("")
        self.overlay_method.addItem("")
        self.horizontalLayout_3.addWidget(self.overlay_method)
        self.load_overlay = QtWidgets.QPushButton(self.vis)
        self.load_overlay.setObjectName("load_overlay")
        self.horizontalLayout_3.addWidget(self.load_overlay)
        self.overlay_side_by_side = QtWidgets.QCheckBox(self.vis)
        self.overlay_side_by_side.setObjectName("overlay_side_by_side")
        self.horizontalLayout_3.addWidget(self.overlay_side_by_side)
        self.check_segmentation = QtWidgets.QCheckBox(self.vis)
        self.check_segmentation.setObjectName("check_segmentation")
        self.horizontalLayout_3.addWidget(self.check_segmentation)
        self.check_tumor_region = QtWidgets.QCheckBox(self.vis)
        self.check_tumor_region.setObjectName("check_tumor_region")
        self.horizontalLayout_3.addWidget(self.check_tumor_region)
        self.check_heatmap = QtWidgets.QCheckBox(self.vis)
        self.check_heatmap.setObjectName("check_heatmap")
        self.horizontalLayout_3.addWidget(self.check_heatmap)
        self.check_nuclei = QtWidgets.QCheckBox(self.vis)
        self.check_nuclei.setObjectName("check_nuclei")
        self.horizontalLayout_3.addWidget(self.check_nuclei)
        self.check_others = QtWidgets.QCheckBox(self.vis)
        self.check_others.setObjectName("check_others")
        self.horizontalLayout_3.addWidget(self.check_others)
        self.verticalLayout_3.addLayout(self.horizontalLayout_3)
        self.horizontalLayout_2 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_2.setSizeConstraint(QtWidgets.QLayout.SetDefaultConstraint)
        self.horizontalLayout_2.setContentsMargins(5, 5, 5, 5)
        self.horizontalLayout_2.setSpacing(6)
        self.horizontalLayout_2.setObjectName("horizontalLayout_2")
        self.orig_image = QtWidgets.QLabel(self.vis)
        self.orig_image.setEnabled(True)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.MinimumExpanding,
                                           QtWidgets.QSizePolicy.MinimumExpanding)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.orig_image.sizePolicy().hasHeightForWidth())
        self.orig_image.setSizePolicy(sizePolicy)
        self.orig_image.setMinimumSize(QtCore.QSize(0, 0))
        self.orig_image.setMouseTracking(True)
        self.orig_image.setAutoFillBackground(False)
        self.orig_image.setScaledContents(True)
        self.orig_image.setObjectName("orig_image")
        self.horizontalLayout_2.addWidget(self.orig_image)
        self.overlay_image = QtWidgets.QLabel(self.vis)
        self.overlay_image.setEnabled(True)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.MinimumExpanding,
                                           QtWidgets.QSizePolicy.MinimumExpanding)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.overlay_image.sizePolicy().hasHeightForWidth())
        self.overlay_image.setSizePolicy(sizePolicy)
        self.overlay_image.setMinimumSize(QtCore.QSize(0, 0))
        self.overlay_image.setMouseTracking(True)
        self.overlay_image.setToolTipDuration(-1)
        self.overlay_image.setScaledContents(True)
        self.overlay_image.setObjectName("overlay_image")
        self.horizontalLayout_2.addWidget(self.overlay_image)
        self.verticalLayout_3.addLayout(self.horizontalLayout_2)
        self.horizontalLayout_5.addLayout(self.verticalLayout_3)
        self.tabs.addTab(self.vis, "")
        self.training = QtWidgets.QWidget()
        self.training.setObjectName("training")
        self.testBox = QtWidgets.QGroupBox(self.training)
        self.testBox.setGeometry(QtCore.QRect(20, 400, 701, 251))
        self.testBox.setCheckable(False)
        self.testBox.setObjectName("testBox")
        self.test_progress = QtWidgets.QProgressBar(self.testBox)
        self.test_progress.setGeometry(QtCore.QRect(20, 200, 321, 23))
        self.test_progress.setProperty("value", 0)
        self.test_progress.setObjectName("test_progress")
        self.mask_path = QtWidgets.QLineEdit(self.testBox)
        self.mask_path.setEnabled(False)
        self.mask_path.setGeometry(QtCore.QRect(230, 110, 441, 20))
        self.mask_path.setReadOnly(True)
        self.mask_path.setObjectName("mask_path")
        self.select_wsi_level = QtWidgets.QComboBox(self.testBox)
        self.select_wsi_level.setEnabled(False)
        self.select_wsi_level.setGeometry(QtCore.QRect(70, 150, 41, 22))
        self.select_wsi_level.setObjectName("select_wsi_level")
        self.stop_eval = QtWidgets.QPushButton(self.testBox)
        self.stop_eval.setGeometry(QtCore.QRect(550, 150, 121, 23))
        self.stop_eval.setObjectName("stop_eval")
        self.select_mask_level = QtWidgets.QComboBox(self.testBox)
        self.select_mask_level.setEnabled(False)
        self.select_mask_level.setGeometry(QtCore.QRect(200, 150, 41, 22))
        self.select_mask_level.setObjectName("select_mask_level")
        self.start_eval = QtWidgets.QPushButton(self.testBox)
        self.start_eval.setGeometry(QtCore.QRect(410, 150, 121, 23))
        self.start_eval.setObjectName("start_eval")
        self.image_path = QtWidgets.QLineEdit(self.testBox)
        self.image_path.setEnabled(False)
        self.image_path.setGeometry(QtCore.QRect(230, 30, 441, 20))
        self.image_path.setReadOnly(True)
        self.image_path.setObjectName("image_path")
        self.select_model = QtWidgets.QPushButton(self.testBox)
        self.select_model.setGeometry(QtCore.QRect(10, 70, 212, 23))
        self.select_model.setObjectName("select_model")
        self.model_path = QtWidgets.QLineEdit(self.testBox)
        self.model_path.setEnabled(False)
        self.model_path.setGeometry(QtCore.QRect(230, 70, 441, 20))
        self.model_path.setReadOnly(True)
        self.model_path.setObjectName("model_path")
        self.label_2 = QtWidgets.QLabel(self.testBox)
        self.label_2.setGeometry(QtCore.QRect(10, 150, 61, 21))
        self.label_2.setObjectName("label_2")
        self.select_patch_size = QtWidgets.QLineEdit(self.testBox)
        self.select_patch_size.setEnabled(False)
        self.select_patch_size.setGeometry(QtCore.QRect(360, 150, 41, 20))
        self.select_patch_size.setObjectName("select_patch_size")
        self.select_image_train = QtWidgets.QPushButton(self.testBox)
        self.select_image_train.setGeometry(QtCore.QRect(10, 30, 212, 23))
        self.select_image_train.setObjectName("select_image_train")
        self.label_3 = QtWidgets.QLabel(self.testBox)
        self.label_3.setGeometry(QtCore.QRect(300, 150, 51, 21))
        self.label_3.setObjectName("label_3")
        self.select_mask = QtWidgets.QPushButton(self.testBox)
        self.select_mask.setGeometry(QtCore.QRect(10, 110, 212, 23))
        self.select_mask.setObjectName("select_mask")
        self.label_4 = QtWidgets.QLabel(self.testBox)
        self.label_4.setGeometry(QtCore.QRect(140, 150, 61, 21))
        self.label_4.setObjectName("label_4")
        self.plainTextEdit = QtWidgets.QPlainTextEdit(self.testBox)
        self.plainTextEdit.setEnabled(False)
        self.plainTextEdit.setGeometry(QtCore.QRect(340, 190, 341, 41))
        self.plainTextEdit.setReadOnly(True)
        self.plainTextEdit.setObjectName("plainTextEdit")
        self.trainBox = QtWidgets.QGroupBox(self.training)
        self.trainBox.setGeometry(QtCore.QRect(20, 20, 491, 351))
        self.trainBox.setObjectName("trainBox")
        self.label_5 = QtWidgets.QLabel(self.trainBox)
        self.label_5.setGeometry(QtCore.QRect(40, 70, 71, 16))
        self.label_5.setObjectName("label_5")
        self.label_6 = QtWidgets.QLabel(self.trainBox)
        self.label_6.setGeometry(QtCore.QRect(40, 110, 61, 16))
        self.label_6.setObjectName("label_6")
        self.label_7 = QtWidgets.QLabel(self.trainBox)
        self.label_7.setGeometry(QtCore.QRect(40, 150, 111, 16))
        self.label_7.setObjectName("label_7")
        self.label_8 = QtWidgets.QLabel(self.trainBox)
        self.label_8.setGeometry(QtCore.QRect(40, 190, 71, 16))
        self.label_8.setObjectName("label_8")
        self.label_9 = QtWidgets.QLabel(self.trainBox)
        self.label_9.setGeometry(QtCore.QRect(40, 230, 41, 16))
        self.label_9.setObjectName("label_9")
        self.comboBox = QtWidgets.QComboBox(self.trainBox)
        self.comboBox.setGeometry(QtCore.QRect(170, 60, 69, 22))
        self.comboBox.setObjectName("comboBox")
        self.comboBox_2 = QtWidgets.QComboBox(self.trainBox)
        self.comboBox_2.setGeometry(QtCore.QRect(170, 100, 69, 22))
        self.comboBox_2.setObjectName("comboBox_2")
        self.comboBox_3 = QtWidgets.QComboBox(self.trainBox)
        self.comboBox_3.setGeometry(QtCore.QRect(170, 150, 69, 22))
        self.comboBox_3.setObjectName("comboBox_3")
        self.lineEdit = QtWidgets.QLineEdit(self.trainBox)
        self.lineEdit.setGeometry(QtCore.QRect(170, 190, 113, 20))
        self.lineEdit.setObjectName("lineEdit")
        self.comboBox_4 = QtWidgets.QComboBox(self.trainBox)
        self.comboBox_4.setGeometry(QtCore.QRect(170, 220, 69, 22))
        self.comboBox_4.setObjectName("comboBox_4")
        self.label_10 = QtWidgets.QLabel(self.trainBox)
        self.label_10.setGeometry(QtCore.QRect(40, 260, 41, 16))
        self.label_10.setObjectName("label_10")
        self.lineEdit_2 = QtWidgets.QLineEdit(self.trainBox)
        self.lineEdit_2.setGeometry(QtCore.QRect(170, 260, 113, 20))
        self.lineEdit_2.setObjectName("lineEdit_2")
        self.label_13 = QtWidgets.QLabel(self.trainBox)
        self.label_13.setGeometry(QtCore.QRect(40, 290, 81, 16))
        self.label_13.setObjectName("label_13")
        self.label_14 = QtWidgets.QLabel(self.trainBox)
        self.label_14.setGeometry(QtCore.QRect(40, 320, 47, 13))
        self.label_14.setObjectName("label_14")
        self.lineEdit_3 = QtWidgets.QLineEdit(self.trainBox)
        self.lineEdit_3.setGeometry(QtCore.QRect(170, 30, 113, 20))
        self.lineEdit_3.setObjectName("lineEdit_3")
        self.pushButton = QtWidgets.QPushButton(self.trainBox)
        self.pushButton.setGeometry(QtCore.QRect(40, 30, 111, 23))
        self.pushButton.setObjectName("pushButton")
        self.start_train = QtWidgets.QPushButton(self.trainBox)
        self.start_train.setGeometry(QtCore.QRect(340, 170, 121, 23))
        self.start_train.setObjectName("start_train")
        self.stop_train = QtWidgets.QPushButton(self.trainBox)
        self.stop_train.setGeometry(QtCore.QRect(340, 200, 121, 23))
        self.stop_train.setObjectName("stop_train")
        self.patchBox = QtWidgets.QGroupBox(self.training)
        self.patchBox.setGeometry(QtCore.QRect(550, 30, 531, 191))
        self.patchBox.setObjectName("patchBox")
        self.gen_image_path = QtWidgets.QLineEdit(self.patchBox)
        self.gen_image_path.setEnabled(False)
        self.gen_image_path.setGeometry(QtCore.QRect(150, 30, 371, 20))
        self.gen_image_path.setObjectName("gen_image_path")
        self.select_gen_images = QtWidgets.QPushButton(self.patchBox)
        self.select_gen_images.setGeometry(QtCore.QRect(30, 30, 111, 23))
        self.select_gen_images.setObjectName("select_gen_images")
        self.select_gen_patch_size = QtWidgets.QLineEdit(self.patchBox)
        self.select_gen_patch_size.setEnabled(False)
        self.select_gen_patch_size.setGeometry(QtCore.QRect(230, 70, 41, 20))
        self.select_gen_patch_size.setObjectName("select_gen_patch_size")
        self.select_gen_wsi_level = QtWidgets.QComboBox(self.patchBox)
        self.select_gen_wsi_level.setEnabled(False)
        self.select_gen_wsi_level.setGeometry(QtCore.QRect(100, 70, 41, 22))
        self.select_gen_wsi_level.setObjectName("select_gen_wsi_level")
        self.label_15 = QtWidgets.QLabel(self.patchBox)
        self.label_15.setGeometry(QtCore.QRect(40, 70, 61, 21))
        self.label_15.setObjectName("label_15")
        self.label_16 = QtWidgets.QLabel(self.patchBox)
        self.label_16.setGeometry(QtCore.QRect(170, 70, 51, 21))
        self.label_16.setObjectName("label_16")
        self.start_gen_patch = QtWidgets.QPushButton(self.patchBox)
        self.start_gen_patch.setGeometry(QtCore.QRect(120, 110, 121, 23))
        self.start_gen_patch.setObjectName("start_gen_patch")
        self.stop_gen_patch = QtWidgets.QPushButton(self.patchBox)
        self.stop_gen_patch.setGeometry(QtCore.QRect(270, 110, 121, 23))
        self.stop_gen_patch.setObjectName("stop_gen_patch")
        self.patch_progress = QtWidgets.QProgressBar(self.patchBox)
        self.patch_progress.setGeometry(QtCore.QRect(300, 70, 221, 23))
        self.patch_progress.setProperty("value", 0)
        self.patch_progress.setObjectName("patch_progress")
        self.start_tf_record = QtWidgets.QPushButton(self.patchBox)
        self.start_tf_record.setGeometry(QtCore.QRect(200, 150, 121, 23))
        self.start_tf_record.setObjectName("start_tf_record")
        self.tabs.addTab(self.training, "")
        self.horizontalLayout_4.addWidget(self.tabs)
        MainWindow.setCentralWidget(self.centralWidget)
        self.menuBar = QtWidgets.QMenuBar(MainWindow)
        self.menuBar.setGeometry(QtCore.QRect(0, 0, 1306, 21))
        self.menuBar.setObjectName("menuBar")
        self.menuWindow = QtWidgets.QMenu(self.menuBar)
        self.menuWindow.setObjectName("menuWindow")
        MainWindow.setMenuBar(self.menuBar)
        self.mainToolBar = QtWidgets.QToolBar(MainWindow)
        self.mainToolBar.setObjectName("mainToolBar")
        MainWindow.addToolBar(QtCore.Qt.TopToolBarArea, self.mainToolBar)
        self.statusBar = QtWidgets.QStatusBar(MainWindow)
        self.statusBar.setObjectName("statusBar")
        MainWindow.setStatusBar(self.statusBar)
        self.menuBar.addAction(self.menuWindow.menuAction())

        self.initialize_signals_slots()
        self.initialize_worker_thread()
        self.retranslateUi(MainWindow)
        self.overlay_method.setCurrentIndex(0)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "MainWindow"))
        MainWindow.setToolTip(_translate("MainWindow", "Inside Main Window"))
        self.load_image.setText(_translate("MainWindow", "Load Image"))
        self.save_image.setText(_translate("MainWindow", "PushButton"))
        self.info.setText(_translate("MainWindow", "TextLabel"))
        self.overlay_group.setTitle(_translate("MainWindow", "GroupBox"))
        self.class_0.setText(_translate("MainWindow", "CheckBox"))
        self.class_1.setText(_translate("MainWindow", "CheckBox"))
        self.class_2.setText(_translate("MainWindow", "CheckBox"))
        self.class_3.setText(_translate("MainWindow", "CheckBox"))
        self.class_4.setText(_translate("MainWindow", "CheckBox"))
        self.class_5.setText(_translate("MainWindow", "CheckBox"))
        self.current_level.setPlaceholderText(_translate("MainWindow", "NA"))
        self.current_level_label.setText(_translate("MainWindow", "Current Level"))
        self.overlay_method.setItemText(0, _translate("MainWindow", "Segmentation Mask (by Pixel)"))
        self.overlay_method.setItemText(1, _translate("MainWindow", "Tumor Region"))
        self.overlay_method.setItemText(2, _translate("MainWindow", "Heatmap"))
        self.overlay_method.setItemText(3, _translate("MainWindow", "Nuclei Position"))
        self.overlay_method.setItemText(4, _translate("MainWindow", "Segmentation Mask (By Patch)"))
        self.load_overlay.setText(_translate("MainWindow", "Load Overlay"))
        self.overlay_side_by_side.setText(_translate("MainWindow", "Overlay Side-by-Side"))
        self.check_segmentation.setText(_translate("MainWindow", "Segmentation"))
        self.check_tumor_region.setText(_translate("MainWindow", "Tumor Region"))
        self.check_heatmap.setText(_translate("MainWindow", "HeatMap"))
        self.check_nuclei.setText(_translate("MainWindow", "Nuclei Position"))
        self.check_others.setText(_translate("MainWindow", "Others"))
        self.orig_image.setToolTip(_translate("MainWindow", "\"Over the Original Image\""))
        self.orig_image.setText(_translate("MainWindow", "Original Image"))
        self.overlay_image.setToolTip(_translate("MainWindow", "\"Overlay Image\""))
        self.overlay_image.setText(_translate("MainWindow", "Overlay Image"))
        self.tabs.setTabText(self.tabs.indexOf(self.vis), _translate("MainWindow", "Visualisation"))
        self.testBox.setTitle(_translate("MainWindow", "Testing"))
        self.stop_eval.setText(_translate("MainWindow", "Stop Evaluation"))
        self.start_eval.setText(_translate("MainWindow", "Start Evaluation"))
        self.select_model.setText(_translate("MainWindow", "Select Model"))
        self.label_2.setText(_translate("MainWindow", "WSI Level:"))
        self.select_image_train.setText(_translate("MainWindow", "Select Image"))
        self.label_3.setText(_translate("MainWindow", "Patch Size"))
        self.select_mask.setText(_translate("MainWindow", "Select Mask"))
        self.label_4.setText(_translate("MainWindow", "Mask Level:"))
        self.trainBox.setTitle(_translate("MainWindow", "Training"))
        self.label_5.setText(_translate("MainWindow", "Preprocessing"))
        self.label_6.setText(_translate("MainWindow", "Optimiser"))
        self.label_7.setText(_translate("MainWindow", "Network Architecture"))
        self.label_8.setText(_translate("MainWindow", "Learning Rate"))
        self.label_9.setText(_translate("MainWindow", "Loss"))
        self.label_10.setText(_translate("MainWindow", "Epoch"))
        self.label_13.setText(_translate("MainWindow", "Problem Type:"))
        self.label_14.setText(_translate("MainWindow", "Labels"))
        self.pushButton.setText(_translate("MainWindow", "Training Dataset"))
        self.start_train.setText(_translate("MainWindow", "Start Training"))
        self.stop_train.setText(_translate("MainWindow", "Stop Training"))
        self.patchBox.setTitle(_translate("MainWindow", "Generate Patches"))
        self.select_gen_images.setText(_translate("MainWindow", "Images"))
        self.label_15.setText(_translate("MainWindow", "WSI Level:"))
        self.label_16.setText(_translate("MainWindow", "Patch Size"))
        self.start_gen_patch.setText(_translate("MainWindow", "Generate Patches"))
        self.stop_gen_patch.setText(_translate("MainWindow", "Cancel"))
        self.start_tf_record.setText(_translate("MainWindow", "Convert to TFRecord"))
        self.tabs.setTabText(self.tabs.indexOf(self.training), _translate("MainWindow", "Training"))
        self.menuWindow.setTitle(_translate("MainWindow", "Window"))

    def initialize_signals_slots(self):
        # Bind all the signal and slots here
        self.if_image = False
        self.if_image_overlay = 0
        self.c_zoom_level = 0
        self.prev_mouse_pos = None
        self.if_mouse_pressed = False
        self.load_image.clicked.connect(self.get_file)
        self.load_overlay.clicked.connect(self.get_file_overlay)
        self.zoomSlider.valueChanged.connect(self.updateBar)
        self.orig_image.mousePressEvent = self.mouse_orig
        self.orig_image.mouseReleaseEvent = self.mouse_orig_clear
        self.orig_image.mouseMoveEvent = self.mouse_orig
        self.orig_image.wheelEvent = self.wheel_zoom
        self.info.mousePressEvent = self.get_random_location

        ##Overlays
        self.overlay_side_by_side.stateChanged.connect(self.overlay_state_changed)
        self.overlay_image.hide()
        self.overlay_side_by_side.setEnabled(False)
        self.check_segmentation.setEnabled(False)
        self.check_tumor_region.setEnabled(False)
        self.check_heatmap.setEnabled(False)
        self.check_nuclei.setEnabled(False)
        self.check_others.setEnabled(False)
        self.overlay_states = {"Seg": False, "Reg": False, "Heat": False, "Nuclei": False}
        self.check_segmentation.stateChanged.connect(self.select_overlays)
        self.check_tumor_region.stateChanged.connect(self.select_overlays)
        self.check_heatmap.stateChanged.connect(self.select_overlays)
        self.check_nuclei.stateChanged.connect(self.select_overlays)
        self.check_others.stateChanged.connect(self.select_overlays)

        self.overlay_group.hide()
        self.overlay_group.setEnabled(False)
        self.overlay_group_dict = {0: self.class_0, 1: self.class_1, 2: self.class_2,
                                   3: self.class_3, 4: self.class_4, 5: self.class_5}
        self.overlay_group_states = {0: False, 1: False, 2: False,
                                   3: False, 4: False, 5: False}
        self.colors = [(255, 0, 255, 255), (255, 0, 0, 255), (0, 255, 0, 255),
                       (255, 128, 0, 255), (0, 0, 0, 255), (0, 0, 255, 255)]
        self.class_0.stateChanged.connect(self.select_class)
        self.class_1.stateChanged.connect(self.select_class)
        self.class_2.stateChanged.connect(self.select_class)
        self.class_3.stateChanged.connect(self.select_class)
        self.class_4.stateChanged.connect(self.select_class)
        self.class_5.stateChanged.connect(self.select_class)

        ##DL Part
        self.select_image_train.clicked.connect(self.select_WSI)
        self.select_model.clicked.connect(self.select_dl_model)
        self.select_mask.clicked.connect(self.select_mask_path)
        self.start_eval.clicked.connect(self.start_testing)

        self.select_gen_images.clicked.connect(self.select_gen_WSI)
        self.start_gen_patch.clicked.connect(self.start_generating_patch)
        self.start_tf_record.clicked.connect(self.start_creating_dataset)
        self.start_train.clicked.connect(self.start_training)

    def initialize_worker_thread(self):
        self.test_model = Test()
        self.thread_test = QtCore.QThread()
        self.test_model.epoch.connect(self.update_test_progress)
        self.test_model.moveToThread(self.thread_test)
        self.test_model.finished.connect(self.thread_test.quit)
        self.thread_test.started.connect(self.test_model.test)
        self.stop_eval.clicked.connect(lambda: self.test_model.stop_call())

        self.generate_patches = PatchGenerator()
        self.thread_patch = QtCore.QThread()
        self.generate_patches.moveToThread(self.thread_patch)
        self.generate_patches.finished.connect(self.thread_patch.quit)
        self.thread_patch.started.connect(self.generate_patches.run)
        self.stop_gen_patch.clicked.connect(lambda: self.generate_patches.stop_call())

        self.create_dataset = TFRecordConverter()
        self.thread_dataset = QtCore.QThread()
        self.create_dataset.moveToThread(self.thread_dataset)
        self.create_dataset.finished.connect(self.thread_dataset.quit)
        self.thread_dataset.started.connect(self.create_dataset.run)
        # self.stop_gen_patch.clicked.connect(lambda: self.create_dataset.stop_call())

        self.train_model = Train()
        self.thread_train = QtCore.QThread()
        self.train_model.moveToThread(self.thread_train)
        self.train_model.finished.connect(self.thread_train.quit)
        self.thread_train.started.connect(self.train_model.train)

    def update_test_progress(self, i):
        self.test_progress.setValue(i)

    ## Functions for reading files, setting PixMaps
    def get_file(self):
        fname = QFileDialog.getOpenFileName(self.menuWindow, "Open File", os.getcwd(), "(*.tif *.jp2 *.ndpi"
                                                                                       " *.vms *.vmu *.svs"
                                                                                       " *.tiff *.mrxs *.scn"
                                                                                       "*.svslide *.bif)")
        if fname[0]:
            self.ImageView = DisplayImage(fname[0],self.orig_image.height(), self.orig_image.width())
            self.if_image = True
            orim, curim, nlevel = self.ImageView.read_first()
            self.setImage(curim)
            self.updateInfo(orim)
            self.update_coordinates()
            self.zoomSlider.setMaximum(nlevel)
            self.zoomSlider.setValue(0)
            self.c_zoom_level = 0
            self.current_level.setText(str(self.ImageView.level))

    def get_file_overlay(self):
        print("Reached Overlay Callback")
        if self.if_image:
            if self.overlay_method.currentText()=="Segmentation Mask (by Pixel)":
                fname = QFileDialog.getOpenFileName(self.menuWindow, "Open File", os.getcwd(), "(*.tif *.png)")
                if fname[0]:
                    tim = self.ImageView.read_first_overlay(fname[0], method=self.overlay_method.currentText(),
                                                            states=self.overlay_states)
                    self.setImageOverlay(tim)
                    self.if_image_overlay += 1
                    self.check_segmentation.setEnabled(True)
                    self.check_segmentation.setChecked(True)
            elif self.overlay_method.currentText()=="Tumor Region":
                fname = QFileDialog.getOpenFileName(self.menuWindow, "Open File", os.getcwd(), "(*.mat)")
                if fname[0]:
                    tim = self.ImageView.read_first_overlay(fname[0], method=self.overlay_method.currentText(),
                                                            states=self.overlay_states)
                    self.setImageOverlay(tim)
                    self.if_image_overlay += 1
                    self.check_tumor_region.setEnabled(True)
                    self.check_tumor_region.setChecked(True)
            elif self.overlay_method.currentText()=="Heatmap":
                fname = QFileDialog.getExistingDirectory(self.menuWindow, "Choose Directory", os.getcwd(),
                                                         QFileDialog.ShowDirsOnly)
                if fname:
                    tim = self.ImageView.read_first_overlay(fname, method=self.overlay_method.currentText(),
                                                            states=self.overlay_states)
                    self.setImageOverlay(tim)
                    self.if_image_overlay += 1
                    self.check_heatmap.setEnabled(True)
                    self.check_heatmap.setChecked(True)
            elif self.overlay_method.currentText()=="Nuclei Position":
                fname = QFileDialog.getOpenFileName(self.menuWindow, "Open File", os.getcwd(), "(*.mat)")
                if fname[0]:
                    tim = self.ImageView.read_first_overlay(fname[0], method=self.overlay_method.currentText(),
                                                            states=self.overlay_states)
                    self.setImageOverlay(tim)
                    self.if_image_overlay += 1
                    self.check_nuclei.setEnabled(True)
                    self.check_nuclei.setChecked(True)

    def setImage(self, image):
        self.orig_image.setPixmap(QPixmap.fromImage(image))

    def setImageOverlay(self, image):
        if self.overlay_side_by_side.isChecked():
            self.overlay_image.setPixmap(QPixmap.fromImage(image))
        else:
            self.orig_image.setPixmap(QPixmap.fromImage(image))

    def updateInfo(self, image):
        self.info.setPixmap(QPixmap.fromImage(image))

    ## Panning Operation
    def mouse_orig_clear(self, event):
        self.if_mouse_pressed = False
        self.prev_mouse_pos = None

    def mouse_orig(self, event):
        if event.button()==QtCore.Qt.LeftButton:
            self.if_mouse_pressed = True
        if self.if_mouse_pressed:
            if not self.prev_mouse_pos:
                self.prev_mouse_pos = event.pos()
            else:
                self.pan_direction_ops(self.prev_mouse_pos - event.pos())
                self.prev_mouse_pos = event.pos()
        else:
            self.prev_mouse_pos = None

    def pan_direction_ops(self, value):
        if self.if_image:
            im, updated = self.ImageView.pan(value_x=value.x(), value_y=value.y())
            if updated:
                self.setImage(im)
                self.updateInfo(self.ImageView.get_info())
                self.update_coordinates()
                if self.if_image_overlay:
                    self.setImageOverlay(self.ImageView.update_overlay(method_update="down", states = self.overlay_states))

    ## Zooming Operations
    def wheel_zoom(self, event):
        # print("Wheel Event has been called", event, event.angleDelta())
        if event.angleDelta().y() > 0:
            self.c_zoom_level += 1
            self.zoomSlider.setValue(self.zoomSlider.value() + 1)
            self.zoom_in_ops()
        else:
            self.c_zoom_level -= 1
            self.zoomSlider.setValue(self.zoomSlider.value() - 1)
            self.zoom_out_ops()

    def updateBar(self):
        zdiff = self.c_zoom_level - self.zoomSlider.value() - 1
        if zdiff==0:
            return
        if zdiff==-1:
            self.c_zoom_level += 1
            self.zoom_in_ops()
        elif zdiff==1:
            self.c_zoom_level -= 1
            self.zoom_out_ops()

    def zoom_in_ops(self):
        if self.if_image:
            factor = 2
            self.setImage(self.ImageView.get_image_in(factor))
            self.updateInfo(self.ImageView.get_info())
            self.update_coordinates()
            if self.if_image_overlay:
                self.setImageOverlay(self.ImageView.update_overlay(method_update="zoom_in", states = self.overlay_states))
            self.current_level.setText(str(self.ImageView.level))

    def zoom_out_ops(self):
        if self.if_image:
            factor = 2
            self.setImage(self.ImageView.get_image_out(factor))
            self.updateInfo(self.ImageView.get_info())
            self.update_coordinates()
            if self.if_image_overlay:
                self.setImageOverlay(self.ImageView.update_overlay(method_update="zoom_out", states = self.overlay_states))
            self.current_level.setText(str(self.ImageView.level))

    ## Random seek using info
    def get_random_location(self, event):
        print(event.pos())
        if self.if_image:
            # print(self.info.size())
            self.setImage(self.ImageView.random_seek(event.pos().x(), event.pos().y(), self.info.size()))
            self.updateInfo(self.ImageView.get_info())
            self.update_coordinates()
            print("Random Seek in Overlay")
            if self.if_image_overlay:
                self.setImageOverlay(self.ImageView.update_overlay(method_update="init", states=self.overlay_states))

    ## Managing Overlays
    def select_overlays(self):
        snd = self.menuWindow.sender()
        print("Inside Selecting Overlays: ", snd.text())
        if snd.text()=="Segmentation":
            if self.if_image_overlay:
                self.overlay_states["Seg"] = not self.overlay_states["Seg"]
                self.setImageOverlay(self.ImageView.update_overlay(method_update="init", states=self.overlay_states,
                                                                   ov_no_update=not self.overlay_states["Seg"]))
        elif snd.text()=="Tumor Region":
            if self.if_image_overlay:
                self.overlay_states["Reg"] = not self.overlay_states["Reg"]
                self.setImageOverlay(self.ImageView.update_overlay(method_update="init", states=self.overlay_states,
                                                                   ov_no_update=not self.overlay_states["Reg"]))
        elif snd.text()=="HeatMap":
            if self.if_image_overlay:
                self.overlay_states["Heat"] = not self.overlay_states["Heat"]
                self.setImageOverlay(self.ImageView.update_overlay(method_update="init", states=self.overlay_states,
                                                                   ov_no_update=not self.overlay_states["Heat"]))
        elif snd.text()=="Nuclei Position":
            if self.if_image_overlay:
                self.overlay_states["Nuclei"] = not self.overlay_states["Nuclei"]
                self.setImageOverlay(self.ImageView.update_overlay(method_update="init", states=self.overlay_states,
                                                                   ov_no_update=not self.overlay_states["Nuclei"],
                                                                   class_states=self.overlay_group_states))
                if self.overlay_states["Nuclei"]:
                    self.overlay_group.setEnabled(True)
                    for i in range(self.ImageView.get_number_classes()):
                        self.overlay_group_dict[i].setEnabled(True)
                        self.overlay_group_dict[i].setText("Class " + str(i))
                        self.overlay_group_dict[i].setStyleSheet("color: rgb" + str(self.colors[i]))
                        self.overlay_group_dict[i].setChecked(True)
                    self.overlay_group.show()
                else:
                    self.overlay_group.hide()

    def select_class(self):
        print("Value of state changed of ", self.menuWindow.sender().objectName(), self.menuWindow.sender())
        snd = self.menuWindow.sender().objectName()
        self.overlay_group_states[int(snd.split('_')[1])] = not self.overlay_group_states[int(snd.split('_')[1])]
        print("From select class function: ", self.overlay_group_states)
        self.setImageOverlay(self.ImageView.update_overlay(method_update="init", states=self.overlay_states,
                                                           ov_no_update=not self.overlay_states["Nuclei"],
                                                           class_states=self.overlay_group_states))

    def overlay_state_changed(self):
        if self.overlay_side_by_side.isChecked():
            self.overlay_image.show()
        else:
            self.overlay_image.hide()

    def select_WSI(self):
        fname = QFileDialog.getOpenFileName(self.menuWindow, "Select Whole Slide Image", os.getcwd(), "(*.tif *.jp2 *.ndpi"
                                                                                       " *.vms *.vmu *.svs"
                                                                                       " *.tiff *.mrxs *.scn"
                                                                                       "*.svslide *.bif)")
        if fname[0]:
            print(fname[0])
            self.image_path.setEnabled(True)
            self.image_path.setText(fname[0])
            self.select_patch_size.setEnabled(True)
            self.select_wsi_level.setEnabled(True)
            nlevel = ImageClass(self.image_path.text()).level_count
            self.select_wsi_level.addItem("None")
            [self.select_wsi_level.addItem(str(i)) for i in range(nlevel)]

    def select_gen_WSI(self):
        fname = QFileDialog.getExistingDirectory(self.menuWindow, "Choose Directory", os.getcwd(),
                                                 QFileDialog.ShowDirsOnly)
        if fname:
            print(fname)
            self.gen_image_path.setEnabled(True)
            self.gen_image_path.setText(fname)
            self.select_gen_patch_size.setEnabled(True)
            self.select_gen_wsi_level.setEnabled(True)
            self.select_gen_wsi_level.addItem("None")
            [self.select_gen_wsi_level.addItem(str(i)) for i in range(12)]

    def select_dl_model(self):
        fname = QFileDialog.getOpenFileName(self.menuWindow, "Select DL Checkpoint", os.getcwd(), "*.ckpt-*")
        if fname[0]:
            print(fname[0])
            self.model_path.setEnabled(True)
            self.model_path.setText(fname[0])

    def select_mask_path(self):
        fname = QFileDialog.getOpenFileName(self.menuWindow, "Select Mask(if available)", os.getcwd(), "*.tif")
        if fname[0]:
            print(fname[0])
            self.mask_path.setEnabled(True)
            self.mask_path.setText(fname[0])
            self.select_mask_level.setEnabled(True)
            self.select_mask_level.addItem("None")
            [self.select_mask_level.addItem(str(i)) for i in range(12)]

    def check_test_fields(self):
        if self.image_path.isEnabled() and self.model_path.isEnabled() and self.mask_path.isEnabled() \
                and self.select_patch_size.text()!='' and self.select_wsi_level.currentIndex()!=0\
                and self.select_mask_level.currentIndex()!=0:
            return True
        return False

    def start_testing(self):
        if self.check_test_fields():
            Config.WSI_PATH = self.image_path.text()
            Config.CHECKPOINT_PATH = self.model_path.text()
            Config.MASK_PATH = self.mask_path.text()
            Config.PATCH_SIZE = int(self.select_patch_size.text())
            Config.LEVEL_FETCH = self.select_wsi_level.currentIndex()-1
            Config.LEVEL_UPGRADE = self.select_mask_level.currentIndex() - 1 - Config.LEVEL_FETCH
            self.thread_test.start()

    def check_patch_fields(self):
        if self.gen_image_path.isEnabled() and self.select_gen_patch_size.text()!=''\
                and self.select_gen_wsi_level.currentIndex()!=0:
            return True
        return False

    def start_generating_patch(self):
        # if self.check_patch_fields():
        #     PatchConfig.WSI_FOLDER_PATH = self.gen_image_path.text()
        #     PatchConfig.LEVEL_FETCH = self.select_gen_wsi_level.currentIndex() - 1
        #     PatchConfig.PATCH_SIZE = int(self.select_gen_patch_size.text())
        self.thread_patch.start()

    def start_creating_dataset(self):
        self.thread_dataset.start()

    def start_training(self):
        self.thread_train.start()

    def update_coordinates(self):
        w, h = self.ImageView.get_current_coordinates()
        self.curX.setText(str(w))
        self.curY.setText(str(h))