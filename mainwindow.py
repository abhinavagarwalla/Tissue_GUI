# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'mainwindow.ui'
#
# Created by: PyQt5 UI code generator 5.8.2
#
# WARNING! All changes made in this file will be lost!

from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtWidgets import QFileDialog
from PyQt5.QtGui import QImage, QPixmap
from image_ops import SlImage
import os
os.environ['PYQTGRAPH_QT_LIB'] = 'PyQt5'

class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(1306, 678)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Preferred)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(MainWindow.sizePolicy().hasHeightForWidth())
        MainWindow.setSizePolicy(sizePolicy)
        self.centralWidget = QtWidgets.QWidget(MainWindow)
        self.centralWidget.setObjectName("centralWidget")
        self.horizontalLayout_4 = QtWidgets.QHBoxLayout(self.centralWidget)
        self.horizontalLayout_4.setContentsMargins(11, 11, 11, 11)
        self.horizontalLayout_4.setSpacing(6)
        self.horizontalLayout_4.setObjectName("horizontalLayout_4")
        self.verticalLayout_2 = QtWidgets.QVBoxLayout()
        self.verticalLayout_2.setContentsMargins(5, 5, 5, 5)
        self.verticalLayout_2.setSpacing(6)
        self.verticalLayout_2.setObjectName("verticalLayout_2")
        self.verticalLayout = QtWidgets.QVBoxLayout()
        self.verticalLayout.setContentsMargins(5, 5, 5, 5)
        self.verticalLayout.setSpacing(6)
        self.verticalLayout.setObjectName("verticalLayout")
        self.load_image = QtWidgets.QPushButton(self.centralWidget)
        self.load_image.setObjectName("load_image")
        self.verticalLayout.addWidget(self.load_image)
        self.save_image = QtWidgets.QPushButton(self.centralWidget)
        self.save_image.setObjectName("save_image")
        self.verticalLayout.addWidget(self.save_image)
        self.info = QtWidgets.QLabel(self.centralWidget)
        self.info.setObjectName("info")
        self.verticalLayout.addWidget(self.info)
        self.verticalLayout_2.addLayout(self.verticalLayout)
        self.horizontalLayout = QtWidgets.QHBoxLayout()
        self.horizontalLayout.setContentsMargins(5, 5, 5, 5)
        self.horizontalLayout.setSpacing(6)
        self.horizontalLayout.setObjectName("horizontalLayout")
        self.zoom_in = QtWidgets.QPushButton(self.centralWidget)
        self.zoom_in.setObjectName("zoom_in")
        self.horizontalLayout.addWidget(self.zoom_in)
        self.zoom_out = QtWidgets.QPushButton(self.centralWidget)
        self.zoom_out.setObjectName("zoom_out")
        self.horizontalLayout.addWidget(self.zoom_out)
        self.verticalLayout_2.addLayout(self.horizontalLayout)
        self.gridLayout = QtWidgets.QGridLayout()
        self.gridLayout.setContentsMargins(5, 5, 5, 5)
        self.gridLayout.setSpacing(6)
        self.gridLayout.setObjectName("gridLayout")
        self.current_level_label = QtWidgets.QLabel(self.centralWidget)
        self.current_level_label.setObjectName("current_level_label")
        self.gridLayout.addWidget(self.current_level_label, 3, 0, 1, 1)
        self.current_level = QtWidgets.QLineEdit(self.centralWidget)
        self.current_level.setReadOnly(True)
        self.current_level.setObjectName("current_level")
        self.gridLayout.addWidget(self.current_level, 3, 1, 1, 1)
        self.pan_right = QtWidgets.QPushButton(self.centralWidget)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Preferred)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.pan_right.sizePolicy().hasHeightForWidth())
        self.pan_right.setSizePolicy(sizePolicy)
        self.pan_right.setObjectName("pan_right")
        self.gridLayout.addWidget(self.pan_right, 0, 1, 1, 1)
        self.spinBox = QtWidgets.QSpinBox(self.centralWidget)
        self.spinBox.setMinimum(1)
        self.spinBox.setSingleStep(5)
        self.spinBox.setProperty("value", 11)
        self.spinBox.setObjectName("spinBox")
        self.gridLayout.addWidget(self.spinBox, 2, 1, 1, 1)
        self.pan_up = QtWidgets.QPushButton(self.centralWidget)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Preferred)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.pan_up.sizePolicy().hasHeightForWidth())
        self.pan_up.setSizePolicy(sizePolicy)
        self.pan_up.setObjectName("pan_up")
        self.gridLayout.addWidget(self.pan_up, 1, 0, 1, 1)
        self.pan_left = QtWidgets.QPushButton(self.centralWidget)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Preferred)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.pan_left.sizePolicy().hasHeightForWidth())
        self.pan_left.setSizePolicy(sizePolicy)
        self.pan_left.setObjectName("pan_left")
        self.gridLayout.addWidget(self.pan_left, 0, 0, 1, 1)
        self.pan_down = QtWidgets.QPushButton(self.centralWidget)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Preferred)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.pan_down.sizePolicy().hasHeightForWidth())
        self.pan_down.setSizePolicy(sizePolicy)
        self.pan_down.setObjectName("pan_down")
        self.gridLayout.addWidget(self.pan_down, 1, 1, 1, 1)
        self.pan_step = QtWidgets.QLabel(self.centralWidget)
        self.pan_step.setObjectName("pan_step")
        self.gridLayout.addWidget(self.pan_step, 2, 0, 1, 1)
        self.verticalLayout_2.addLayout(self.gridLayout)
        self.horizontalLayout_4.addLayout(self.verticalLayout_2)
        self.verticalLayout_3 = QtWidgets.QVBoxLayout()
        self.verticalLayout_3.setContentsMargins(5, 5, 5, 5)
        self.verticalLayout_3.setSpacing(6)
        self.verticalLayout_3.setObjectName("verticalLayout_3")
        self.horizontalLayout_3 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_3.setContentsMargins(5, 5, 5, 5)
        self.horizontalLayout_3.setSpacing(6)
        self.horizontalLayout_3.setObjectName("horizontalLayout_3")
        self.overlay_method = QtWidgets.QComboBox(self.centralWidget)
        self.overlay_method.setObjectName("overlay_method")
        self.overlay_method.addItem("")
        self.overlay_method.addItem("")
        self.overlay_method.addItem("")
        self.overlay_method.addItem("")
        self.horizontalLayout_3.addWidget(self.overlay_method)
        self.load_overlay = QtWidgets.QPushButton(self.centralWidget)
        self.load_overlay.setObjectName("load_overlay")
        self.horizontalLayout_3.addWidget(self.load_overlay)
        self.checkBox = QtWidgets.QCheckBox(self.centralWidget)
        self.checkBox.setObjectName("checkBox")
        self.horizontalLayout_3.addWidget(self.checkBox)
        self.checkBox_2 = QtWidgets.QCheckBox(self.centralWidget)
        self.checkBox_2.setObjectName("checkBox_2")
        self.horizontalLayout_3.addWidget(self.checkBox_2)
        self.verticalLayout_3.addLayout(self.horizontalLayout_3)
        self.horizontalLayout_2 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_2.setContentsMargins(5, 5, 5, 5)
        self.horizontalLayout_2.setSpacing(6)
        self.horizontalLayout_2.setObjectName("horizontalLayout_2")
        self.orig_image = QtWidgets.QLabel(self.centralWidget)
        self.orig_image.setEnabled(True)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.orig_image.sizePolicy().hasHeightForWidth())
        self.orig_image.setSizePolicy(sizePolicy)
        self.orig_image.setMinimumSize(QtCore.QSize(512, 512))
        self.orig_image.setObjectName("orig_image")
        self.horizontalLayout_2.addWidget(self.orig_image)
        self.overlay_image = QtWidgets.QLabel(self.centralWidget)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.overlay_image.sizePolicy().hasHeightForWidth())
        self.overlay_image.setSizePolicy(sizePolicy)
        self.overlay_image.setMinimumSize(QtCore.QSize(512, 512))
        self.overlay_image.setMouseTracking(True)
        self.overlay_image.setToolTipDuration(10)
        self.overlay_image.setObjectName("overlay_image")
        self.horizontalLayout_2.addWidget(self.overlay_image)
        self.verticalLayout_3.addLayout(self.horizontalLayout_2)
        self.horizontalLayout_4.addLayout(self.verticalLayout_3)
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

        self.intialize_signals_slots()
        self.retranslateUi(MainWindow)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

    def intialize_signals_slots(self):
        # Bind all the signal and slots here
        self.if_image = False
        self.if_image_overlay = False
        self.load_image.clicked.connect(self.get_file)
        self.load_overlay.clicked.connect(self.get_file_overlay)
        self.zoom_in.clicked.connect(self.zoom_in_ops)
        self.zoom_out.clicked.connect(self.zoom_out_ops)
        self.pan_left.clicked.connect(self.pan_left_ops)
        self.pan_right.clicked.connect(self.pan_right_ops)
        self.pan_up.clicked.connect(self.pan_up_ops)
        self.pan_down.clicked.connect(self.pan_down_ops)
        # self.overlay_image.setToolTip("Inside the Overlay")
        # self.orig_image.setBackgroundRole(QtGui.QPalette.Dark)
        # self.orig_image.setScaledContents(True)
        # self.scrollArea.setWidgetResizable(True)

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "MainWindow"))
        self.load_image.setText(_translate("MainWindow", "Load Image"))
        self.save_image.setText(_translate("MainWindow", "PushButton"))
        self.info.setText(_translate("MainWindow", "TextLabel"))
        self.zoom_in.setText(_translate("MainWindow", "ZoomIn"))
        self.zoom_out.setText(_translate("MainWindow", "ZoomOut"))
        self.current_level_label.setText(_translate("MainWindow", "Current Level"))
        self.current_level.setPlaceholderText(_translate("MainWindow", "NA"))
        self.pan_right.setText(_translate("MainWindow", "Right"))
        self.pan_up.setText(_translate("MainWindow", "Up"))
        self.pan_left.setText(_translate("MainWindow", "Left"))
        self.pan_down.setText(_translate("MainWindow", "Down"))
        self.pan_step.setText(_translate("MainWindow", "Pan Step (%)"))
        self.overlay_method.setItemText(0, _translate("MainWindow", "Segmentation Mask (by Pixel)"))
        self.overlay_method.setItemText(1, _translate("MainWindow", "Segmentation Mask (By Patch)"))
        self.overlay_method.setItemText(2, _translate("MainWindow", "Nuclei Position"))
        self.overlay_method.setItemText(3, _translate("MainWindow", "Tumor Region"))
        self.load_overlay.setText(_translate("MainWindow", "Load Overlay"))
        self.checkBox.setText(_translate("MainWindow", "CheckBox"))
        self.checkBox_2.setText(_translate("MainWindow", "CheckBox"))
        self.orig_image.setText(_translate("MainWindow", "Original Image"))
        self.overlay_image.setToolTip(_translate("MainWindow", "\"Where are you?\""))
        self.overlay_image.setText(_translate("MainWindow", "Overlay Image"))
        self.menuWindow.setTitle(_translate("MainWindow", "Window"))

    def get_file(self):
        fname = QFileDialog.getOpenFileName(self.menuWindow, "Open File", "C:\\Users\\abhinav\\Desktop\\Tissue_GUI\\data", "*.tif")
        if fname[0]:
            self.ImageView = SlImage(fname[0],self.orig_image.height(), self.orig_image.width())
            self.scale = 1.
            self.if_image = True
            orim, curim = self.ImageView.read_first()
            self.setImage(curim)
            self.orimap = QPixmap.fromImage(orim)
            self.info.setPixmap(self.orimap)
            self.current_level.setText(str(self.ImageView.level))

    def get_file_overlay(self):
        print("Reached Callback")
        if self.if_image:
            fname = QFileDialog.getOpenFileName(self.menuWindow, "Open File", "C:\\Users\\abhinav\\Desktop\\Tissue_GUI\\data",
                                                "(*.tif *.png)")
            if fname[0]:
                tim = self.ImageView.read_first_overlay(fname[0], method=self.overlay_method.currentIndex())
                self.setImageOverlay(tim)
                self.if_image_overlay = True

    def pan_left_ops(self):
        if self.if_image:
            self.pan_left.setEnabled(False)
            im, updated = self.ImageView.pan(direction='left', step=self.spinBox.value() / 100.)
            self.setImage(im)
            if self.if_image_overlay and updated:
                self.setImageOverlay(self.ImageView.update_overlay(method_update="left", step=self.spinBox.value()/100.))
            self.pan_left.setEnabled(True)

    def pan_right_ops(self):
        if self.if_image:
            self.pan_right.setEnabled(False)
            im, updated = self.ImageView.pan(direction='right', step=self.spinBox.value() / 100.)
            self.setImage(im)
            if self.if_image_overlay and updated:
                self.setImageOverlay(self.ImageView.update_overlay(method_update="right", step=self.spinBox.value()/100.))
            self.pan_right.setEnabled(True)

    def pan_up_ops(self):
        if self.if_image:
            self.pan_up.setEnabled(False)
            im, updated = self.ImageView.pan(direction='up', step=self.spinBox.value() / 100.)
            self.setImage(im)
            if self.if_image_overlay and updated:
                self.setImageOverlay(self.ImageView.update_overlay(method_update="up", step=self.spinBox.value()/100.))
            self.pan_up.setEnabled(True)

    def pan_down_ops(self):
        if self.if_image:
            self.pan_down.setEnabled(False)
            im, updated = self.ImageView.pan(direction='down', step=self.spinBox.value()/100.)
            self.setImage(im)
            if self.if_image_overlay and updated:
                self.setImageOverlay(self.ImageView.update_overlay(method_update="down", step=self.spinBox.value()/100.))
            self.pan_down.setEnabled(True)

    def setImage(self, image):
        # self.orig_image.setPixmap(self.orimap.scaled(self.orig_image.size(), QtCore.Qt.KeepAspectRatio))
        self.orig_image.setPixmap(QPixmap.fromImage(image))

    def setImageOverlay(self, image):
        print("OverLay Image is being set")
        self.overlay_image.setPixmap(QPixmap.fromImage(image))
        print("OverLay Image is set")

    def zoom_in_ops(self):
        if self.if_image:
            self.zoom_in.setEnabled(False)
            factor = 2
            self.setImage(self.ImageView.get_image_in(factor))
            print("Started Zooming into Overlay")
            if self.if_image_overlay:
                self.setImageOverlay(self.ImageView.update_overlay(method_update="zoom_in"))
            self.current_level.setText(str(self.ImageView.level))
            self.zoom_in.setEnabled(True)
            # factor = 1.2
            # self.scale = factor*self.scale
            # self.orig_image.resize(self.scale*self.orig_image.pixmap().size())

    def zoom_out_ops(self):
        if self.if_image:
            print("Inside Zoom Out Ops")
            self.zoom_out.setEnabled(False)
            factor = 2
            self.setImage(self.ImageView.get_image_out(factor))
            if self.if_image_overlay:
                self.setImageOverlay(self.ImageView.update_overlay(method_update="zoom_out"))
            self.current_level.setText(str(self.ImageView.level))
            self.zoom_out.setEnabled(True)
