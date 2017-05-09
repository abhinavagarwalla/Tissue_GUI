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
        MainWindow.resize(1306, 647)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Preferred)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(MainWindow.sizePolicy().hasHeightForWidth())
        MainWindow.setSizePolicy(sizePolicy)
        self.centralWidget = QtWidgets.QWidget(MainWindow)
        self.centralWidget.setObjectName("centralWidget")
        self.widget = QtWidgets.QWidget(self.centralWidget)
        self.widget.setGeometry(QtCore.QRect(30, 50, 151, 431))
        self.widget.setObjectName("widget")
        self.verticalLayout = QtWidgets.QVBoxLayout(self.widget)
        self.verticalLayout.setContentsMargins(11, 11, 11, 11)
        self.verticalLayout.setSpacing(6)
        self.verticalLayout.setObjectName("verticalLayout")
        self.load_image = QtWidgets.QPushButton(self.widget)
        self.load_image.setObjectName("load_image")
        self.verticalLayout.addWidget(self.load_image)
        self.save_image = QtWidgets.QPushButton(self.widget)
        self.save_image.setObjectName("save_image")
        self.verticalLayout.addWidget(self.save_image)
        self.info = QtWidgets.QLabel(self.widget)
        self.info.setObjectName("info")
        self.verticalLayout.addWidget(self.info)
        self.widget1 = QtWidgets.QWidget(self.centralWidget)
        self.widget1.setGeometry(QtCore.QRect(240, 50, 1011, 511))
        self.widget1.setObjectName("widget1")
        self.image_layout = QtWidgets.QHBoxLayout(self.widget1)
        self.image_layout.setSizeConstraint(QtWidgets.QLayout.SetDefaultConstraint)
        self.image_layout.setContentsMargins(5, 5, 5, 5)
        self.image_layout.setSpacing(6)
        self.image_layout.setObjectName("image_layout")
        self.orig_image = QtWidgets.QLabel(self.widget1)
        self.orig_image.setEnabled(True)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Preferred)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.orig_image.sizePolicy().hasHeightForWidth())
        self.orig_image.setSizePolicy(sizePolicy)
        self.orig_image.setObjectName("orig_image")
        self.image_layout.addWidget(self.orig_image)
        self.overlay_image = QtWidgets.QLabel(self.widget1)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Preferred)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.overlay_image.sizePolicy().hasHeightForWidth())
        self.overlay_image.setSizePolicy(sizePolicy)
        self.overlay_image.setObjectName("overlay_image")
        self.image_layout.addWidget(self.overlay_image)
        self.widget2 = QtWidgets.QWidget(self.centralWidget)
        self.widget2.setGeometry(QtCore.QRect(20, 510, 166, 33))
        self.widget2.setObjectName("widget2")
        self.horizontalLayout = QtWidgets.QHBoxLayout(self.widget2)
        self.horizontalLayout.setContentsMargins(5, 5, 5, 5)
        self.horizontalLayout.setSpacing(6)
        self.horizontalLayout.setObjectName("horizontalLayout")
        self.zoom_in = QtWidgets.QPushButton(self.widget2)
        self.zoom_in.setObjectName("zoom_in")
        self.horizontalLayout.addWidget(self.zoom_in)
        self.zoom_out = QtWidgets.QPushButton(self.widget2)
        self.zoom_out.setObjectName("zoom_out")
        self.horizontalLayout.addWidget(self.zoom_out)
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
        self.load_image.clicked.connect(self.get_file)
        self.zoom_in.clicked.connect(self.zoom_in_ops)
        self.zoom_out.clicked.connect(self.zoom_out_ops)

        # self.orig_image.setBackgroundRole(QtGui.QPalette.Dark)
        # self.orig_image.setScaledContents(True)
        # self.scrollArea.setWidgetResizable(True)

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "MainWindow"))
        self.zoom_in.setText(_translate("MainWindow", "ZoomIn"))
        self.zoom_out.setText(_translate("MainWindow", "ZoomOut"))
        self.orig_image.setText(_translate("MainWindow", "Original Image here"))
        self.overlay_image.setText(_translate("MainWindow", "Overlayed Image"))
        self.load_image.setText(_translate("MainWindow", "Load Image"))
        self.save_image.setText(_translate("MainWindow", "PushButton"))
        self.info.setText(_translate("MainWindow", "TextLabel"))
        self.menuWindow.setTitle(_translate("MainWindow", "Window"))

    def get_file(self):
        print("Reached Callback")
        fname = QFileDialog.getOpenFileName(self.menuWindow, "Open File", "C:\\Users\\abhinav\\Desktop\\Tissue_GUI\\data", "*.tif")
        self.ImageView = SlImage(fname[0],self.orig_image.height(), self.orig_image.width())
        self.scale = 1.
        self.if_image = True
        self.setImage(self.ImageView.read_first())

    def setImage(self, image):
        self.orimap = QPixmap.fromImage(image)
        # self.orig_image.setPixmap(self.orimap.scaled(self.orig_image.size(), QtCore.Qt.KeepAspectRatio))
        self.orig_image.setPixmap(self.orimap)

    def zoom_in_ops(self):
        if self.if_image:
            factor = 2
            self.setImage(self.ImageView.get_image(factor))
            # factor = 1.2
            # self.scale = factor*self.scale
            # self.orig_image.resize(self.scale*self.orig_image.pixmap().size())

    def zoom_out_ops(self):
        if self.if_image:
            factor = 0.8
            self.scale = factor * self.scale
            self.orig_image.resize(self.scale * self.orig_image.pixmap().size())