# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'mainwindow.ui'
#
# Created by: PyQt5 UI code generator 5.8.2
#
# WARNING! All changes made in this file will be lost!

from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtWidgets import QFileDialog
from PyQt5.QtGui import QImage, QPixmap
from image_ops import DisplayImage
import os
os.environ['PYQTGRAPH_QT_LIB'] = 'PyQt5'

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
        self.verticalLayout_2 = QtWidgets.QVBoxLayout()
        self.verticalLayout_2.setContentsMargins(11, 11, 11, 11)
        self.verticalLayout_2.setSpacing(6)
        self.verticalLayout_2.setObjectName("verticalLayout_2")
        self.horizontalLayout = QtWidgets.QHBoxLayout()
        self.horizontalLayout.setContentsMargins(11, 11, 11, 11)
        self.horizontalLayout.setSpacing(6)
        self.horizontalLayout.setObjectName("horizontalLayout")
        self.zoom_in = QtWidgets.QPushButton(self.vis)
        self.zoom_in.setObjectName("zoom_in")
        self.horizontalLayout.addWidget(self.zoom_in)
        self.zoom_out = QtWidgets.QPushButton(self.vis)
        self.zoom_out.setObjectName("zoom_out")
        self.horizontalLayout.addWidget(self.zoom_out)
        self.verticalLayout_2.addLayout(self.horizontalLayout)
        self.verticalLayout_4.addLayout(self.verticalLayout_2)
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
        self.gridLayout = QtWidgets.QGridLayout()
        self.gridLayout.setContentsMargins(5, 5, 5, 5)
        self.gridLayout.setSpacing(6)
        self.gridLayout.setObjectName("gridLayout")
        self.current_level_label = QtWidgets.QLabel(self.vis)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.current_level_label.sizePolicy().hasHeightForWidth())
        self.current_level_label.setSizePolicy(sizePolicy)
        self.current_level_label.setObjectName("current_level_label")
        self.gridLayout.addWidget(self.current_level_label, 0, 0, 1, 1)
        self.current_level = QtWidgets.QLineEdit(self.vis)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.current_level.sizePolicy().hasHeightForWidth())
        self.current_level.setSizePolicy(sizePolicy)
        self.current_level.setReadOnly(True)
        self.current_level.setObjectName("current_level")
        self.gridLayout.addWidget(self.current_level, 0, 1, 1, 1)
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

        self.intialize_signals_slots()
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
        self.zoom_in.setText(_translate("MainWindow", "ZoomIn"))
        self.zoom_out.setText(_translate("MainWindow", "ZoomOut"))
        self.current_level_label.setText(_translate("MainWindow", "Current Level"))
        self.current_level.setPlaceholderText(_translate("MainWindow", "NA"))
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
        self.tabs.setTabText(self.tabs.indexOf(self.training), _translate("MainWindow", "Training"))
        self.menuWindow.setTitle(_translate("MainWindow", "Window"))

    def intialize_signals_slots(self):
        # Bind all the signal and slots here
        self.if_image = False
        self.if_image_overlay = False
        self.c_zoom_level = 0
        self.prev_mouse_pos = None
        self.if_mouse_pressed = False
        self.load_image.clicked.connect(self.get_file)
        self.load_overlay.clicked.connect(self.get_file_overlay)
        self.zoom_in.clicked.connect(self.zoom_in_ops)
        self.zoom_out.clicked.connect(self.zoom_out_ops)
        self.zoom_in.clicked.connect(self.updateBar)
        self.zoom_out.clicked.connect(self.updateBar)
        self.zoomSlider.valueChanged.connect(self.updateBar_2)
        self.orig_image.mousePressEvent = self.mouse_orig
        self.orig_image.mouseReleaseEvent = self.mouse_orig_clear
        self.orig_image.mouseMoveEvent = self.mouse_orig
        self.orig_image.wheelEvent = self.wheel_zoom
        self.info.mousePressEvent = self.get_random_location
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

    def select_overlays(self):
        snd = self.menuWindow.sender()
        print("Inside Selecting Overlays: ", snd.text())
        if snd.text()=="Segmentation":
            if self.if_image_overlay:
                self.overlay_states["Seg"] = not self.overlay_states["Seg"]
                self.setImageOverlay(self.ImageView.update_overlay(method_update="init", states=self.overlay_states))
        elif snd.text()=="Tumor Region":
            if self.if_image_overlay:
                self.overlay_states["Reg"] = not self.overlay_states["Reg"]
                self.setImageOverlay(self.ImageView.update_overlay(method_update="init", states=self.overlay_states))
        elif snd.text()=="HeatMap":
            if self.if_image_overlay:
                self.overlay_states["Heat"] = not self.overlay_states["Heat"]
                self.setImageOverlay(self.ImageView.update_overlay(method_update="init", states=self.overlay_states))

    def overlay_state_changed(self):
        if self.overlay_side_by_side.isChecked():
            self.overlay_image.show()
        else:
            self.overlay_image.hide()

    def wheel_zoom(self, event):
        print("Wheel Event has been called", event, event.angleDelta())
        if event.angleDelta().y() > 0:
            print("should be zooming in")
            self.c_zoom_level += 1
            self.zoomSlider.setValue(self.zoomSlider.value() + 1)
            self.zoom_in_ops()
        else:
            print("should be zooming out")
            self.c_zoom_level -= 1
            self.zoomSlider.setValue(self.zoomSlider.value() - 1)
            self.zoom_out_ops()

    def mouse_orig_clear(self, event):
        self.if_mouse_pressed = False
        self.prev_mouse_pos = None

    def mouse_orig(self, event):
        # if event.button()==QtCore.Qt.NoButton:
        #     print("No Button")
        if event.button()==QtCore.Qt.LeftButton:
            print(event, event.pos())
            self.if_mouse_pressed = True
        if self.if_mouse_pressed:
            if not self.prev_mouse_pos:
                self.prev_mouse_pos = event.pos()
            else:
                self.pan_direction_ops(self.prev_mouse_pos - event.pos())
                self.prev_mouse_pos = event.pos()
        else:
            self.prev_mouse_pos = None

    def get_file(self):
        fname = QFileDialog.getOpenFileName(self.menuWindow, "Open File", "C:\\Users\\abhinav\\Desktop\\Tissue_GUI\\data", "*.tif")
        if fname[0]:
            self.ImageView = DisplayImage(fname[0],self.orig_image.height(), self.orig_image.width())
            self.scale = 1.
            self.if_image = True
            orim, curim, nlevel = self.ImageView.read_first()
            self.setImage(curim)
            self.updateInfo(orim)
            self.zoomSlider.setMaximum(nlevel)
            self.c_zoom_level = 0
            self.current_level.setText(str(self.ImageView.level))

    def updateInfo(self, image):
        orimap = QPixmap.fromImage(image)
        self.info.setPixmap(orimap)

    def get_file_overlay(self):
        print("Reached Callback")
        if self.if_image:
            if self.overlay_method.currentText()=="Segmentation Mask (by Pixel)":
                fname = QFileDialog.getOpenFileName(self.menuWindow, "Open File", "C:\\Users\\abhinav\\Desktop\\Tissue_GUI\\data",
                                                    "(*.tif *.png)")
                if fname[0]:
                    tim = self.ImageView.read_first_overlay(fname[0], method=self.overlay_method.currentText(),
                                                            states=self.overlay_states)
                    self.setImageOverlay(tim)
                    self.if_image_overlay = True
                    self.check_segmentation.setEnabled(True)
                    self.check_segmentation.setChecked(True)
            elif self.overlay_method.currentText()=="Tumor Region":
                fname = QFileDialog.getOpenFileName(self.menuWindow, "Open File",
                                                    "C:\\Users\\abhinav\\Desktop\\Tissue_GUI\\data", "(*.mat)")
                if fname[0]:
                    tim = self.ImageView.read_first_overlay(fname[0], method=self.overlay_method.currentText(),
                                                            states=self.overlay_states)
                    self.setImageOverlay(tim)
                    self.if_image_overlay = True
                    self.check_tumor_region.setEnabled(True)
                    self.check_tumor_region.setChecked(True)
            elif self.overlay_method.currentText()=="Heatmap":
                # fname = QFileDialog.getOpenFileName(self.menuWindow, "Open File",
                #                                     "C:\\Users\\abhinav\\Desktop\\Tissue_GUI\\data", "(*.png)")
                fname = QFileDialog.getExistingDirectory(self.menuWindow, "Choose Directory",
                                                    "C:\\Users\\abhinav\\Desktop\\Tissue_GUI\\data", QFileDialog.ShowDirsOnly)
                if fname:
                    tim = self.ImageView.read_first_overlay(fname, method=self.overlay_method.currentText(),
                                                            states=self.overlay_states)
                    self.setImageOverlay(tim)
                    self.if_image_overlay = True
                    self.check_heatmap.setEnabled(True)
                    self.check_heatmap.setChecked(True)

    def pan_direction_ops(self, value):
        if self.if_image:
            im, updated = self.ImageView.pan(direction='mouse', value_x=value.x(), value_y=value.y())
            self.setImage(im)
            self.updateInfo(self.ImageView.get_info())
            # if self.if_image_overlay and updated:
            if self.if_image_overlay:
                self.setImageOverlay(self.ImageView.update_overlay(method_update="down", states = self.overlay_states))

    def setImage(self, image):
        # self.orig_image.setPixmap(self.orimap.scaled(self.orig_image.size(), QtCore.Qt.KeepAspectRatio))
        self.orig_image.setPixmap(QPixmap.fromImage(image))

    def setImageOverlay(self, image):
        print("OverLay Image is being set")
        if self.overlay_side_by_side.isChecked():
            self.overlay_image.setPixmap(QPixmap.fromImage(image))
        else:
            self.orig_image.setPixmap(QPixmap.fromImage(image))
        print("OverLay Image is set")

    def updateBar(self):
        snd = self.menuWindow.sender()
        if snd.text()=="ZoomIn":
            self.c_zoom_level += 1
            self.zoomSlider.setValue(self.zoomSlider.value() + 1)
        elif snd.text()=="ZoomOut":
            self.c_zoom_level -= 1
            self.zoomSlider.setValue(self.zoomSlider.value() - 1)
        else:
            print("SlideBar Value has been changed")

    def updateBar_2(self):
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
            self.zoom_in.setEnabled(False)
            self.zoom_out.setEnabled(False)
            factor = 2
            self.setImage(self.ImageView.get_image_in(factor))
            self.updateInfo(self.ImageView.get_info())
            print("Started Zooming into Overlay")
            if self.if_image_overlay:
                self.setImageOverlay(self.ImageView.update_overlay(method_update="zoom_in", states = self.overlay_states))
            self.current_level.setText(str(self.ImageView.level))
            self.zoom_out.setEnabled(True)
            self.zoom_in.setEnabled(True)

    def zoom_out_ops(self):
        if self.if_image:
            print("Inside Zoom Out Ops")
            self.zoom_out.setEnabled(False)
            self.zoom_in.setEnabled(False)
            factor = 2
            self.setImage(self.ImageView.get_image_out(factor))
            self.updateInfo(self.ImageView.get_info())
            if self.if_image_overlay:
                self.setImageOverlay(self.ImageView.update_overlay(method_update="zoom_out", states = self.overlay_states))
            self.current_level.setText(str(self.ImageView.level))
            self.zoom_in.setEnabled(True)
            self.zoom_out.setEnabled(True)

    def get_random_location(self, event):
        print(event.pos())
        if self.if_image:
            self.zoom_in.setEnabled(False)
            self.zoom_out.setEnabled(False)
            print(self.info.size())
            self.setImage(self.ImageView.random_seek(event.pos().x(), event.pos().y(), self.info.size()))
            self.updateInfo(self.ImageView.get_info())
            print("Random Seek in Overlay")
            if self.if_image_overlay:
                self.setImageOverlay(self.ImageView.update_overlay(method_update="init", states=self.overlay_states))
            self.current_level.setText(str(self.ImageView.level))
            self.zoom_out.setEnabled(True)
            self.zoom_in.setEnabled(True)
