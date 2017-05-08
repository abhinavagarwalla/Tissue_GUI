# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'mainwindow.ui'
#
# Created by: PyQt5 UI code generator 5.8.2
#
# WARNING! All changes made in this file will be lost!

from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtWidgets import QFileDialog

class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(1080, 599)
        self.centralWidget = QtWidgets.QWidget(MainWindow)
        self.centralWidget.setObjectName("centralWidget")
        self.load_image = QtWidgets.QPushButton(self.centralWidget)
        self.load_image.setGeometry(QtCore.QRect(10, 20, 75, 23))
        self.load_image.setObjectName("load_image")
        self.save_image = QtWidgets.QPushButton(self.centralWidget)
        self.save_image.setGeometry(QtCore.QRect(10, 50, 75, 23))
        self.save_image.setObjectName("save_image")
        self.orig_image = QtWidgets.QLabel(self.centralWidget)
        self.orig_image.setEnabled(True)
        self.orig_image.setGeometry(QtCore.QRect(180, 80, 421, 421))
        self.orig_image.setObjectName("orig_image")
        self.overlay_image = QtWidgets.QLabel(self.centralWidget)
        self.overlay_image.setEnabled(True)
        self.overlay_image.setGeometry(QtCore.QRect(610, 80, 421, 421))
        self.overlay_image.setObjectName("overlay_image")
        MainWindow.setCentralWidget(self.centralWidget)
        self.menuBar = QtWidgets.QMenuBar(MainWindow)
        self.menuBar.setGeometry(QtCore.QRect(0, 0, 1080, 21))
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

        self.intiliaze_signals_slots()
        self.retranslateUi(MainWindow)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

    def intiliaze_signals_slots(self):
        # Bind all the signal and slots here
        self.load_image.clicked.connect(self.get_file)

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "MainWindow"))
        self.load_image.setText(_translate("MainWindow", "Load Image"))
        self.save_image.setText(_translate("MainWindow", "PushButton"))
        self.orig_image.setText(_translate("MainWindow", "Placeholder for Real Image"))
        self.overlay_image.setText(_translate("MainWindow", "Placeholder for OverLay Image"))
        self.menuWindow.setTitle(_translate("MainWindow", "Window"))

    def get_file(self):
        print("Reached Callback")
        options = QFileDialog.Options()
        options |= QFileDialog.DontUseNativeDialog
        fname = QFileDialog.getOpenFileName(self, 'Open File', 'C:\\Users', '*.tif', options=options)
        print(fname)

if __name__ == "__main__":
    import sys
    app = QtWidgets.QApplication(sys.argv)
    MainWindow = QtWidgets.QMainWindow()
    ui = Ui_MainWindow()
    ui.setupUi(MainWindow)
    MainWindow.show()
    sys.exit(app.exec_())