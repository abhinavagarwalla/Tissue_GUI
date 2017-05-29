from PyQt5.QtCore import QThread, QObject, pyqtSignal, pyqtSlot
import time
from model_test import test

class Worker(QObject):
    finished = pyqtSignal()
    intReady = pyqtSignal(int)


    @pyqtSlot()
    def procCounter(self): # A slot takes no params
        # for i in range(1, 100):
        #     time.sleep(1)
        #     self.intReady.emit(i)
        # # main()
        # self.finished.emit()
        test()