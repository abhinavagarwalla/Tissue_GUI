from PyQt5.QtCore import QObject, pyqtSignal, pyqtSlot

from interface.model_test import Test


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
        print("Starting Testing")
        tc = Test()
        tc.test()