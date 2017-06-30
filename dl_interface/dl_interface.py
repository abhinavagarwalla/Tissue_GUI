# Copyright 2017 Abhinav Agarwalla. All Rights Reserved.
# Contact: agarwallaabhinav@gmail.com, abhinavagarwalla@iitkgp.ac.in
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

from PyQt5.QtCore import QObject, pyqtSignal, pyqtSlot

from dl_interface.model_test import Test


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