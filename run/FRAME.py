from PySide2 import QtWidgets

import sys
sys.path.append('./src/')
sys.path.append('../src/')
import FrameGUI

app = QtWidgets.QApplication()
w = FrameGUI.MainWindow()
app.exec_()
