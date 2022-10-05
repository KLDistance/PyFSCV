import sys, os
from PyQt6 import QtCore
from PyQt6.QtWidgets import QApplication
import func

if __name__ == '__main__':
    app = QApplication(sys.argv)
    myapp = func.MyApp()
    app.aboutToQuit.connect(myapp.on_window_exit)
    myapp.show()
    sys.exit(app.exec())
