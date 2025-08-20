# -*- coding: utf-8 -*-
from PyQt5 import QtCore, QtGui, QtWidgets
import napari
import numpy as np

class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(1100, 600)

        # ---- central widget + horizontal splitter ----
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        MainWindow.setCentralWidget(self.centralwidget)

        root_layout = QtWidgets.QVBoxLayout(self.centralwidget)

        # title
        self.label = QtWidgets.QLabel("Suitable Dental Implants (Sorted by Volume)")
        font = QtGui.QFont(); font.setPointSize(9)
        self.label.setFont(font)
        root_layout.addWidget(self.label)

        # splitter to hold table (left) and napari (right)
        splitter = QtWidgets.QSplitter(QtCore.Qt.Horizontal)
        root_layout.addWidget(splitter, 1)  # stretch=1 so it fills remaining space

        # ---- left: your table ----
        self.tableView = QtWidgets.QTableView()
        self.tableView.setMinimumWidth(360)
        splitter.addWidget(self.tableView)

        # ---- right: napari viewer widget ----
        # create a napari viewer without showing a separate window
        self.viewer = napari.Viewer(show=False)

        # you can embed either the whole napari window or just the central viewer:
        # widget_to_embed = self.viewer.window._qt_window        # full Napari window (menus, docks)
        widget_to_embed = self.viewer.window._qt_viewer          # only the canvas/controls area

        splitter.addWidget(widget_to_embed)
        splitter.setSizes([400, 700])  # initial sizes (left,right)

        # status bar
        self.statusbar = QtWidgets.QStatusBar(MainWindow)
        MainWindow.setStatusBar(self.statusbar)

        # optional: load a file and set rendering config
        self._demo_load()

    def _demo_load(self):
        # Example: add an image layer with custom rendering
        img = np.random.random((128, 128, 64))  # fake 3D data
        layer = self.viewer.add_image(
            img,
            name="demo",
            colormap="magma",
            contrast_limits=(0, 1),
            rendering="attenuated_mip",  # 3D rendering
            interpolation="nearest",
        )
        # If you open from a path:
        # layer = self.viewer.open("/path/to/your/file.tif")[0]
        # layer.colormap = "viridis"; layer.blending = "additive"; layer.contrast_limits = (50, 200)

if __name__ == "__main__":
    import sys, os
    # Ensure napari uses PyQt5 (QtPy picks this up); usually not needed, but safe:
    os.environ.setdefault("QT_API", "pyqt5")

    app = QtWidgets.QApplication(sys.argv)
    MainWindow = QtWidgets.QMainWindow()
    ui = Ui_MainWindow()
    ui.setupUi(MainWindow)
    MainWindow.setWindowTitle("Recommendation + Napari")
    MainWindow.show()
    sys.exit(app.exec_())