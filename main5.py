# -*- coding: utf-8 -*-
"""
main5.py
- Tab 7 mirrors "Selected CBCT: <folder>/d_side_volume/side_volume.npy"
- Embeds a Napari viewer and loads the CBCT volume as a 3D rendering
- 3D controls: rendering mode, 2D/3D toggle, reset view
"""
from PyQt5 import QtCore, QtGui, QtWidgets
import subprocess, os, sys, importlib.util
import numpy as np

def _diagnose_env(parent=None, title="Python Env"):
    spec = importlib.util.find_spec("napari")
    QtWidgets.QMessageBox.information(
        parent, title,
        f"PYTHON: {sys.executable}\n"
        f"napari found: {spec is not None}\n"
        f"napari spec: {spec}\n"
    )

class Ui_Dialog(object):
    def setupUi(self, Dialog):
        Dialog.setObjectName("Dialog")
        Dialog.resize(1100, 650)

        self.tabWidget = QtWidgets.QTabWidget(Dialog)
        self.tabWidget.setGeometry(QtCore.QRect(20, 20, 1060, 600))
        self.tabWidget.setObjectName("tabWidget")

        # ----------------------------
        # TAB 1: Load Patient Folder
        # ----------------------------
        self.tab = QtWidgets.QWidget()
        self.tab.setObjectName("tab")

        self.tabWidget_2 = QtWidgets.QTabWidget(self.tab)
        self.tabWidget_2.setGeometry(QtCore.QRect(10, 20, 600, 360))
        self.tabWidget_2.setObjectName("tabWidget_2")

        # subtab: Convert to DICOMDIR
        self.tab_3 = QtWidgets.QWidget()
        self.tab_3.setObjectName("tab_3")
        self.pushButton = QtWidgets.QPushButton(self.tab_3)
        self.pushButton.setGeometry(QtCore.QRect(300, 20, 200, 32))
        self.pushButton.setObjectName("pushButton")
        self.pushButton_2 = QtWidgets.QPushButton(self.tab_3)
        self.pushButton_2.setGeometry(QtCore.QRect(20, 70, 120, 28))
        self.pushButton_2.setObjectName("pushButton_2")
        self.lineEdit = QtWidgets.QLineEdit(self.tab_3)
        self.lineEdit.setGeometry(QtCore.QRect(20, 20, 260, 32))
        self.lineEdit.setObjectName("lineEdit")
        self.tabWidget_2.addTab(self.tab_3, "")

        # subtab: Load DICOMDIR Files
        self.tab_4 = QtWidgets.QWidget()
        self.tab_4.setObjectName("tab_4")
        self.lineEdit_2 = QtWidgets.QLineEdit(self.tab_4)
        self.lineEdit_2.setGeometry(QtCore.QRect(20, 30, 260, 32))
        self.lineEdit_2.setObjectName("lineEdit_2")
        self.pushButton_3 = QtWidgets.QPushButton(self.tab_4)
        self.pushButton_3.setGeometry(QtCore.QRect(300, 30, 200, 32))
        self.pushButton_3.setObjectName("pushButton_3")
        self.tabWidget_2.addTab(self.tab_4, "")

        self.tabWidget.addTab(self.tab, "")

        # ----------------------------
        # TAB 2: Annotate The Jaw Curve
        # ----------------------------
        self.tab_2 = QtWidgets.QWidget()
        self.tab_2.setObjectName("tab_2")
        self.pushButton_4 = QtWidgets.QPushButton(self.tab_2)
        self.pushButton_4.setGeometry(QtCore.QRect(10, 40, 250, 40))
        self.pushButton_4.setObjectName("pushButton_4")
        self.tabWidget.addTab(self.tab_2, "")

        # ----------------------------
        # TAB 3: Set Dental Implant Angle (Napari 3D)
        # ----------------------------
        self.tab_7 = QtWidgets.QWidget()
        self.tab_7.setObjectName("tab_7")

        # Layout for tab_7
        layout7 = QtWidgets.QVBoxLayout(self.tab_7)

        # Path label
        self.lblMirror = QtWidgets.QLabel(self.tab_7)
        f = QtGui.QFont(); f.setPointSize(10)
        self.lblMirror.setFont(f)
        self.lblMirror.setText("Selected CBCT: ")
        self.lblMirror.setWordWrap(True)

        # Top control bar
        controls = QtWidgets.QHBoxLayout()
        controls.setSpacing(12)

        self.chk3D = QtWidgets.QCheckBox("3D view")
        self.chk3D.setChecked(True)

        self.cmbRendering = QtWidgets.QComboBox()
        self.cmbRendering.addItems(["attenuated_mip", "mip", "translucent", "additive", "iso"])
        self.cmbRendering.setCurrentText("attenuated_mip")

        self.btnReset = QtWidgets.QPushButton("Reset view")

        controls.addWidget(QtWidgets.QLabel("Rendering:"))
        controls.addWidget(self.cmbRendering)
        controls.addSpacing(20)
        controls.addWidget(self.chk3D)
        controls.addSpacing(20)
        controls.addWidget(self.btnReset)
        controls.addStretch(1)

        # Napari viewer
        os.environ.setdefault("QT_API", "pyqt5")
        os.environ.setdefault("VISPY_USE_APP", "pyqt5")
        import napari
        self.viewer = napari.Viewer(show=False)
        self.viewerWidget = self.viewer.window._qt_viewer  # embed canvas+controls (no menubar)

        layout7.addWidget(self.lblMirror)
        layout7.addLayout(controls)
        layout7.addWidget(self.viewerWidget, stretch=1)

        self.tabWidget.addTab(self.tab_7, "")

        # ----------------------------
        # TAB 4: Dental Implant Recommendation
        # ----------------------------
        self.tab_5 = QtWidgets.QWidget()
        self.tab_5.setObjectName("tab_5")
        self.lineEdit_3 = QtWidgets.QLineEdit(self.tab_5)
        self.lineEdit_3.setGeometry(QtCore.QRect(30, 50, 260, 32))
        self.lineEdit_3.setObjectName("lineEdit_3")
        self.pushButton_5 = QtWidgets.QPushButton(self.tab_5)
        self.pushButton_5.setGeometry(QtCore.QRect(310, 50, 200, 32))
        self.pushButton_5.setObjectName("pushButton_5")
        self.pushButton_6 = QtWidgets.QPushButton(self.tab_5)
        self.pushButton_6.setGeometry(QtCore.QRect(30, 100, 260, 32))
        self.pushButton_6.setObjectName("pushButton_6")
        self.tabWidget.addTab(self.tab_5, "")

        # ----------------------------
        # TAB 5: Dental Implant Database
        # ----------------------------
        self.tab_6 = QtWidgets.QWidget()
        self.tab_6.setObjectName("tab_6")
        self.tableView = QtWidgets.QTableView(self.tab_6)
        self.tableView.setGeometry(QtCore.QRect(40, 160, 760, 230))
        self.tableView.setObjectName("tableView")
        self.label = QtWidgets.QLabel(self.tab_6); self.label.setGeometry(QtCore.QRect(50, 24, 120, 20)); self.label.setText("ID")
        self.label_2 = QtWidgets.QLabel(self.tab_6); self.label_2.setGeometry(QtCore.QRect(50, 64, 120, 20)); self.label_2.setText("Diameter (mm)")
        self.label_3 = QtWidgets.QLabel(self.tab_6); self.label_3.setGeometry(QtCore.QRect(50, 104, 120, 20)); self.label_3.setText("Height (mm)")
        self.tabWidget.addTab(self.tab_6, "")

        # ----------------------------
        # SIGNALS
        # ----------------------------
        self.pushButton.clicked.connect(self.browse_directory)
        self.pushButton_2.clicked.connect(self.run_convert)
        self.lineEdit.textChanged.connect(self._update_tab7_cbct)

        # 3D / rendering controls
        self.chk3D.toggled.connect(self._toggle_3d)
        self.cmbRendering.currentTextChanged.connect(self._set_rendering)
        self.btnReset.clicked.connect(self._reset_view)

        self.retranslateUi(Dialog)
        self.tabWidget.setCurrentIndex(0)
        self.tabWidget_2.setCurrentIndex(0)
        QtCore.QMetaObject.connectSlotsByName(Dialog)

    # ----------------------------
    # Actions / Logic (Tab 7)
    # ----------------------------
    def _update_tab7_cbct(self, txt):
        """Update label and load CBCT volume into Napari with 3D rendering."""
        if not txt:
            self.lblMirror.setText("Selected CBCT: ")
            self.viewer.layers.clear()
            return

        path = os.path.join(txt, "d_side_volume", "side_volume.npy")
        self.lblMirror.setText(f"Selected CBCT: {path}")

        # Clear old data
        self.viewer.layers.clear()

        # Load if exists
        if os.path.isfile(path):
            try:
                vol = np.load(path)  # Expecting (Z, Y, X) or (Y, X, Z)
                # ensure 3D array
                if vol.ndim == 3:
                    layer = self.viewer.add_image(
                        vol,
                        name="CBCT",
                        colormap="gray",
                        interpolation="linear",
                        blending="additive",
                        rendering=self.cmbRendering.currentText(),  # 3D renderer
                        rgb=False,
                    )
                    # Force 3D display if checked
                    self.viewer.dims.ndisplay = 3 if self.chk3D.isChecked() else 2

                    # Fit/center the camera for a better start
                    self.viewer.reset_view()
                    # Optional: tweak camera zoom (smaller value = zoomed out)
                    self.viewer.camera.zoom = 1.0
                else:
                    self.lblMirror.setText(f"Selected CBCT: {path}  (unsupported ndim={vol.ndim})")
            except Exception as e:
                self.lblMirror.setText(f"Error loading CBCT: {e}")
        # else: file missing → leave viewer empty

    def _toggle_3d(self, checked):
        self.viewer.dims.ndisplay = 3 if checked else 2

    def _set_rendering(self, mode):
        # apply to the first image layer if present
        for lyr in self.viewer.layers:
            if hasattr(lyr, "rendering"):
                lyr.rendering = mode

    def _reset_view(self):
        self.viewer.reset_view()
        # keep dims as chosen
        self.viewer.dims.ndisplay = 3 if self.chk3D.isChecked() else 2

    # ----------------------------
    # Other actions
    # ----------------------------
    def browse_directory(self):
        directory = QtWidgets.QFileDialog.getExistingDirectory(None, "Select Patient Folder", "")
        if directory:
            self.lineEdit.setText(directory)  # triggers _update_tab7_cbct

    def run_convert(self):
        directory = self.lineEdit.text().strip()
        if not directory or not os.path.isdir(directory):
            QtWidgets.QMessageBox.warning(None, "Error", "Please select a valid directory first.")
            return
        try:
            subprocess.run(
                ["dcmodify", "-nb", "-gin", "-i", "(0020,0010)=STUDY123", "-i", "(0020,0011)=1", "*."],
                cwd=directory, shell=True, check=True
            )
            subprocess.run(["dcmmkdir", "--recurse"], cwd=directory, shell=True, check=True)
            QtWidgets.QMessageBox.information(None, "Success", "Conversion finished successfully!")
        except subprocess.CalledProcessError as e:
            QtWidgets.QMessageBox.critical(None, "Error", f"Command failed:\n{e}")

    def open_ian_tool(self):
        exe_path = r"C:\Users\user\Documents\[Exe-Only] IAN Annotation Tools Work\exe.win-amd64-3.7\IAN Annotation Tool.exe"
        if os.path.isfile(exe_path):
            subprocess.Popen([exe_path], cwd=os.path.dirname(exe_path))
        else:
            QtWidgets.QMessageBox.warning(None, "Not found", f"Executable not found:\n{exe_path}")

    def show_recommendation(self):
        try:
            os.environ.setdefault("QT_API", "pyqt5")
            os.environ.setdefault("VISPY_USE_APP", "pyqt5")
            from recommendation import Ui_MainWindow
            win = QtWidgets.QMainWindow()
            ui = Ui_MainWindow()
            ui.setupUi(win)
            win.setWindowTitle("Recommendation + Napari")
            win.show()
        except Exception as e:
            QtWidgets.QMessageBox.critical(None, "Error", f"Failed to open recommendation window:\n{e}")

    def retranslateUi(self, Dialog):
        _ = QtCore.QCoreApplication.translate
        Dialog.setWindowTitle(_("Dialog", "Dental Implant Planning Suite"))
        self.pushButton.setText(_("Dialog", "Load Patient Folder"))
        self.pushButton_2.setText(_("Dialog", "Convert"))
        self.tabWidget_2.setTabText(self.tabWidget_2.indexOf(self.tab_3), _("Dialog", "Convert to DICOMDIR"))
        self.pushButton_3.setText(_("Dialog", "Load Patient Folder"))
        self.tabWidget_2.setTabText(self.tabWidget_2.indexOf(self.tab_4), _("Dialog", "Load DICOMDIR Files"))
        self.tabWidget.setTabText(self.tabWidget.indexOf(self.tab), _("Dialog", "1. Load Patient Folder"))
        self.pushButton_4.setText(_("Dialog", "Annotate The Jaw Curve"))
        self.tabWidget.setTabText(self.tabWidget.indexOf(self.tab_2), _("Dialog", "2. Annotate The Jaw Curve"))
        self.tabWidget.setTabText(self.tabWidget.indexOf(self.tab_7), _("Dialog", "3. Set Dental Implant Angle"))
        self.pushButton_5.setText(_("Dialog", "Load The Jaw Curve"))
        self.pushButton_6.setText(_("Dialog", "Get Dental Implant Recommendation"))
        self.tabWidget.setTabText(self.tabWidget.indexOf(self.tab_5), _("Dialog", "4. Dental Implant Recommendation"))
        self.tabWidget.setTabText(self.tabWidget.indexOf(self.tab_6), _("Dialog", "Dental Implant Database"))

# Entrypoint
if __name__ == "__main__":
    # Optional: choose software rendering if you hit OpenGL issues:
    # os.environ["QT_OPENGL"] = "software"
    app = QtWidgets.QApplication(sys.argv)
    Dialog = QtWidgets.QDialog()
    ui = Ui_Dialog()
    ui.setupUi(Dialog)
    Dialog.show()
    sys.exit(app.exec_())
