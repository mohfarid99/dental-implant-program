# -*- coding: utf-8 -*-
"""
main7.py
- Tab 7 mirrors "Selected CBCT: <folder>/d_side_volume/side_volume.npy"
- Embeds a Napari viewer and loads the CBCT volume as a 3D rendering
- 3D controls: rendering mode, 2D/3D toggle, reset view
- Textbox + two buttons to set implant angle for LEFT/RIGHT teeth
- "Missing teeth slice" row with start/end boxes + 2 buttons
- NEW: LEFT/RIGHT actions rotate side_volume.npy, save rotated_expanded_zaxis.npy, and reload
- NEW: Reset View reloads side_volume.npy (original)
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

        # --- Implant angle input row (textbox + LEFT/RIGHT buttons + label) ---
        angleRow = QtWidgets.QHBoxLayout()
        angleRow.setSpacing(10)
        self.txtImplantAngle = QtWidgets.QLineEdit()
        self.txtImplantAngle.setPlaceholderText("Enter implant angle (deg)")
        self.txtImplantAngle.setFixedWidth(200)

        self.btnSetAngleLeft = QtWidgets.QPushButton("Set Angle (Left Teeth)")
        self.btnSetAngleRight = QtWidgets.QPushButton("Set Angle (Right Teeth)")

        self.lblImplantAngle = QtWidgets.QLabel("Current implant angle: —°")
        angleRow.addWidget(QtWidgets.QLabel("Implant Angle:"))
        angleRow.addWidget(self.txtImplantAngle)
        angleRow.addWidget(self.btnSetAngleLeft)
        angleRow.addWidget(self.btnSetAngleRight)
        angleRow.addSpacing(12)
        angleRow.addWidget(self.lblImplantAngle)
        angleRow.addStretch(1)

        # --- Missing teeth slice row (label + 2 boxes + 2 buttons) ---
        missingRow = QtWidgets.QHBoxLayout()
        missingRow.setSpacing(10)

        self.lblMissingSlice = QtWidgets.QLabel("Missing teeth slice:")
        self.txtMissingStart = QtWidgets.QLineEdit()
        self.txtMissingEnd = QtWidgets.QLineEdit()
        self.txtMissingStart.setPlaceholderText("start slice")
        self.txtMissingEnd.setPlaceholderText("end slice")
        self.txtMissingStart.setFixedWidth(100)
        self.txtMissingEnd.setFixedWidth(100)

        int_validator = QtGui.QIntValidator(0, 10**9, self.tab_7)
        self.txtMissingStart.setValidator(int_validator)
        self.txtMissingEnd.setValidator(int_validator)

        self.btnSetMissingSlice = QtWidgets.QPushButton("Set the missing teeth slice")
        self.btnGetMissingAuto = QtWidgets.QPushButton("Get missing teeth slice automatically")

        missingRow.addWidget(self.lblMissingSlice)
        missingRow.addWidget(self.txtMissingStart)
        missingRow.addWidget(self.txtMissingEnd)
        missingRow.addWidget(self.btnSetMissingSlice)
        missingRow.addWidget(self.btnGetMissingAuto)
        missingRow.addStretch(1)

        # Napari viewer
        os.environ.setdefault("QT_API", "pyqt5")
        os.environ.setdefault("VISPY_USE_APP", "pyqt5")
        import napari
        self.viewer = napari.Viewer(show=False)
        self.viewerWidget = self.viewer.window._qt_viewer  # embed canvas+controls (no menubar)

        # Existing extra button in Tab 7
        self.btnGetImplantRec = QtWidgets.QPushButton("Get Dental Implant Recommendation")
        self.btnGetImplantRec.setObjectName("btnGetImplantRec")
        self.btnGetImplantRec.setFixedHeight(36)

        layout7.addWidget(self.lblMirror)
        layout7.addLayout(controls)
        layout7.addLayout(angleRow)
        layout7.addLayout(missingRow)
        layout7.addWidget(self.viewerWidget, stretch=1)
        layout7.addWidget(self.btnGetImplantRec, alignment=QtCore.Qt.AlignRight)

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
        # STATE
        # ----------------------------
        self.implant_angle_deg = None              # last set (for compatibility)
        self.implant_angle_left_deg = None         # explicit left
        self.implant_angle_right_deg = None        # explicit right
        self.missing_slice_start = None            # manual start slice
        self.missing_slice_end = None              # manual end slice

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

        # Implant angle handlers (left/right)
        self.btnSetAngleLeft.clicked.connect(self._set_implant_angle_left)
        self.btnSetAngleRight.clicked.connect(self._set_implant_angle_right)
        self.txtImplantAngle.returnPressed.connect(self._set_implant_angle_left)

        # Missing teeth slice handlers (placeholder logic)
        self.btnSetMissingSlice.clicked.connect(self._set_missing_slice_manual)
        self.btnGetMissingAuto.clicked.connect(self._get_missing_slice_automatically)

        self.retranslateUi(Dialog)
        self.tabWidget.setCurrentIndex(0)
        self.tabWidget_2.setCurrentIndex(0)
        QtCore.QMetaObject.connectSlotsByName(Dialog)

    # ----------------------------
    # Helpers for CBCT paths & loading
    # ----------------------------
    def _cbct_dir(self):
        """Return folder containing side_volume.npy, or None."""
        root = self.lineEdit.text().strip()
        if not root:
            return None
        return os.path.join(root, "d_side_volume")

    def _path_side_volume(self):
        d = self._cbct_dir()
        if not d: return None
        return os.path.join(d, "side_volume.npy")

    def _path_rotated_volume(self):
        d = self._cbct_dir()
        if not d: return None
        return os.path.join(d, "rotated_expanded_zaxis.npy")

    def _load_into_viewer(self, vol, name="CBCT"):
        # Clear and load volume with current rendering mode
        self.viewer.layers.clear()
        self.viewer.add_image(
            vol, name=name, colormap="gray",
            interpolation="linear", blending="additive",
            rendering=self.cmbRendering.currentText(), rgb=False
        )
        self.viewer.dims.ndisplay = 3 if self.chk3D.isChecked() else 2
        self.viewer.reset_view()
        self.viewer.camera.zoom = 1.0

    # ----------------------------
    # Actions / Logic (Tab 7)
    # ----------------------------
    def _update_tab7_cbct(self, txt):
        """Update label and load CBCT volume into Napari with 3D rendering."""
        if not txt:
            self.lblMirror.setText("Selected CBCT: ")
            self.viewer.layers.clear()
            return

        side_path = self._path_side_volume()
        self.lblMirror.setText(f"Selected CBCT: {side_path if side_path else ''}")

        self.viewer.layers.clear()

        if side_path and os.path.isfile(side_path):
            try:
                vol = np.load(side_path)
                if vol.ndim == 3:
                    self._load_into_viewer(vol, name="CBCT")
                else:
                    self.lblMirror.setText(f"Selected CBCT: {side_path}  (unsupported ndim={vol.ndim})")
            except Exception as e:
                self.lblMirror.setText(f"Error loading CBCT: {e}")

    def _toggle_3d(self, checked):
        self.viewer.dims.ndisplay = 3 if checked else 2

    def _set_rendering(self, mode):
        # apply to the first image layer if present
        for lyr in self.viewer.layers:
            if hasattr(lyr, "rendering"):
                lyr.rendering = mode

    def _reset_view(self):
        """Reload ORIGINAL side_volume.npy and reset camera."""
        side_path = self._path_side_volume()
        if side_path and os.path.isfile(side_path):
            try:
                vol = np.load(side_path)
                if vol.ndim == 3:
                    self._load_into_viewer(vol, name="CBCT")
                else:
                    QtWidgets.QMessageBox.warning(None, "Unsupported",
                                                  f"side_volume.npy has ndim={vol.ndim}, expected 3.")
            except Exception as e:
                QtWidgets.QMessageBox.critical(None, "Load error", f"Failed to load side_volume.npy:\n{e}")
        else:
            self.viewer.reset_view()
        # keep dims as chosen
        self.viewer.dims.ndisplay = 3 if self.chk3D.isChecked() else 2

    # ----------------------------
    # Implant angle (LEFT / RIGHT) with rotation+save+reload
    # ----------------------------
    def _parse_angle_from_box(self):
        txt = self.txtImplantAngle.text().strip()
        if not txt:
            QtWidgets.QMessageBox.warning(None, "No value", "Please enter an angle (degrees).")
            return None
        try:
            return float(txt)
        except ValueError:
            QtWidgets.QMessageBox.critical(None, "Invalid value", f"'{txt}' is not a number.")
            return None

    def _rotate_save_reload(self, angle_deg):
        """
        Load side_volume.npy, rotate around Z by angle_deg,
        save to rotated_expanded_zaxis.npy, and reload into napari.
        """
        side_path = self._path_side_volume()
        save_path = self._path_rotated_volume()
        if not side_path or not os.path.isfile(side_path):
            QtWidgets.QMessageBox.warning(None, "Missing volume",
                                          "Cannot find side_volume.npy. Load a patient folder first.")
            return

        try:
            import scipy.ndimage as ndimage
        except Exception:
            QtWidgets.QMessageBox.critical(None, "Missing dependency",
                                           "scipy is required for rotation (pip install scipy).")
            return

        try:
            volume = np.load(side_path)
            if volume.ndim != 3:
                QtWidgets.QMessageBox.warning(None, "Unsupported",
                                              f"side_volume.npy has ndim={volume.ndim}, expected 3.")
                return

            # Rotate in (Y, X) plane => around Z axis
            rotated = ndimage.rotate(
                volume,
                angle=float(angle_deg),
                axes=(1, 0),       # (Y, X)
                reshape=True,      # expand to fit rotated content
                order=1,           # bilinear
                mode='nearest'     # border handling
            )

            # Save rotated volume next to side_volume.npy
            np.save(save_path, rotated)

            # Reload rotated into napari
            self._load_into_viewer(rotated, name="CBCT (rotated)")

        except Exception as e:
            QtWidgets.QMessageBox.critical(None, "Rotation error",
                                           f"Failed to rotate/save/reload volume:\n{e}")

    def _set_implant_angle_left(self):
        angle = self._parse_angle_from_box()
        if angle is None:
            return
        self.implant_angle_left_deg = angle
        self.implant_angle_deg = angle
        self.lblImplantAngle.setText(f"Current implant angle (Left): {self.implant_angle_left_deg:.1f}°")
        # Apply rotation, save, and reload
        self._rotate_save_reload(angle_deg=self.implant_angle_left_deg)

    def _set_implant_angle_right(self):
        angle = self._parse_angle_from_box()
        if angle is None:
            return
        right_angle = (360.0 - angle) % 360.0
        self.implant_angle_right_deg = right_angle
        self.implant_angle_deg = right_angle
        self.lblImplantAngle.setText(f"Current implant angle (Right): {self.implant_angle_right_deg:.1f}°")
        # Apply rotation, save, and reload
        self._rotate_save_reload(angle_deg=self.implant_angle_right_deg)

    # ----------------------------
    # Missing teeth slice (manual/auto) — placeholder logic
    # ----------------------------
    def _set_missing_slice_manual(self):
        s_txt = self.txtMissingStart.text().strip()
        e_txt = self.txtMissingEnd.text().strip()
        if not s_txt or not e_txt:
            QtWidgets.QMessageBox.warning(None, "Missing value", "Please enter both start and end slice numbers.")
            return
        try:
            s_val = int(s_txt)
            e_val = int(e_txt)
        except ValueError:
            QtWidgets.QMessageBox.critical(None, "Invalid value", "Start/end must be integers.")
            return
        if e_val < s_val:
            QtWidgets.QMessageBox.warning(None, "Range error", "End slice must be >= start slice.")
            return
        self.missing_slice_start = s_val
        self.missing_slice_end = e_val
        QtWidgets.QMessageBox.information(None, "Set", f"Missing teeth slice set to [{s_val}, {e_val}].")

    def _get_missing_slice_automatically(self):
        # Placeholder stub — replace with your algorithm later.
        if self.missing_slice_start is None or self.missing_slice_end is None:
            QtWidgets.QMessageBox.information(
                None, "Auto-detect",
                "Automatic detection is not implemented yet.\n\n"
                "Tip: implement your algorithm here, then set:\n"
                "  self.missing_slice_start, self.missing_slice_end\n"
                "and update the text boxes."
            )
        else:
            QtWidgets.QMessageBox.information(
                None, "Auto-detect",
                f"Current missing teeth slice range: [{self.missing_slice_start}, {self.missing_slice_end}]"
            )

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
