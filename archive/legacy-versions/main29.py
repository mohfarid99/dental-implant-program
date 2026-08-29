# -*- coding: utf-8 -*-
"""
main.py
- Tab 7 mirrors "Selected CBCT: <folder>/d_side_volume/side_volume.npy"
- Embeds a Napari viewer and loads the CBCT volume as a 3D rendering
- 3D controls: rendering mode, 2D/3D toggle, reset view
- Textbox + two buttons to set implant angle for LEFT/RIGHT teeth
- NEW: Rotation axis selector (Z-axis or X-axis)
- "Missing teeth slice" row with start/end boxes + 2 buttons
- LEFT/RIGHT actions:
    * rotate around chosen axis:
        - Z-axis => axes=(1,2)
        - X-axis => axes=(0,1)
    * ✅ rotations are chained and saved to ONE file: rotated_current.npy
    * export rotated PNG slices into dedicated folders
    * reload rotated volume in Napari
- Reset View reloads side_volume.npy (original) and ✅ deletes rotated_current.npy and legacy rotated files.

- Auto classification: loads classification.pt (EfficientNet-B0 2-logit with 2.5D input)
  to pick lowest & highest indices predicted as class 0 ("missing")

- NEW on Tab index 4 ("Dental Implant Database"):
  * CSV-backed table (implant_database.csv next to this script)
  * Add / Update / Delete rows (ID, Diameter(mm), Height(mm))
  * Click row to edit; ID must be unique

- NEW: "Get Dental Implant Recommendation" runs YOLOv8 segmentation (segmentation.pt)
  * Loads segmentation.pt from the SAME FOLDER as this file
  * Segments only the selected missing-slice range (start..end)
  * Saves binary masks into d_side_volume/segmented_AB and d_side_volume/segmented_MC
  * Folders are cleared each run
  * Computes best implant candidates (sorted by volume)
  * Saves final_AB.npy, final_MC.npy, and lets you APPLY any implant option to overwrite final_implant.npy
  * Visualizes: base volume (mip), AB (red, iso, translucent_no_depth),
                Implant (green, iso, translucent), MC (yellow, iso, translucent)
  * Layer order: base, final_AB, final_implant, final_MC

- NEW: Tab 7 iso-threshold controls for final_AB and final_MC like Slicer:
  * Slider + spinbox (0..1) + percent label for each
  * Defaults to 0.99 (99%) and live-updates napari iso layers

- NEW: Tab 7 implant options chooser:
  * After recommendation, show dropdown of all applicable implants (sorted by volume)
  * Apply any selected option -> regenerate final_implant.npy and refresh viewer

- UPDATED: Implant list now comes EXCLUSIVELY from implant_database.csv
  * IDs are preserved in candidate list, dropdown labels, and report file.
"""
import os
os.environ.pop("QT_OPENGL", None)
os.environ.pop("VISPY_FORCE_GL", None)
os.environ["QT_OPENGL"] = "desktop"    # request desktop GL

from PyQt5 import QtCore, QtGui, QtWidgets
fmt = QtGui.QSurfaceFormat()
fmt.setRenderableType(QtGui.QSurfaceFormat.OpenGL)    # desktop GL
fmt.setVersion(2, 1)
fmt.setProfile(QtGui.QSurfaceFormat.CompatibilityProfile)
fmt.setDepthBufferSize(24)
fmt.setStencilBufferSize(8)
fmt.setSamples(0)
QtGui.QSurfaceFormat.setDefaultFormat(fmt)

from vispy import app
app.use_app('pyqt5')                                 # no gloo.gl.use_gl(...)

import napari
import subprocess, os, sys, importlib.util
import numpy as np

# ---- new: imports for auto classification / preprocessing ----
import traceback
import cv2  # OpenCV
# Torch/Torchvision are imported lazily in _load_classifier_model

def _diagnose_env(parent=None, title="Python Env"):
    spec = importlib.util.find_spec("napari")
    QtWidgets.QMessageBox.information(
        parent, title,
        f"PYTHON: {sys.executable}\n"
        f"napari found: {spec is not None}\n"
        f"napari spec: {spec}\n"
    )

import scipy.ndimage as ndi

def _clean_binary_stack(arr, open_iter=1, close_iter=2, min_voxels=500):
    arr = (arr > 0)
    # 3D opening/closing
    arr = ndi.binary_opening(arr, structure=np.ones((3,3,3)), iterations=open_iter)
    arr = ndi.binary_closing(arr, structure=np.ones((3,3,3)), iterations=close_iter)
    # fill holes slice-wise (optional)
    for z in range(arr.shape[0]):
        arr[z] = ndi.binary_fill_holes(arr[z])
    # remove small components
    lbl, n = ndi.label(arr)
    if n > 0:
        sizes = np.bincount(lbl.ravel())
        keep = sizes >= min_voxels
        keep[0] = False
        arr = keep[lbl]
    return arr.astype(np.uint8)

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

        # Implant angle row
        angleRow = QtWidgets.QHBoxLayout()
        angleRow.setSpacing(10)

        # --- NEW: rotation axis chooser ---
        self.cmbAxis = QtWidgets.QComboBox()
        self.cmbAxis.addItems(["Z-axis", "X-axis"])  # default: Z-axis
        self.cmbAxis.setToolTip("Choose which axis to rotate the volume around")
        # ----------------------------------

        self.txtImplantAngle = QtWidgets.QLineEdit()
        self.txtImplantAngle.setPlaceholderText("Enter implant angle (deg)")
        self.txtImplantAngle.setFixedWidth(200)
        self.btnSetAngleLeft = QtWidgets.QPushButton("Set Angle (Left Teeth)")
        self.btnSetAngleRight = QtWidgets.QPushButton("Set Angle (Right Teeth)")
        self.lblImplantAngle = QtWidgets.QLabel("Current implant angle: —°")

        angleRow.addWidget(QtWidgets.QLabel("Rotate around:"))
        angleRow.addWidget(self.cmbAxis)
        angleRow.addSpacing(12)
        angleRow.addWidget(QtWidgets.QLabel("Implant Angle:"))
        angleRow.addWidget(self.txtImplantAngle)
        angleRow.addWidget(self.btnSetAngleLeft)
        angleRow.addWidget(self.btnSetAngleRight)
        angleRow.addSpacing(12)
        angleRow.addWidget(self.lblImplantAngle)
        angleRow.addStretch(1)

        # Missing slice row
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
        # Patch Napari status checker to avoid noisy exceptions
        try:
            import napari._qt.threads.status_checker as sc # type: ignore
            _orig_calc_status = sc.StatusChecker.calculate_status
            def _safe_calc_status(self, *a, **kw):
                try:
                    return _orig_calc_status(self, *a, **kw)
                except TypeError as e:
                    if "'NoneType' object is not subscriptable" in str(e):
                        return None
                    raise
                except Exception:
                    return None
            sc.StatusChecker.calculate_status = _safe_calc_status
        except Exception:
            pass

        self.viewerWidget = self.viewer.window._qt_viewer  # embed canvas+controls (no menubar)

        # ---- Iso-threshold controls for AB / MC (Slicer-like) ----
        thrRow = QtWidgets.QHBoxLayout(); thrRow.setSpacing(10)

        # AB controls
        self.lblAbThr = QtWidgets.QLabel("AB iso threshold:")
        self.sldAbThr = QtWidgets.QSlider(QtCore.Qt.Horizontal); self.sldAbThr.setRange(0, 100); self.sldAbThr.setValue(99)
        self.spnAbThr = QtWidgets.QDoubleSpinBox(); self.spnAbThr.setRange(0.0, 1.0); self.spnAbThr.setSingleStep(0.01); self.spnAbThr.setDecimals(2); self.spnAbThr.setValue(0.99)
        self.lblAbPct = QtWidgets.QLabel("99%")

        # MC controls
        self.lblMcThr = QtWidgets.QLabel("MC iso threshold:")
        self.sldMcThr = QtWidgets.QSlider(QtCore.Qt.Horizontal); self.sldMcThr.setRange(0, 100); self.sldMcThr.setValue(99)
        self.spnMcThr = QtWidgets.QDoubleSpinBox(); self.spnMcThr.setRange(0.0, 1.0); self.spnMcThr.setSingleStep(0.01); self.spnMcThr.setDecimals(2); self.spnMcThr.setValue(0.99)
        self.lblMcPct = QtWidgets.QLabel("99%")

        thrRow.addWidget(self.lblAbThr); thrRow.addWidget(self.sldAbThr); thrRow.addWidget(self.spnAbThr); thrRow.addWidget(self.lblAbPct)
        thrRow.addSpacing(20)
        thrRow.addWidget(self.lblMcThr); thrRow.addWidget(self.sldMcThr); thrRow.addWidget(self.spnMcThr); thrRow.addWidget(self.lblMcPct)
        thrRow.addStretch(1)

        # Run segmentation + recommendation
        self.btnGetImplantRec = QtWidgets.QPushButton("Get Dental Implant Recommendation")
        self.btnGetImplantRec.setObjectName("btnGetImplantRec")
        self.btnGetImplantRec.setFixedHeight(36)

        layout7.addWidget(self.lblMirror)
        layout7.addLayout(controls)
        layout7.addLayout(angleRow)
        layout7.addLayout(missingRow)
        layout7.addLayout(thrRow)  # add threshold controls here
        layout7.addWidget(self.viewerWidget, stretch=1)
        layout7.addWidget(self.btnGetImplantRec, alignment=QtCore.Qt.AlignRight)

        # ---- NEW: applicable implant chooser (hidden until run produces options) ----
        chooserRow = QtWidgets.QHBoxLayout()
        self.cmbImplantOptions = QtWidgets.QComboBox()
        self.cmbImplantOptions.setEnabled(False)
        self.btnApplyImplant = QtWidgets.QPushButton("Apply Selected Implant")
        self.btnApplyImplant.setEnabled(False)
        chooserRow.addWidget(QtWidgets.QLabel("Other applicable options:"))
        chooserRow.addWidget(self.cmbImplantOptions, stretch=1)
        chooserRow.addWidget(self.btnApplyImplant)
        layout7.addLayout(chooserRow)

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
        # TAB 5: Dental Implant Database (CRUD + CSV)
        # ----------------------------
        self.tab_6 = QtWidgets.QWidget()
        self.tab_6.setObjectName("tab_6")

        form = QtWidgets.QGridLayout()
        form.setHorizontalSpacing(12)
        form.setVerticalSpacing(8)

        self.label = QtWidgets.QLabel(self.tab_6); self.label.setText("ID")
        self.label_2 = QtWidgets.QLabel(self.tab_6); self.label_2.setText("Diameter (mm)")
        self.label_3 = QtWidgets.QLabel(self.tab_6); self.label_3.setText("Height (mm)")

        self.edId = QtWidgets.QLineEdit(self.tab_6); self.edId.setPlaceholderText("Model Number (unique)")
        self.edDia = QtWidgets.QLineEdit(self.tab_6); self.edDia.setPlaceholderText("e.g. 4.5")
        self.edHgt = QtWidgets.QLineEdit(self.tab_6); self.edHgt.setPlaceholderText("e.g. 11.5")

        dbl = QtGui.QDoubleValidator(0.0, 100.0, 2, self.tab_6)
        dbl.setNotation(QtGui.QDoubleValidator.StandardNotation)
        self.edDia.setValidator(dbl)
        self.edHgt.setValidator(dbl)

        form.addWidget(self.label,   0, 0)
        form.addWidget(self.edId,    0, 1)
        form.addWidget(self.label_2, 1, 0)
        form.addWidget(self.edDia,   1, 1)
        form.addWidget(self.label_3, 2, 0)
        form.addWidget(self.edHgt,   2, 1)

        btnRow = QtWidgets.QHBoxLayout()
        self.btnAdd = QtWidgets.QPushButton("Add")
        self.btnUpdate = QtWidgets.QPushButton("Update")
        self.btnDelete = QtWidgets.QPushButton("Delete")
        self.btnSave = QtWidgets.QPushButton("Save CSV")
        self.btnLoad = QtWidgets.QPushButton("Reload CSV")
        btnRow.addWidget(self.btnAdd)
        btnRow.addWidget(self.btnUpdate)
        btnRow.addWidget(self.btnDelete)
        btnRow.addStretch(1)
        btnRow.addWidget(self.btnLoad)
        btnRow.addWidget(self.btnSave)

        self.tableView = QtWidgets.QTableView(self.tab_6)
        self.tableView.setSelectionBehavior(QtWidgets.QAbstractItemView.SelectRows)
        self.tableView.setSelectionMode(QtWidgets.QAbstractItemView.SingleSelection)
        self.tableView.setEditTriggers(QtWidgets.QAbstractItemView.NoEditTriggers)
        self.tableView.horizontalHeader().setStretchLastSection(True)

        self._db_model = QtGui.QStandardItemModel(self.tableView)
        self._db_model.setHorizontalHeaderLabels(["ID", "Diameter (mm)", "Height (mm)"])
        self.tableView.setModel(self._db_model)

        v6 = QtWidgets.QVBoxLayout(self.tab_6)
        v6.addLayout(form)
        v6.addLayout(btnRow)
        v6.addWidget(self.tableView)

        self.tabWidget.addTab(self.tab_6, "")

        # ----------------------------
        # STATE
        # ----------------------------
        self.implant_angle_deg = None
        self.implant_angle_left_deg = None
        self.implant_angle_right_deg = None
        self.missing_slice_start = None
        self.missing_slice_end = None
        self._overlay_layer = None
        self._boundary_layer = None

        # iso layers & thresholds
        self._ab_layer = None
        self._mc_layer = None
        self._imp_layer = None
        self._ab_iso = 0.99
        self._mc_iso = 0.99

        self._is_updating_implant = False

        # classifier cache/config (MATCH TRAINING)
        self._clf_model = None
        self._clf_device = "cpu"
        self._clf_img_size = 224
        self._clf_band_frac = (0.0, 1.0)
        self._clf_use_clahe = False
        self._clf_top_cap_frac = 0.0
        self._clf_k = 2
        self._clf_missing_label = 0  # 0 = missing

        # NEW: candidates from last recommendation run
        # (rid, diameter_mm, height_mm, x_px, y_px, start_i, end_i, volume)
        self._candidate_implants = []
        self._candidate_ctx = None

        # ----------------------------
        # SIGNALS
        # ----------------------------
        # open the external IAN annotation tool from Tab 2 button
        self.pushButton_4.clicked.connect(self.open_ian_tool)
        self.pushButton.clicked.connect(self.browse_directory)
        self.pushButton_2.clicked.connect(self.run_convert)
        self.lineEdit.textChanged.connect(self._update_tab7_cbct)

        self.chk3D.toggled.connect(self._toggle_3d)
        self.cmbRendering.currentTextChanged.connect(self._set_rendering)
        self.btnReset.clicked.connect(self._reset_view)

        self.btnSetAngleLeft.clicked.connect(self._set_implant_angle_left)
        self.btnSetAngleRight.clicked.connect(self._set_implant_angle_right)
        self.txtImplantAngle.returnPressed.connect(self._set_implant_angle_left)

        self.btnSetMissingSlice.clicked.connect(self._set_missing_slice_manual)
        self.btnGetMissingAuto.clicked.connect(self._get_missing_slice_automatically)

        self.btnAdd.clicked.connect(self._db_add_row)
        self.btnUpdate.clicked.connect(self._db_update_row)
        self.btnDelete.clicked.connect(self._db_delete_selected)
        self.btnSave.clicked.connect(self._db_save_csv)
        self.btnLoad.clicked.connect(self._db_load_csv)
        self.tableView.selectionModel().selectionChanged.connect(self._db_on_selection)

        # iso threshold sync & handlers
        self.sldAbThr.valueChanged.connect(lambda v: self.spnAbThr.setValue(v/100.0))
        self.spnAbThr.valueChanged.connect(self._on_ab_iso_changed)
        self.sldMcThr.valueChanged.connect(lambda v: self.spnMcThr.setValue(v/100.0))
        self.spnMcThr.valueChanged.connect(self._on_mc_iso_changed)

        # NEW: apply selected implant
        self.btnApplyImplant.clicked.connect(self._apply_selected_implant)

        self._db_load_csv()

        # Full pipeline: segmentation -> recommendation -> visualize with options
        self.btnGetImplantRec.clicked.connect(self._run_full_recommendation)

        self.retranslateUi(Dialog)
        self.tabWidget.setCurrentIndex(0)
        self.tabWidget_2.setCurrentIndex(0)
        QtCore.QMetaObject.connectSlotsByName(Dialog)

    # ----------------------------
    # Helpers for CBCT paths & loading
    # ----------------------------
    def _cbct_dir(self):
        root = self.lineEdit.text().strip()
        if not root:
            return None
        return os.path.join(root, "d_side_volume")

    def _path_side_volume(self):
        d = self._cbct_dir()
        if not d: return None
        return os.path.join(d, "side_volume.npy")

    # --- Rotated volume path helpers (legacy + unified) ---
    def _path_rotated_volume_z(self):
        d = self._cbct_dir()
        if not d: return None
        return os.path.join(d, "rotated_expanded_zaxis.npy")

    def _path_rotated_volume(self):
        return self._path_rotated_volume_z()

    def _path_rotated_volume_x(self):
        d = self._cbct_dir()
        if not d: return None
        return os.path.join(d, "rotated_expanded_xaxis.npy")

    def _path_rotated_volume_unified(self):
        d = self._cbct_dir()
        if not d: return None
        return os.path.join(d, "rotated_current.npy")

    def _delete_rotated_files(self):
        """Remove unified + legacy rotated npys."""
        for p in (self._path_rotated_volume_unified(),
                  self._path_rotated_volume_z(),
                  self._path_rotated_volume_x()):
            try:
                if p and os.path.isfile(p):
                    os.remove(p)
            except Exception:
                pass

    def _base_path_for_rotation(self):
        """
        When rotating:
        - If a unified rotated file exists, rotate THAT (so rotations chain).
        - Else if any legacy rotated exists, rotate the newest legacy.
        - Else rotate original side_volume.npy.
        """
        p_unified = self._path_rotated_volume_unified()
        if p_unified and os.path.isfile(p_unified):
            return p_unified
        p_any_legacy = self._path_any_rotated_volume()
        if p_any_legacy and os.path.isfile(p_any_legacy):
            return p_any_legacy
        return self._path_side_volume()

    def _path_any_rotated_volume(self):
        """Return newest rotated volume, preferring unified if present."""
        cand = []
        pu = self._path_rotated_volume_unified()
        pz = self._path_rotated_volume_z()
        px = self._path_rotated_volume_x()
        for p in (pu, pz, px):
            if p and os.path.isfile(p):
                try:
                    cand.append((os.path.getmtime(p), p))
                except Exception:
                    cand.append((0.0, p))
        if not cand:
            return None
        cand.sort(reverse=True)
        return cand[0][1]

    def _load_into_viewer(self, vol, name="CBCT"):
        # Always render base with 'mip' as requested
        self.viewer.layers.clear()
        self.viewer.add_image(
            vol, name=name, colormap="gray",
            interpolation2d="linear", interpolation3d="linear",
            blending="additive",
            rendering="mip",  # <--- force mip
            rgb=False
        )
        self.viewer.dims.ndisplay = 3 if self.chk3D.isChecked() else 2
        self.viewer.reset_view()
        self.viewer.camera.zoom = 1.0

    # ---------- keep overlays behind the CBCT image (buried inside) ----------
    def _bury_overlays(self, base_layer):
        try:
            layers = self.viewer.layers
            if base_layer in layers:
                base_idx = list(layers).index(base_layer)
                layers.move(base_idx, len(layers) - 1)
            for lay in (self._overlay_layer, self._boundary_layer):
                if lay and lay in layers:
                    idx = list(layers).index(lay)
                    layers.move(idx, 0)
            if self._overlay_layer:
                self._overlay_layer.opacity = 0.6
                try:
                    self._overlay_layer.blending = "translucent"
                except Exception:
                    pass
            if self._boundary_layer:
                try:
                    self._boundary_layer.blending = "translucent"
                except Exception:
                    pass
        except Exception:
            pass

    # ----------------------------
    # Utility: reliably get the CBCT Image layer (not labels/shapes)
    # ----------------------------
    def _get_image_layer(self):
        try:
            import napari
            for lyr in self.viewer.layers:
                try:
                    if isinstance(lyr, napari.layers.Image):
                        return lyr
                except Exception:
                    continue
        except Exception:
            pass
        return None

    # ----------------------------
    # Actions / Logic (Tab 7)
    # ----------------------------
    def _update_tab7_cbct(self, txt):
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
                    zmax = int(vol.shape[0]) - 1
                    self.txtMissingStart.setText("0")
                    self.txtMissingEnd.setText(str(zmax))
                else:
                    self.lblMirror.setText(f"Selected CBCT: {side_path}  (unsupported ndim={vol.ndim})")
            except Exception as e:
                self.lblMirror.setText(f"Error loading CBCT: {e}")

    def _toggle_3d(self, checked):
        self.viewer.dims.ndisplay = 3 if checked else 2

    def _set_rendering(self, mode: str):
        """
        Keep base as 'mip' when (re)loading via _load_into_viewer/_show_final_results_in_napari.
        Do NOT override overlay image layers' rendering (they may be 'iso').
        """
        try:
            import napari
        except Exception:
            return

        # Only force the base layer(s) to MIP. Leave overlays alone.
        base_names = {
            os.path.basename(self._path_side_volume() or ""),
            os.path.basename(self._path_rotated_volume_unified() or ""),
            os.path.basename(self._path_rotated_volume_z() or ""),
            os.path.basename(self._path_rotated_volume_x() or ""),
            "CBCT",
            "CBCT (rotated Z-axis, left)", "CBCT (rotated Z-axis, right)",
            "CBCT (rotated X-axis, left)", "CBCT (rotated X-axis, right)",
        }
        for lyr in list(self.viewer.layers):
            try:
                if isinstance(lyr, napari.layers.Image) and lyr.name in base_names:
                    lyr.rendering = "mip"
            except Exception:
                continue

    def _reset_view(self):
        # NEW: clear any rotated files (unified + legacy)
        self._delete_rotated_files()

        side_path = self._path_side_volume()
        if side_path and os.path.isfile(side_path):
            try:
                vol = np.load(side_path)
                if vol.ndim == 3:
                    self._load_into_viewer(vol, name="CBCT")
                    zmax = int(vol.shape[0]) - 1
                    self.txtMissingStart.setText("0")
                    self.txtMissingEnd.setText(str(zmax))
                    if self._overlay_layer and self._overlay_layer in self.viewer.layers:
                        self.viewer.layers.remove(self._overlay_layer); self._overlay_layer = None
                    if self._boundary_layer and self._boundary_layer in self.viewer.layers:
                        self.viewer.layers.remove(self._boundary_layer); self._boundary_layer = None
                else:
                    QtWidgets.QMessageBox.warning(None, "Unsupported",
                                                  f"side_volume.npy has ndim={vol.ndim}, expected 3.")
            except Exception as e:
                QtWidgets.QMessageBox.critical(None, "Load error", f"Failed to load side_volume.npy:\n{e}")
        else:
            self.viewer.reset_view()
        self.viewer.dims.ndisplay = 3 if self.chk3D.isChecked() else 2

        # clear iso refs on reset
        self._ab_layer = None
        self._mc_layer = None
        self._imp_layer = None

    # ----------------------------
    # Implant angle (LEFT / RIGHT) rotation around chosen axis
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

    # --- Chain-rotate and save to unified file (Z) ---
    def _rotate_and_save_z(self, angle_deg: float, tag: str):
        """
        Chain-rotate around Z (i.e., (Y,X) plane: axes=(1,2)),
        using rotated_current.npy if present; otherwise side_volume.npy (or newest legacy).
        Save back into ONE unified file: rotated_current.npy
        """
        base_path = self._base_path_for_rotation()
        if not base_path or not os.path.isfile(base_path):
            QtWidgets.QMessageBox.warning(None, "Missing volume",
                                          "Cannot find side_volume.npy. Load a patient folder first.")
            return
        try:
            import scipy.ndimage as ndimage
            from PIL import Image
        except Exception:
            QtWidgets.QMessageBox.critical(None, "Missing dependency",
                                           "Required packages not found. Please install scipy and pillow.")
            return

        try:
            patient_dir = os.path.dirname(base_path)
            volume = np.load(base_path)  # (Z, Y, X)
            if volume.ndim != 3:
                QtWidgets.QMessageBox.warning(None, "Unsupported",
                                              f"Volume has ndim={volume.ndim}, expected 3.")
                return

            rotated_volume = ndimage.rotate(volume, angle=float(angle_deg), axes=(1, 2), reshape=True)

            rotated_slices_dir = os.path.join(patient_dir, f'rotated_slices_{tag}')
            os.makedirs(rotated_slices_dir, exist_ok=True)
            for i in range(rotated_volume.shape[0]):
                sl = rotated_volume[i]
                if np.min(sl) == np.max(sl):
                    sl8 = np.zeros_like(sl, dtype=np.uint8)
                else:
                    sl8 = (255 * (sl - np.min(sl)) / (np.max(sl) - np.min(sl))).astype(np.uint8)
                Image.fromarray(sl8).save(os.path.join(rotated_slices_dir, f'slice_{i:03d}.png'))

            # Save to unified file
            save_path = self._path_rotated_volume_unified()
            np.save(save_path, rotated_volume)

            self._load_into_viewer(rotated_volume, name=os.path.basename(save_path))

            if self._overlay_layer and self._overlay_layer in self.viewer.layers:
                self.viewer.layers.remove(self._overlay_layer); self._overlay_layer = None
            if self._boundary_layer and self._boundary_layer in self.viewer.layers:
                self.viewer.layers.remove(self._boundary_layer); self._boundary_layer = None

            zmax = int(rotated_volume.shape[0]) - 1
            self.txtMissingStart.setText("0")
            self.txtMissingEnd.setText(str(zmax))

        except Exception as e:
            QtWidgets.QMessageBox.critical(None, "Rotation error",
                                           f"Failed to rotate/save/reload volume (Z-axis):\n{e}")

    # --- Chain-rotate and save to unified file (X) ---
    def _rotate_and_save_x(self, angle_deg: float, tag: str):
        """
        Chain-rotate around X (i.e., (Z,Y) plane: axes=(0,1)),
        using rotated_current.npy if present; otherwise side_volume.npy (or newest legacy).
        Save back into ONE unified file: rotated_current.npy
        """
        base_path = self._base_path_for_rotation()
        if not base_path or not os.path.isfile(base_path):
            QtWidgets.QMessageBox.warning(None, "Missing volume",
                                          "Cannot find side_volume.npy. Load a patient folder first.")
            return
        try:
            import scipy.ndimage as ndimage
            from PIL import Image
        except Exception:
            QtWidgets.QMessageBox.critical(None, "Missing dependency",
                                           "Required packages not found. Please install scipy and pillow.")
            return

        try:
            patient_dir = os.path.dirname(base_path)
            volume = np.load(base_path)  # (Z, Y, X)
            if volume.ndim != 3:
                QtWidgets.QMessageBox.warning(None, "Unsupported",
                                              f"Volume has ndim={volume.ndim}, expected 3.")
                return

            rotated_volume = ndimage.rotate(volume, angle=float(angle_deg), axes=(0, 1), reshape=True)

            rotated_slices_dir = os.path.join(patient_dir, f'rotated_slices_{tag}_x')
            os.makedirs(rotated_slices_dir, exist_ok=True)
            for i in range(rotated_volume.shape[0]):
                sl = rotated_volume[i]
                if np.min(sl) == np.max(sl):
                    sl8 = np.zeros_like(sl, dtype=np.uint8)
                else:
                    sl8 = (255 * (sl - np.min(sl)) / (np.max(sl) - np.min(sl))).astype(np.uint8)
                Image.fromarray(sl8).save(os.path.join(rotated_slices_dir, f'slice_{i:03d}.png'))

            # Save to unified file
            save_path = self._path_rotated_volume_unified()
            np.save(save_path, rotated_volume)

            self._load_into_viewer(rotated_volume, name=os.path.basename(save_path))

            if self._overlay_layer and self._overlay_layer in self.viewer.layers:
                self.viewer.layers.remove(self._overlay_layer); self._overlay_layer = None
            if self._boundary_layer and self._boundary_layer in self.viewer.layers:
                self.viewer.layers.remove(self._boundary_layer); self._boundary_layer = None

            zmax = int(rotated_volume.shape[0]) - 1
            self.txtMissingStart.setText("0")
            self.txtMissingEnd.setText(str(zmax))

        except Exception as e:
            QtWidgets.QMessageBox.critical(None, "Rotation error",
                                           f"Failed to rotate/save/reload volume (X-axis):\n{e}")

    def _set_implant_angle_left(self):
        angle = self._parse_angle_from_box()
        if angle is None:
            return
        self.implant_angle_left_deg = angle
        self.implant_angle_deg = angle
        axis = self.cmbAxis.currentText()
        self.lblImplantAngle.setText(f"Current implant angle (Left, {axis}): {self.implant_angle_left_deg:.1f}°")
        if axis.startswith("Z"):
            self._rotate_and_save_z(angle_deg=self.implant_angle_left_deg, tag="left")
        else:
            self._rotate_and_save_x(angle_deg=self.implant_angle_left_deg, tag="left")

    def _set_implant_angle_right(self):
        angle = self._parse_angle_from_box()
        if angle is None:
            return
        right_angle = (360.0 - angle) % 360.0
        self.implant_angle_right_deg = right_angle
        self.implant_angle_deg = right_angle
        axis = self.cmbAxis.currentText()
        self.lblImplantAngle.setText(f"Current implant angle (Right, {axis}): {self.implant_angle_right_deg:.1f}°")
        if axis.startswith("Z"):
            self._rotate_and_save_z(angle_deg=self.implant_angle_right_deg, tag="right")
        else:
            self._rotate_and_save_x(angle_deg=self.implant_angle_right_deg, tag="right")

    # ----------------------------
    # Missing teeth slice (manual/auto)
    # ----------------------------
    def _set_missing_slice_manual(self):
        base = self._get_image_layer()
        if base is None:
            QtWidgets.QMessageBox.warning(None, "No volume", "Load a volume first.")
            return

        try:
            vol_shape = tuple(np.asarray(base.data).shape)
        except Exception:
            QtWidgets.QMessageBox.warning(None, "Unsupported", "Viewer layer data not accessible.")
            return

        if len(vol_shape) != 3:
            QtWidgets.QMessageBox.warning(None, "Unsupported", "Expected a 3D volume to overlay on.")
            return

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

        zmax = vol_shape[0] - 1
        if s_val < 0 or e_val < 0 or s_val > zmax or e_val > zmax:
            QtWidgets.QMessageBox.warning(None, "Range error", f"Slices must be within [0, {zmax}].")
            return
        if e_val < s_val:
            QtWidgets.QMessageBox.warning(None, "Range error", "End slice must be >= start slice.")
            return

        self.missing_slice_start = s_val
        self.missing_slice_end = e_val

        labels = np.zeros(vol_shape, dtype=np.uint8)
        labels[s_val, :, :] = 2
        labels[e_val, :, :] = 3

        if self._overlay_layer and self._overlay_layer in self.viewer.layers:
            self.viewer.layers.remove(self._overlay_layer); self._overlay_layer = None
        if self._boundary_layer and self._boundary_layer in self.viewer.layers:
            self.viewer.layers.remove(self._boundary_layer); self._boundary_layer = None

        self._overlay_layer = self.viewer.add_labels(labels, name="Missing Slices", opacity=1.0)
        try:
            self._overlay_layer.color = {
                2: (1.0, 0.0, 0.0, 0.95),
                3: (1.0, 0.0, 0.0, 0.95),
            }
        except Exception:
            pass

        try:
            Y, X = vol_shape[1], vol_shape[2]
            rect_yx = np.array([[0, 0], [0, X - 1], [Y - 1, X - 1], [Y - 1, 0], [0, 0]], dtype=float)
            shapes = []
            for z in (s_val, e_val):
                zcol = np.full((rect_yx.shape[0], 1), float(z))
                coords_zyx = np.concatenate([zcol, rect_yx], axis=1)
                shapes.append(coords_zyx)
            self._boundary_layer = self.viewer.add_shapes(
                shapes, shape_type="path", edge_color="red", edge_width=1,
                face_color="transparent", name="Missing Slice Boundaries",
            )
            try:
                if hasattr(self._boundary_layer, "selectable"):
                    self._boundary_layer.selectable = False
                if hasattr(self._boundary_layer, "editable"):
                    self._boundary_layer.editable = False
                if hasattr(self._boundary_layer, "_allow_pick"):
                    self._boundary_layer._allow_pick = False
            except Exception:
                pass
        except Exception:
            self._boundary_layer = None

        self._bury_overlays(self._get_image_layer())

        QtWidgets.QMessageBox.information(None, "Set",
                                          f"Missing teeth slice set to [{s_val}, {e_val}].\n"
                                          f"Solid red planes drawn on first and last slice only (buried inside CBCT).")

    # ----------------------------
    # Auto classification helpers
    # ----------------------------
    def _load_classifier_model(self):
        if self._clf_model is not None:
            return self._clf_model

        model_path = self._find_model_path()
        if not model_path:
            raise FileNotFoundError(
                "classification.pt not found. Place it next to main.py or inside the patient's d_side_volume folder."
            )

        import torch
        import torch.nn as nn
        from torchvision import models
        from collections import OrderedDict

        try:
            m = torch.jit.load(model_path, map_location=self._clf_device)
            m.eval()
            self._clf_model = m
            return self._clf_model
        except Exception:
            pass

        obj = torch.load(model_path, map_location=self._clf_device)

        if hasattr(obj, "forward"):
            try: obj.eval()
            except Exception: pass
            self._clf_model = obj
            return self._clf_model

        if isinstance(obj, dict) and "state_dict" in obj:
            sd = obj["state_dict"]
        else:
            sd = obj if isinstance(obj, (dict, OrderedDict)) else None

        if sd is None:
            raise RuntimeError("Unsupported model format in classification.pt (not a Module or state_dict).")

        def strip_prefix(state_dict, prefixes=("net.", "module.", "model.")):
            new_sd = OrderedDict()
            for k, v in state_dict.items():
                nk = k
                for p in prefixes:
                    if nk.startswith(p):
                        nk = nk[len(p):]
                new_sd[new_sd.setdefault(nk, nk)] = v
            return new_sd

        sd = strip_prefix(sd)

        in_ch = int(2 * self._clf_k + 1)
        base = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.IMAGENET1K_V1)

        old_first: nn.Conv2d = base.features[0][0]
        new_first = nn.Conv2d(in_ch, old_first.out_channels,
                              kernel_size=old_first.kernel_size,
                              stride=old_first.stride,
                              padding=old_first.padding,
                              bias=False)
        with torch.no_grad():
            w = old_first.weight
            w_mean = w.mean(dim=1, keepdim=True)
            if in_ch == 3:
                new_first.weight.copy_(w)
            else:
                new_first.weight.copy_(w_mean.repeat(1, in_ch, 1, 1))
        base.features[0][0] = new_first

        in_feat = base.classifier[1].in_features
        base.classifier[1] = nn.Linear(in_feat, 2)

        try:
            base.load_state_dict(sd, strict=True)
        except Exception as e_strict:
            model_keys = set(base.state_dict().keys())
            from collections import OrderedDict as OD
            filtered = OD((k, v) for k, v in sd.items() if k in model_keys)
            try:
                base.load_state_dict(filtered, strict=False)
            except Exception as e_non_strict:
                missing = [k for k in model_keys if k not in filtered]
                unexpected = [k for k in sd.keys() if k not in model_keys]
                msg = (
                    "Error(s) in loading state_dict for EfficientNet:\n"
                    f"Strict load error: {e_strict}\n"
                    f"Non-strict load error: {e_non_strict}\n"
                    f"Missing keys (first 10): {missing[:10]}\n"
                    f"Unexpected keys (first 10): {unexpected[:10]}"
                )
                raise RuntimeError(msg)

        base.eval()
        self._clf_model = base
        return self._clf_model

    def _prep_single_slice(self, slc2d: np.ndarray) -> np.ndarray:
        img = slc2d
        if img.dtype != np.uint8:
            vmin, vmax = float(np.min(img)), float(np.max(img))
            if vmax > vmin:
                img = ((img - vmin) / (vmax - vmin) * 255.0).astype(np.uint8)
            else:
                img = np.zeros_like(img, dtype=np.uint8)

        h = img.shape[0]
        y0 = int(h * self._clf_band_frac[0]); y1 = int(h * self._clf_band_frac[1])
        y0 = max(0, min(y0, h-1)); y1 = max(1, min(y1, h))
        if y1 <= y0: y0, y1 = 0, h
        roi = img[y0:y1, :]

        if self._clf_use_clahe:
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
            roi = clahe.apply(roi)

        roi = cv2.resize(roi, (self._clf_img_size, self._clf_img_size), interpolation=cv2.INTER_AREA)
        roi = roi.astype(np.float32)
        mn, mx = float(roi.min()), float(roi.max())
        roi = (roi - mn) / (mx - mn) if (mx - mn) > 1e-6 else roi
        return roi

    def _slice_stack_2p5d(self, vol: np.ndarray, idx: int) -> np.ndarray:
        k = int(self._clf_k)
        Z = vol.shape[0]
        chans = []
        for d in range(-k, k+1):
            j = min(max(idx + d, 0), Z - 1)
            s = vol[j]
            chans.append(self._prep_single_slice(s))
        stack = np.stack(chans, axis=0)
        return stack

    def _classify_volume_missing_range(self, vol: np.ndarray):
        import torch
        model = self._load_classifier_model()
        device = torch.device(self._clf_device)
        model.to(device)

        Z = vol.shape[0]
        preds = []
        with torch.inference_mode():
            for i in range(Z):
                x_np = self._slice_stack_2p5d(vol, i)
                x = torch.from_numpy(x_np).unsqueeze(0)
                x = x.to(device, dtype=torch.float32)
                logits = model(x)
                pred = int(torch.argmax(logits, dim=1).item())
                preds.append(pred)

        cnt0 = sum(1 for p in preds if p == 0)
        cnt1 = Z - cnt0
        print(f"[Classifier] Pred counts -> class0: {cnt0}, class1: {cnt1}, Z={Z}")

        miss_lbl = int(self._clf_missing_label)
        indices_miss = [i for i, p in enumerate(preds) if p == miss_lbl]
        if not indices_miss:
            alt_lbl = 1 - miss_lbl
            indices_alt = [i for i, p in enumerate(preds) if p == alt_lbl]
            if indices_alt:
                QtWidgets.QMessageBox.information(
                    None, "Auto-detect",
                    "Model predicted no slices with the configured missing label.\n"
                    "Assuming the missing label is swapped; using the other class."
                )
                return min(indices_alt), max(indices_alt), preds
            return None, None, preds

        return min(indices_miss), max(indices_miss), preds

    def _get_missing_slice_automatically(self):
        base = self._get_image_layer()
        if base is None:
            QtWidgets.QMessageBox.warning(None, "No volume", "Load a volume first.")
            return

        try:
            vol = np.asarray(base.data)
        except Exception:
            QtWidgets.QMessageBox.warning(None, "Unsupported", "Viewer layer data not accessible.")
            return

        if vol.ndim != 3:
            QtWidgets.QMessageBox.warning(None, "Unsupported", "Expected a 3D volume.")
            return

        try:
            start_idx, end_idx, preds = self._classify_volume_missing_range(vol)
        except Exception as e:
            QtWidgets.QMessageBox.critical(
                None, "Model load error",
                f"Failed to load classification.pt:\n{e}"
            )
            print("Model load / inference traceback:\n", traceback.format_exc())
            return

        if start_idx is None or end_idx is None:
            QtWidgets.QMessageBox.information(
                None, "Auto-detect",
                "Classifier did not find any 'missing' slices.\n"
                "You can set the range manually."
            )
            return

        self.txtMissingStart.setText(str(start_idx))
        self.txtMissingEnd.setText(str(end_idx))
        self._set_missing_slice_manual()

    # ----------------------------
    # Model path helpers
    # ----------------------------
    def _find_model_path(self):
        cand = []
        try:
            script_dir = os.path.dirname(os.path.abspath(sys.argv[0]))
            cand.append(os.path.join(script_dir, "classification.pt"))
        except Exception:
            pass
        d = self._cbct_dir()
        if d:
            cand.append(os.path.join(d, "classification.pt"))
        cand.append(os.path.join(os.getcwd(), "classification.pt"))
        for p in cand:
            if p and os.path.isfile(p):
                return p
        return None

    # ----------------------------
    # CSV-backed Implant Database
    # ----------------------------
    def _db_csv_path(self):
        try:
            base = os.path.dirname(os.path.abspath(sys.argv[0]))
        except Exception:
            base = os.getcwd()
        return os.path.join(base, "implant_database.csv")

    def _db_set_rows(self, rows):
        self._db_model.removeRows(0, self._db_model.rowCount())
        for rid, dia, hgt in rows:
            it_id = QtGui.QStandardItem(str(rid))
            it_d  = QtGui.QStandardItem(f"{float(dia):g}")
            it_h  = QtGui.QStandardItem(f"{float(hgt):g}")
            for it in (it_id, it_d, it_h):
                it.setEditable(False)
            self._db_model.appendRow([it_id, it_d, it_h])
        self.tableView.resizeColumnsToContents()

    def _db_get_rows(self):
        rows = []
        for r in range(self._db_model.rowCount()):
            rid = self._db_model.item(r, 0).text().strip()
            dia = float(self._db_model.item(r, 1).text())
            hgt = float(self._db_model.item(r, 2).text())
            rows.append((rid, dia, hgt))
        return rows

    def _db_load_csv(self):
        path = self._db_csv_path()
        rows = []
        if os.path.isfile(path):
            try:
                with open(path, "r", encoding="utf-8") as f:
                    for line in f:
                        line = line.strip()
                        if not line or line.lower().startswith("id,"):
                            continue
                        parts = [p.strip() for p in line.split(",")]
                        if len(parts) >= 3:
                            rid, d, h = parts[0], float(parts[1]), float(parts[2])
                            rows.append((rid, d, h))
            except Exception as e:
                QtWidgets.QMessageBox.warning(None, "CSV", f"Failed to read CSV:\n{e}")
        else:
            rows = [
                ("GS2M35080", 3.5, 8.5), ("GS2M35100", 3.5, 10), ("GS2M35110", 3.5, 11.5),
                ("GS2M35130", 3.5, 13), ("GS2M35150", 3.5, 15),
                ("GS2R40070", 4.0, 7), ("GS2R40080", 4.0, 8.5), ("GS2R40100", 4.0, 10),
                ("GS2R40110", 4.0, 11.5), ("GS2R40130", 4.0, 13), ("GS2R40150", 4.0, 15),
                ("GS2W45070", 4.5, 7), ("GS2W45080", 4.5, 8.5), ("GS2W45100", 4.5, 10),
                ("GS2W45110", 4.5, 11.5), ("GS2W45130", 4.5, 13), ("GS2W45150", 4.5, 15),
                ("GS2W50070", 5.0, 7), ("GS2W50080", 5.0, 8.5), ("GS2W50100", 5.0, 10),
                ("GS2W50110", 5.0, 11.5), ("GS2W50130", 5.0, 13), ("GS2W50150", 5.0, 15),
                ("MSP25103R", 2.0, 10), ("MSP25104R", 2.0, 10), ("MSP25113R", 2.5, 11.5),
                ("MSP25114R", 2.5, 11.5), ("MSP25133R", 2.5, 13), ("MSP25134R", 2.5, 13),
                ("MSP25153R", 2.5, 15), ("MSP25154R", 2.5, 15),
                ("MSP30103R", 3.0, 10), ("MSP30104R", 3.0, 10), ("MSP30113R", 3.0, 11.5),
                ("MSP30114R", 3.0, 11.5), ("MSP30133R", 3.0, 13), ("MSP30134R", 3.0, 13),
                ("MSP30153R", 3.0, 15), ("MSP30154R", 3.0, 15),
            ]
            try:
                with open(path, "w", encoding="utf-8") as f:
                    f.write("ID,Diameter(mm),Height(mm)\n")
                    for rid, d, h in rows:
                        f.write(f"{rid},{d},{h}\n")
            except Exception:
                pass
        self._db_set_rows(rows)

    def _db_save_csv(self):
        path = self._db_csv_path()
        try:
            with open(path, "w", encoding="utf-8") as f:
                f.write("ID,Diameter(mm),Height(mm)\n")
                for rid, d, h in self._db_get_rows():
                    f.write(f"{rid},{d},{h}\n")
            QtWidgets.QMessageBox.information(None, "Saved", f"Database saved to:\n{path}")
        except Exception as e:
            QtWidgets.QMessageBox.critical(None, "Save error", f"Failed to save CSV:\n{e}")

    def _db_on_selection(self):
        idxs = self.tableView.selectionModel().selectedRows()
        if not idxs:
            return
        r = idxs[0].row()
        self.edId.setText(self._db_model.item(r, 0).text())
        self.edDia.setText(self._db_model.item(r, 1).text())
        self.edHgt.setText(self._db_model.item(r, 2).text())

    def _db_validate_inputs(self):
        rid = self.edId.text().strip()
        dia = self.edDia.text().strip()
        hgt = self.edHgt.text().strip()
        if not rid:
            QtWidgets.QMessageBox.warning(None, "Input", "Please enter an ID.")
            return None
        if not dia or not hgt:
            QtWidgets.QMessageBox.warning(None, "Input", "Please enter diameter and height.")
            return None
        try:
            d = float(dia); h = float(hgt)
        except ValueError:
            QtWidgets.QMessageBox.warning(None, "Input", "Diameter/Height must be numbers.")
            return None
        return rid, d, h

    def _db_find_row_by_id(self, rid: str):
        for r in range(self._db_model.rowCount()):
            if self._db_model.item(r, 0).text().strip().lower() == rid.strip().lower():
                return r
        return -1

    def _db_add_row(self):
        valid = self._db_validate_inputs()
        if not valid:
            return
        rid, d, h = valid
        if self._db_find_row_by_id(rid) != -1:
            QtWidgets.QMessageBox.warning(None, "Duplicate ID", f"'{rid}' already exists. Use Update instead.")
            return
        it_id = QtGui.QStandardItem(rid)
        it_d  = QtGui.QStandardItem(f"{d:g}")
        it_h  = QtGui.QStandardItem(f"{h:g}")
        for it in (it_id, it_d, it_h):
            it.setEditable(False)
        self._db_model.appendRow([it_id, it_d, it_h])
        self.tableView.resizeColumnsToContents()

    def _db_update_row(self):
        valid = self._db_validate_inputs()
        if not valid:
            return
        rid, d, h = valid
        idxs = self.tableView.selectionModel().selectedRows()
        r = idxs[0].row() if idxs else self._db_find_row_by_id(rid)
        if r == -1:
            QtWidgets.QMessageBox.warning(None, "Update", "Select a row or ensure the ID exists.")
            return
        old_id = self._db_model.item(r, 0).text()
        if rid.lower() != old_id.lower() and self._db_find_row_by_id(rid) != -1:
            QtWidgets.QMessageBox.warning(None, "Duplicate ID", f"'{rid}' already exists.")
            return
        self._db_model.item(r, 0).setText(rid)
        self._db_model.item(r, 1).setText(f"{d:g}")
        self._db_model.item(r, 2).setText(f"{h:g}")
        self.tableView.selectRow(r)

    def _db_delete_selected(self):
        idxs = self.tableView.selectionModel().selectedRows()
        if not idxs:
            QtWidgets.QMessageBox.information(None, "Delete", "Select a row to delete.")
            return
        r = idxs[0].row()
        rid = self._db_model.item(r, 0).text()
        if QtWidgets.QMessageBox.question(None, "Delete",
                                          f"Delete implant '{rid}'?",
                                          QtWidgets.QMessageBox.Yes | QtWidgets.QMessageBox.No
                                          ) == QtWidgets.QMessageBox.Yes:
            self._db_model.removeRow(r)

    # ---------- YOLOv8 segmentation helpers ----------
    def _find_segmentation_model_path(self):
        """Find segmentation.pt ONLY in the same folder as this file (main.py)."""
        try:
            script_dir = os.path.dirname(os.path.abspath(sys.argv[0]))
        except Exception:
            script_dir = os.getcwd()
        p = os.path.join(script_dir, "segmentation.pt")
        return p if os.path.isfile(p) else None

    def _slice_to_rgb(self, slc2d: np.ndarray) -> np.ndarray:
        s = np.asarray(slc2d)
        if s.dtype != np.uint8:
            vmin, vmax = float(s.min()), float(s.max())
            if vmax > vmin:
                s = ((s - vmin) / (vmax - vmin) * 255.0).astype(np.uint8)
            else:
                s = np.zeros_like(s, dtype=np.uint8)
        return cv2.merge((s, s, s))  # BGR

    def _clear_and_prepare_dir(self, path: str):
        os.makedirs(path, exist_ok=True)
        for name in os.listdir(path):
            fp = os.path.join(path, name)
            try:
                if os.path.isfile(fp) or os.path.islink(fp):
                    os.remove(fp)
                elif os.path.isdir(fp):
                    import shutil
                    shutil.rmtree(fp)
            except Exception:
                pass

    def _run_yolo_segmentation(self):
        """
        Segments AB (class 0) and MC (class 1) over the missing-slice range using YOLOv8 segment model.
        Loads segmentation.pt from SAME FOLDER as this file.
        Saves binary masks (original slice size) into segmented_AB/ and segmented_MC/ (cleared each run).
        """
        base = self._get_image_layer()
        if base is None:
            QtWidgets.QMessageBox.warning(None, "No volume", "Load a volume first.")
            return

        try:
            vol = np.asarray(base.data)
        except Exception as e:
            QtWidgets.QMessageBox.critical(None, "Data error", f"Viewer layer data not accessible:\n{e}")
            return

        if vol.ndim != 3:
            QtWidgets.QMessageBox.warning(None, "Unsupported", "Expected a 3D volume.")
            return

        try:
            s_val = int(self.txtMissingStart.text().strip())
            e_val = int(self.txtMissingEnd.text().strip())
        except Exception:
            QtWidgets.QMessageBox.warning(None, "Range", "Set a valid start/end slice first.")
            return

        Z = vol.shape[0]
        if s_val < 0 or e_val < 0 or s_val >= Z or e_val >= Z or e_val < s_val:
            QtWidgets.QMessageBox.warning(None, "Range", f"Slice range must be within [0, {Z-1}] and end >= start.")
            return

        patient_dir = self._cbct_dir()
        if not patient_dir:
            QtWidgets.QMessageBox.warning(None, "Folder", "Select a valid patient folder first.")
            return

        ab_dir = os.path.join(patient_dir, "segmented_AB")
        mc_dir = os.path.join(patient_dir, "segmented_MC")
        self._clear_and_prepare_dir(ab_dir)
        self._clear_and_prepare_dir(mc_dir)

        model_path = self._find_segmentation_model_path()
        if not model_path:
            QtWidgets.QMessageBox.critical(None, "Model not found",
                                           "segmentation.pt not found in the same folder as main.py.")
            return

        H, W = vol.shape[1], vol.shape[2]
        imgs_640 = []
        src_indices = list(range(s_val, e_val + 1))
        for idx in src_indices:
            rgb = self._slice_to_rgb(vol[idx])
            resized = cv2.resize(rgb, (640, 640), interpolation=cv2.INTER_AREA)
            imgs_640.append(resized)

        try:
            from ultralytics import YOLO
        except Exception:
            QtWidgets.QMessageBox.critical(None, "Missing dependency",
                                           "ultralytics not installed. `pip install ultralytics`")
            return

        try:
            model = YOLO(model_path, task='segment')
        except Exception as e:
            QtWidgets.QMessageBox.critical(None, "Model load error", f"Failed to load segmentation.pt:\n{e}")
            return

        try:
            results = model(imgs_640, save=False, imgsz=640, verbose=False)
        except Exception as e:
            QtWidgets.QMessageBox.critical(None, "Inference error", f"YOLO inference failed:\n{e}")
            return

        import torch
        previous_mc_mask = None

        i = s_val
        for res in results:
            ab_bin = np.zeros((H, W), dtype=np.uint8)
            mc_bin = None

            if getattr(res, "masks", None) is not None and res.masks is not None:
                try:
                    boxes = res.boxes.data if res.boxes is not None else None
                    if boxes is not None and boxes.numel() > 0:
                        clss = boxes[:, 5]
                    else:
                        clss = torch.empty((0,))

                    ab_idx = torch.where(clss == 0)[0]
                    mc_idx = torch.where(clss == 1)[0]

                    mdata = res.masks.data  # (n,640,640)
                    if mdata is not None and mdata.numel() > 0:
                        if ab_idx.numel() > 0:
                            ab_stack = mdata[ab_idx].cpu().numpy()
                            ab_union = (ab_stack > 0.5).any(axis=0).astype(np.uint8) * 255
                            _, ab_thresh = cv2.threshold(ab_union, 127, 255, cv2.THRESH_BINARY)
                            cnts, _ = cv2.findContours(ab_thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                            ab_filled_640 = np.zeros_like(ab_union)
                            cv2.drawContours(ab_filled_640, cnts, -1, 255, thickness=cv2.FILLED)
                            ab_bin = cv2.resize(ab_filled_640, (W, H), interpolation=cv2.INTER_NEAREST)

                        if mc_idx.numel() > 0:
                            mc_stack = mdata[mc_idx].cpu().numpy()
                            mc_union = (mc_stack > 0.5).any(axis=0).astype(np.uint8) * 255
                            mc_bin = cv2.resize(mc_union, (W, H), interpolation=cv2.INTER_NEAREST)
                            previous_mc_mask = mc_bin.copy()
                        else:
                            mc_bin = previous_mc_mask.copy() if previous_mc_mask is not None else np.zeros((H, W), dtype=np.uint8)
                    else:
                        mc_bin = previous_mc_mask.copy() if previous_mc_mask is not None else np.zeros((H, W), dtype=np.uint8)

                except Exception:
                    mc_bin = previous_mc_mask.copy() if previous_mc_mask is not None else np.zeros((H, W), dtype=np.uint8)
            else:
                mc_bin = previous_mc_mask.copy() if previous_mc_mask is not None else np.zeros((H, W), dtype=np.uint8)

            ab_out = os.path.join(ab_dir, f"merged_segs_{i}.jpg")
            mc_out = os.path.join(mc_dir, f"merged_segs_{i}.jpg")
            try:
                cv2.imwrite(ab_out, ab_bin)
                cv2.imwrite(mc_out, mc_bin)
            except Exception as e:
                print("Save error:", e)

            i += 1

        QtWidgets.QMessageBox.information(None, "Segmentation done",
                                          f"Saved AB masks to:\n{ab_dir}\n\nSaved MC masks to:\n{mc_dir}")

    # ---------- Full pipeline: segmentation -> recommendation -> options & apply ----------
    def _run_full_recommendation(self):
        # 1) Run segmenter
        self._run_yolo_segmentation()

        # 2) Compute candidates and save AB/MC stacks.
        try:
            candidates, ctx = self._compute_implant_recommendation_and_save(return_candidates=True)
        except Exception as e:
            QtWidgets.QMessageBox.critical(None, "Recommendation error", f"Failed to compute recommendation:\n{e}")
            return

        # >>> NEW: show AB/MC right away, regardless of candidates
        try:
            self._show_final_results_in_napari()
        except Exception as e:
            # don’t block the rest; just notify
            QtWidgets.QMessageBox.warning(None, "Visualization", f"Could not load AB/MC into viewer yet:\n{e}")

        # 3) Handle candidate list
        self._candidate_implants = candidates or []
        self._candidate_ctx = ctx if candidates else None

        if not self._candidate_implants:
            QtWidgets.QMessageBox.information(None, "No options", "No suitable implant found for the selected range.")
            self._populate_implant_options_ui()
            return

        # 4) Auto-apply the best (highest volume)
        try:
            self._save_implant_choice(self._candidate_implants[0], self._candidate_ctx)
        except Exception as e:
            QtWidgets.QMessageBox.critical(None, "Apply error", f"Failed to render the top-choice implant:\n{e}")
            return

        # 5) Show the options in UI so user can switch
        self._populate_implant_options_ui()

    # ---------- Recommendation code (reads implants from CSV; returns candidates + context) ----------
    def _compute_implant_recommendation_and_save(self, return_candidates=False):
        """
        Implements implant-selection algorithm taking start/end slices from the UI.
        Saves final_AB.npy/final_MC.npy.
        Returns (candidates_sorted_by_volume, context) if return_candidates=True.

        Implant list is loaded from the current in-memory CSV table:
        rows = self._db_get_rows() -> [(ID, Diameter_mm, Height_mm), ...]
        """
        import math, re
        from PIL import Image

        # --- Paths & inputs
        patient_dir = self._cbct_dir()
        if not patient_dir:
            raise RuntimeError("Invalid patient folder.")

        # Get range from UI
        try:
            ui_start = int(self.txtMissingStart.text().strip())
            ui_end   = int(self.txtMissingEnd.text().strip())
        except Exception:
            raise RuntimeError("Enter valid integers for start/end slices first.")

        if ui_end < ui_start:
            raise RuntimeError("End slice must be >= start slice.")

        start_filename = ui_start
        end_filename   = ui_end

        alveolar_bone_path = os.path.join(patient_dir, 'segmented_AB')
        mandibular_canal_path = os.path.join(patient_dir, 'segmented_MC')
        output_path = os.path.join(patient_dir, 'implant_planted')

        # choose *newest* rotated volume if present, else original
        _any = self._path_any_rotated_volume()
        all_slices_path = _any if (_any and os.path.isfile(_any)) else self._path_side_volume()

        implant_only_path = os.path.join(patient_dir, 'implant_only')
        final_AB_path = os.path.join(patient_dir, 'final_AB')
        final_MC_path = os.path.join(patient_dir, 'final_MC')
        patient_path_AB = os.path.join(patient_dir, 'final_AB.npy')
        patient_path_MC = os.path.join(patient_dir, 'final_MC.npy')
        patient_path_imp = os.path.join(patient_dir, 'final_implant.npy')  # may be written later

        for p in (output_path, implant_only_path, final_AB_path, final_MC_path):
            os.makedirs(p, exist_ok=True)

        def binarize_image(image):
            _, binary_image = cv2.threshold(image, 127, 255, cv2.THRESH_BINARY)
            return binary_image

        def _largest_contour(cnts):
            if not cnts: return None
            areas = [cv2.contourArea(c) for c in cnts]
            return cnts[int(np.argmax(areas))]

        def load_images(alv_path, mc_path):
            # Load only images in AB/MC folders (they were already created only for [start..end])
            alv = [binarize_image(cv2.imread(os.path.join(alv_path, img), cv2.IMREAD_GRAYSCALE))
                   for img in sorted(os.listdir(alv_path), key=self._natural_key)
                   if img.lower().endswith(('.png','.jpg','.jpeg','.bmp','.tiff'))]
            mc  = [binarize_image(cv2.imread(os.path.join(mc_path, img), cv2.IMREAD_GRAYSCALE))
                   for img in sorted(os.listdir(mc_path), key=self._natural_key)
                   if img.lower().endswith(('.png','.jpg','.jpeg','.bmp','.tiff'))]
            alv = [x for x in alv if x is not None]
            mc  = [x for x in mc if x is not None]
            return alv, mc

        def convert_to_pixels(mm, pixel_size=0.3): return int(mm / pixel_size)

        def get_segmented_area(image):
            binary_image = binarize_image(image)
            contours, _ = cv2.findContours(binary_image.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if contours:
                c = _largest_contour(contours)
                x, y, w, h = cv2.boundingRect(c)
                return x, y, w, h, [c]
            return None, None, None, None, None

        def get_canal_position(image):
            binary_image = binarize_image(image)
            contours, _ = cv2.findContours(binary_image.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if contours:
                c = _largest_contour(contours)
                x, y, w, h = cv2.boundingRect(c)
                return y, y + h
            return None

        def get_image_resolution(directory):
            images = [f for f in os.listdir(directory) if f.lower().endswith(('.png','.jpg','.jpeg','.bmp','.tiff'))]
            if not images: raise ValueError("No image files in " + directory)
            image_path = os.path.join(directory, images[0])
            with Image.open(image_path) as img:
                w, h = img.size
            return w, h

        def extract_number(filename):
            m = re.search(r'\d+', filename)
            return int(m.group()) if m else -1

        # ==== SAVING AS (Z,Y,X) ONLY ====
        def read_and_convert_images_to_npy(input_folder, output_file):
            files = [f for f in os.listdir(input_folder) if f.lower().endswith(('.png','.jpg','.jpeg','.bmp'))]
            files.sort(key=lambda x: extract_number(x))
            imgs = []
            for f in files:
                im = cv2.imread(os.path.join(input_folder, f), cv2.IMREAD_UNCHANGED)
                if im is None:
                    continue
                # force to 2D grayscale
                if im.ndim == 3:
                    if im.shape[2] == 1:
                        im = im[:, :, 0]
                    else:
                        im = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
                imgs.append(im)
            if imgs:
                arr = np.stack(imgs, axis=0)     # (Z, Y, X)
            else:
                arr = np.zeros((0,), dtype=np.uint8)
            if arr.ndim == 4 and arr.shape[-1] == 1:
                arr = arr[..., 0]
            np.save(output_file, arr)

        def copy_files(src, dst):
            for filename in os.listdir(src):
                sf = os.path.join(src, filename)
                df = os.path.join(dst, filename)
                if os.path.isfile(sf):
                    import shutil
                    shutil.copy(sf, df)

        alveolar_bone_images, mandibular_canal_images = load_images(alveolar_bone_path, mandibular_canal_path)
        if not alveolar_bone_images or not mandibular_canal_images:
            raise RuntimeError("No segmented AB/MC images found. Make sure segmentation succeeded.")

        # ============ IMPLANTS FROM CSV ============
        db_rows = self._db_get_rows()
        if not db_rows:
            raise RuntimeError("Implant database is empty. Please add entries in the 'Dental Implant Database' tab.")
        implants = [(rid, float(d), float(h)) for (rid, d, h) in db_rows]

        safety_zone_between_teeth_mm = 1
        safety_zone_slices = math.ceil((safety_zone_between_teeth_mm * 1) / 0.3)

        def find_best_implant_middle(alv_imgs, mc_imgs, imps, start_slice):
            def calculate_volume(d, h):
                r = d / 2
                return 3.14159 * (r ** 2) * h

            number = len(alv_imgs)
            if number == 0: return None, None
            middle_index = number // 2 - 1 if number % 2 == 0 else number // 2

            safety_zone_pixels = convert_to_pixels(1)
            top_margin_pixels = convert_to_pixels(2)
            line_margin_3mm_pixels = convert_to_pixels(3)

            thickness_to_diameter = len(alv_imgs) * 0.3
            fitting = []

            alveolar_bone = binarize_image(alv_imgs[middle_index])
            mandibular_canal = binarize_image(mc_imgs[middle_index])

            x_ab, y_ab, w_ab, h_ab, contours_ab = get_segmented_area(alveolar_bone)
            canal_position = get_canal_position(mandibular_canal)
            if (x_ab is None) or (canal_position is None): return None, None
            canal_top, _ = canal_position

            # fallback center if needed
            top_contour_points = [pt[0][0] for pt in contours_ab[0] if pt[0][1] == y_ab]
            center_top_x = sum(top_contour_points)//len(top_contour_points) if top_contour_points else (x_ab + w_ab//2)

            below_3mm_points = [pt[0] for pt in contours_ab[0] if pt[0][1] >= y_ab + line_margin_3mm_pixels]
            if not below_3mm_points: return None, None

            iter_erosion = max(1, int(convert_to_pixels(1)))
            eroded_ab = cv2.erode(alveolar_bone, np.ones((safety_zone_pixels, safety_zone_pixels), np.uint8), iterations=iter_erosion)

            H, W = alveolar_bone.shape[:2]
            for rid, diameter, height in imps:
                d_px = convert_to_pixels(diameter)
                h_px = convert_to_pixels(height)

                implant_x = max(0, min(center_top_x - (d_px // 2), W - d_px))
                implant_y = y_ab

                # clearance inside eroded AB at least 3mm below crest
                ok_h = True
                ys = y_ab + line_margin_3mm_pixels
                ye = min(ys + h_px, H)
                if implant_x < 0 or implant_x + d_px > W or ys < 0 or ye > H:
                    ok_h = False
                else:
                    area = eroded_ab[ys:ye, implant_x:implant_x+d_px]
                    if area.size == 0 or not np.all(area == 255):
                        ok_h = False
                if not ok_h: continue

                # lower implant until just above canal margin
                for dy in range(implant_y, canal_top - top_margin_pixels - h_px + 1):
                    area = alveolar_bone[dy:dy+h_px, implant_x:implant_x+d_px]
                    if area.size > 0 and np.all(area == 255):
                        implant_y = dy
                        break

                if implant_y + h_px >= canal_top - top_margin_pixels:
                    continue

                vol = calculate_volume(diameter, height)
                if diameter <= thickness_to_diameter:
                    fitting.append((rid, diameter, height, vol, implant_x, implant_y))

            if not fitting: return None, None
            fitting.sort(key=lambda x: x[3], reverse=True)  # by volume
            print('middle index:', middle_index)
            return fitting, middle_index

        def find_best_implant(alv_imgs, mc_imgs, imps, start_slice, end_slice):
            mid_imps, mid_idx = find_best_implant_middle(alv_imgs, mc_imgs, imps, start_slice)
            if mid_imps is None: return []

            valid = []
            Z = len(alv_imgs)
            length_bt = end_slice - start_slice + 1  # UI-driven thickness

            for rid, diameter, height, volume, initial_x, initial_y in mid_imps:
                d_sl = math.ceil(diameter / 0.3)
                # center around mid_idx
                start_i = mid_idx - (d_sl // 2) - (0 if d_sl % 2 else 1)
                end_i   = mid_idx + (d_sl // 2)
                # clamp
                start_i = max(0, start_i)
                end_i   = min(Z - 1, end_i)
                print('start_i:', start_i)
                print('end_i:', end_i)
                if start_i > end_i: continue

                if d_sl + safety_zone_slices > length_bt:
                    # diameter too large for the UI-selected gap
                    continue

                d_px = convert_to_pixels(diameter)
                h_px = convert_to_pixels(height)
                safety_zone_pixels = convert_to_pixels(1)
                top_margin_pixels  = convert_to_pixels(2)
                line_margin_3mm_pixels = convert_to_pixels(3)

                suitable = True
                for i in range(start_i, end_i + 1):
                    alv = binarize_image(alv_imgs[i]); mc = binarize_image(mc_imgs[i])
                    x_ab, y_ab, w_ab, h_ab, contours_ab = get_segmented_area(alv)
                    canal_position = get_canal_position(mc)
                    if None in (x_ab, y_ab, w_ab, h_ab) or (canal_position is None):
                        suitable = False; break
                    canal_top, _ = canal_position

                    eroded = cv2.erode(alv, np.ones((safety_zone_pixels, safety_zone_pixels), np.uint8),
                                       iterations=max(1, int(convert_to_pixels(1))))

                    ys = y_ab + line_margin_3mm_pixels
                    ye = min(ys + h_px, eroded.shape[0])
                    xe = min(initial_x + d_px, eroded.shape[1])

                    if initial_x < 0 or xe > eroded.shape[1] or ys < 0 or ye > eroded.shape[0]:
                        suitable = False; break

                    area = eroded[ys:ye, initial_x:xe]
                    if area.size == 0 or not np.all(area == 255):
                        suitable = False; break

                    if initial_y + h_px >= canal_top - top_margin_pixels:
                        suitable = False; break

                if suitable:
                    valid.append((rid, diameter, height, initial_x, initial_y, start_i, end_i))
            return valid

        def draw_implant_in_images(alv_imgs_unused, diameter, height, implant_x, implant_y,
                           out_folder, starting_slices, ending_slices, resolution):
            """
            Draw a *cylindrical* implant:
            - Axis along Z (stack depth).
            - Circular cross-section across Z with max diameter = `diameter` (mm).
            - Each slice draws a vertical bar whose width follows the circle profile.
            """
            d_px = convert_to_pixels(diameter)
            h_px = convert_to_pixels(height)
            W, H = resolution  # (width, height)

            cx = implant_x + d_px // 2
            z0 = int(starting_slices)
            z1 = int(ending_slices)
            zc = 0.5 * (z0 + z1)
            r  = d_px / 2.0

            y0 = max(0, min(implant_y, H - h_px))
            y1 = min(H, y0 + h_px)

            for z in range(z0, z1 + 1):
                dz = abs(z - zc)
                if dz > r:
                    continue
                half_w = int(round((r*r - dz*dz) ** 0.5))
                wz = 2 * half_w
                if wz <= 0:
                    continue

                x0 = max(0, min(int(cx - half_w), W))
                x1 = max(0, min(int(cx + half_w), W))
                if x1 <= x0 or y1 <= y0:
                    continue

                implant_image = np.zeros((H, W), dtype=np.uint8)
                cv2.rectangle(implant_image, (x0, y0), (x1, y1), 255, -1)

                cv2.imwrite(os.path.join(out_folder, f"implant_{z}.png"), implant_image)

        def fill_black(end_total, range_start, range_end, fmt, out_folder, resolution):
            W, H = resolution
            for num in range(end_total):
                if not (range_start <= num <= range_end):
                    image = np.zeros((H, W), dtype=np.uint8)
                    cv2.imwrite(os.path.join(out_folder, fmt.format(num=num)), image)

        def draw_combined_image_with_implant(alv_imgs, mc_imgs, diameter, height, out_path, initial_x, initial_y, start_i, end_i, start_filename):
            # Optional viz; not essential for Napari rendering
            d_px = convert_to_pixels(diameter)
            h_px = convert_to_pixels(height)
            safety_zone_pixels = convert_to_pixels(1)
            line_margin_3mm_pixels = convert_to_pixels(3)
            line_margin_6mm_pixels = convert_to_pixels(6)

            H, W = alv_imgs[0].shape[:2]

            for i in range(len(alv_imgs)):
                alv = binarize_image(alv_imgs[i])
                mc  = binarize_image(mc_imgs[i])

                x_ab, y_ab, w_ab, h_ab, contours_ab = get_segmented_area(alv)
                canal_position = get_canal_position(mc)
                if None in (x_ab, y_ab, w_ab, h_ab) or (canal_position is None):
                    continue

                combined = np.zeros((H, W, 3), dtype=np.uint8)
                combined[alv == 255] = [0,0,255]   # AB red
                eroded = cv2.erode(alv, np.ones((safety_zone_pixels, safety_zone_pixels), np.uint8),
                                   iterations=max(1, int(convert_to_pixels(1))))
                er_cnts, _ = cv2.findContours(eroded, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                if er_cnts:
                    cv2.drawContours(combined, er_cnts, -1, (255,255,255), thickness=cv2.FILLED)
                    cv2.drawContours(combined, er_cnts, -1, (0,0,255), 1)

                combined[mc == 255] = [0,255,255]  # MC yellow

                # 3mm & 6mm reference lines within AB bbox
                line_y_3mm = min(H-1, y_ab + line_margin_3mm_pixels)
                line_y_6mm = min(H-1, y_ab + line_margin_6mm_pixels)
                for x in range(x_ab, min(W, x_ab + w_ab)):
                    if cv2.pointPolygonTest(contours_ab[0], (x, line_y_3mm), False) >= 0:
                        combined[line_y_3mm, x] = [255,0,0]
                    if cv2.pointPolygonTest(contours_ab[0], (x, line_y_6mm), False) >= 0:
                        combined[line_y_6mm, x] = [255,0,0]

                # crest center (cyan dot)
                top_pts = [pt[0][0] for pt in contours_ab[0] if pt[0][1] == y_ab]
                cx = sum(top_pts)//len(top_pts) if top_pts else (x_ab + w_ab//2)
                cy = y_ab
                cv2.circle(combined, (cx, cy), radius=3, color=(0,255,255), thickness=-3)

                # draw implant where applicable (rectangle preview)
                if start_i <= i <= end_i:
                    x0 = max(0, min(initial_x, W - d_px))
                    y0 = max(0, min(implant_y, H - h_px))
                    x1 = min(W-1, x0 + d_px)
                    y1 = min(H-1, y0 + h_px)
                    cv2.rectangle(combined, (x0, y0), (x1, y1), (0,255,0), 1)

                cv2.imwrite(os.path.join(out_path, f"combined_implant_image_{i+start_filename}.png"), combined)

        # ---- compute best implants using UI-provided start/end
        best_implants = find_best_implant(alveolar_bone_images, mandibular_canal_images, implants,
                                          start_filename, end_filename)

        # Attach volumes and sort (keep ID)
        def _volume_cyl(d_mm, h_mm):
            r = d_mm / 2.0
            return 3.14159 * (r ** 2) * h_mm

        cand_with_vol = []
        for (rid, d, h, x, y, si, ei) in best_implants:
            cand_with_vol.append((rid, d, h, x, y, si, ei, _volume_cyl(d, h)))
        cand_with_vol.sort(key=lambda t: t[-1], reverse=True)

        # Save report
        with open(os.path.join(patient_dir, "Suitable Implants.txt"), "w") as f:
            f.write("ALL SUITABLE IMPLANTS (sorted by volume):\n")
            for idx, (rid, d, h, _x, _y, si, ei, vol) in enumerate(cand_with_vol, start=1):
                f.write(f"{idx}. {rid} | Ø{d:g}×{h:g} mm | vol={vol:.2f} | start={si + start_filename}, end={ei + start_filename}\n")

        # Save AB/MC stacks (Z,Y,X) for viewer
        w, h = get_image_resolution(alveolar_bone_path)
        total_slices = int(np.load(all_slices_path).shape[0])

        def _fill_black_range(dst_folder, start, end, fmt):
            os.makedirs(dst_folder, exist_ok=True)
            # clear first
            for name in os.listdir(dst_folder):
                try:
                    os.remove(os.path.join(dst_folder, name))
                except Exception:
                    pass
            # copy AB/MC range
            if dst_folder == final_AB_path:
                src_folder = alveolar_bone_path
            else:
                src_folder = mandibular_canal_path
            copy_files(src_folder, dst_folder)
            # fill others
            fill_black(total_slices, start, end, fmt=fmt, out_folder=dst_folder, resolution=(w, h))

        _fill_black_range(final_AB_path, start_filename, end_filename, fmt="merged_segs_{num}.jpg")
        _fill_black_range(final_MC_path, start_filename, end_filename, fmt="merged_segs_{num}.jpg")

        # Save final npy stacks for viewer (strictly (Z,Y,X))
        read_and_convert_images_to_npy(final_AB_path, patient_path_AB)
        read_and_convert_images_to_npy(final_MC_path, patient_path_MC)

        AB = np.load(patient_path_AB); MC = np.load(patient_path_MC)
        AB = _clean_binary_stack(AB, open_iter=1, close_iter=2, min_voxels=800)
        MC = _clean_binary_stack(MC, open_iter=1, close_iter=1, min_voxels=300)
        np.save(patient_path_AB, AB); np.save(patient_path_MC, MC)

        QtWidgets.QMessageBox.information(None, "Recommendation",
                                          "Saved final_AB.npy and final_MC.npy.\nSelect and apply any implant option to create final_implant.npy.")

        # Build context for later "apply"
        ctx = {
            "patient_dir": patient_dir,
            "alveolar_bone_path": alveolar_bone_path,
            "mandibular_canal_path": mandibular_canal_path,
            "implant_only_path": implant_only_path,
            "final_AB_path": final_AB_path,
            "final_MC_path": final_MC_path,
            "all_slices_path": all_slices_path,
            "start_filename": start_filename,
            "end_filename": end_filename,
            "get_image_resolution": get_image_resolution,
            "convert_to_pixels": convert_to_pixels,
            "draw_implant_in_images": draw_implant_in_images,
            "fill_black": fill_black,
            "copy_files": copy_files,
            "read_and_convert_images_to_npy": read_and_convert_images_to_npy,
        }

        if return_candidates:
            return cand_with_vol, ctx

        # Legacy fallback: directly apply the best
        if cand_with_vol:
            self._save_implant_choice(cand_with_vol[0], ctx)
        else:
            raise RuntimeError("No suitable implant found.")

    def _natural_key(self, s: str):
        import re
        return [int(t) if t.isdigit() else t.lower() for t in re.split(r'(\d+)', s)]

    def _add_iso_image(self, volume01, name, cmap, blending, iso_thr=0.5):
        if volume01.ndim != 3 or not np.any(volume01):
            return None
        arr = np.ascontiguousarray(volume01.astype(np.float32))
        layer = self.viewer.add_image(
            arr,
            name=name,
            colormap=cmap,
            blending=blending,
            rendering="iso",
            contrast_limits=(0.0, 1.0),
            interpolation2d="nearest",   # was "linear"
            interpolation3d="nearest",   # was "linear"
        )
        try:
            layer.iso_threshold = iso_thr
        except Exception:
            pass
        return layer

    # ========== NEW: safe implant-only layer reload ==========
    def _load_implant_layer_only(self) -> bool:
        """
        Safely (re)load final_implant.npy into the existing implant layer, without
        touching the base CBCT or AB/MC layers. Returns True if visible data loaded.
        """
        try:
            patient_dir = self._cbct_dir()
            if not patient_dir:
                return False
            imp_path = os.path.join(patient_dir, 'final_implant.npy')
            if not os.path.isfile(imp_path):
                return False

            IMP = np.load(imp_path)
            if IMP.ndim == 4 and IMP.shape[-1] == 1:
                IMP = IMP[..., 0]
            IMP = (IMP > 0).astype(np.uint8)
            if not np.any(IMP):
                if self._imp_layer is not None:
                    self._imp_layer.visible = False
                return False

            arr = np.ascontiguousarray(IMP.astype(np.float32))

            # Create once, then just swap data
            if (self._imp_layer is None) or ('final_implant' not in self.viewer.layers):
                self._imp_layer = self.viewer.add_image(
                    arr, name="final_implant", colormap="green", blending="translucent",
                    rendering="iso", contrast_limits=(0.0, 1.0),
                    interpolation2d="nearest", interpolation3d="nearest",
                )
            else:
                self._imp_layer.data = arr
                try:
                    self._imp_layer.visible = True
                except Exception:
                    pass

            try:
                self._imp_layer.iso_threshold = 0.5
            except Exception:
                pass

            # keep intended layer order as best as we can
            try:
                order_names = [
                    os.path.basename(self._path_any_rotated_volume() or self._path_side_volume() or ""),
                    "final_AB",
                    "final_implant",
                    "final_MC",
                ]
                existing = [nm for nm in order_names if nm in self.viewer.layers]
                for idx, nm in enumerate(existing):
                    self.viewer.layers.move(self.viewer.layers.index(nm), idx)
            except Exception:
                pass

            return True
        except Exception:
            return False

    def _ensure_final_ab_mc_npys(self):
        """
        If final_AB.npy / final_MC.npy are missing or empty, rebuild them directly
        from segmented_AB/ and segmented_MC/ so Napari can show them again.
        """
        patient_dir = self._cbct_dir()
        if not patient_dir:
            return

        ab_npy  = os.path.join(patient_dir, 'final_AB.npy')
        mc_npy  = os.path.join(patient_dir, 'final_MC.npy')
        ab_dir  = os.path.join(patient_dir, 'segmented_AB')
        mc_dir  = os.path.join(patient_dir, 'segmented_MC')

        def _valid_npy(p):
            try:
                if not os.path.isfile(p):
                    return False
                arr = np.load(p, mmap_mode='r')
                return (arr.ndim == 3) and arr.size > 0
            except Exception:
                return False

        if _valid_npy(ab_npy) and _valid_npy(mc_npy):
            return

        # Base volume to know Z, and one mask to know H/W
        _any = self._path_any_rotated_volume()
        base_path = _any if (_any and os.path.isfile(_any)) else self._path_side_volume()
        if not base_path or not os.path.isfile(base_path):
            return

        try:
            vol = np.load(base_path, mmap_mode='r')
            Z = int(vol.shape[0])
        except Exception:
            return

        def _first_image_shape(folder):
            if not os.path.isdir(folder):
                return None
            for fn in sorted(os.listdir(folder)):
                if fn.lower().endswith(('.png','.jpg','.jpeg','.bmp','.tiff')):
                    im = cv2.imread(os.path.join(folder, fn), cv2.IMREAD_GRAYSCALE)
                    if im is not None:
                        return im.shape[:2]  # H, W
            return None

        ab_shape = _first_image_shape(ab_dir)
        mc_shape = _first_image_shape(mc_dir)
        hw = ab_shape or mc_shape
        if hw is None:
            return
        H, W = hw

        AB = np.zeros((Z, H, W), dtype=np.uint8)
        MC = np.zeros((Z, H, W), dtype=np.uint8)

        import re
        def _num_from_name(name):
            m = re.search(r'(\d+)', name)
            return int(m.group(1)) if m else None

        def _binarize(img):
            if img is None:
                return None
            if img.dtype != np.uint8:
                img = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
            _, bin_im = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)
            return (bin_im > 0).astype(np.uint8)

        if os.path.isdir(ab_dir):
            for fn in os.listdir(ab_dir):
                if not fn.lower().endswith(('.png','.jpg','.jpeg','.bmp','.tiff')):
                    continue
                z = _num_from_name(fn)
                if z is None or z < 0 or z >= Z:
                    continue
                im = cv2.imread(os.path.join(ab_dir, fn), cv2.IMREAD_GRAYSCALE)
                if im is None:
                    continue
                if im.shape[:2] != (H, W):
                    im = cv2.resize(im, (W, H), interpolation=cv2.INTER_NEAREST)
                m = _binarize(im)
                if m is not None:
                    AB[z] = m

        if os.path.isdir(mc_dir):
            for fn in os.listdir(mc_dir):
                if not fn.lower().endswith(('.png','.jpg','.jpeg','.bmp','.tiff')):
                    continue
                z = _num_from_name(fn)
                if z is None or z < 0 or z >= Z:
                    continue
                im = cv2.imread(os.path.join(mc_dir, fn), cv2.IMREAD_GRAYSCALE)
                if im is None:
                    continue
                if im.shape[:2] != (H, W):
                    im = cv2.resize(im, (W, H), interpolation=cv2.INTER_NEAREST)
                m = _binarize(im)
                if m is not None:
                    MC[z] = m

        AB = _clean_binary_stack(AB, open_iter=1, close_iter=2, min_voxels=800)
        MC = _clean_binary_stack(MC, open_iter=1, close_iter=1, min_voxels=300)
        np.save(ab_npy, AB)
        np.save(mc_npy, MC)

    def _show_final_results_in_napari(self):
        """Ensure final_AB/MC exist; then load base + AB + implant + MC as ISO overlays."""
        # Base path: newest rotated (X/Z/unified) if exists, else original
        _any = self._path_any_rotated_volume()
        base_path = _any if (_any and os.path.isfile(_any)) else self._path_side_volume()
        if not base_path or not os.path.isfile(base_path):
            raise RuntimeError("Missing base volume (side_volume.npy).")

        # If AB/MC npys are missing/empty, rebuild them from segmented_* folders
        self._ensure_final_ab_mc_npys()

        patient_dir = self._cbct_dir()
        ab_path  = os.path.join(patient_dir, 'final_AB.npy')
        mc_path  = os.path.join(patient_dir, 'final_MC.npy')
        imp_path = os.path.join(patient_dir, 'final_implant.npy')

        # Load arrays
        vol = np.load(base_path)

        if not os.path.isfile(ab_path) or not os.path.isfile(mc_path):
            raise RuntimeError("final_AB.npy / final_MC.npy not found. Run recommendation first.")

        AB  = np.load(ab_path)
        MC  = np.load(mc_path)

        # final_implant may not exist until an option is applied; handle gracefully
        IMP = None
        if os.path.isfile(imp_path):
            IMP = np.load(imp_path)

        # Ensure (Z, Y, X) and binary 0/1
        if AB.ndim == 4 and AB.shape[-1] == 1:  AB  = AB[..., 0]
        if MC.ndim == 4 and MC.shape[-1] == 1:  MC  = MC[..., 0]
        AB  = (AB > 0).astype(np.uint8)
        MC  = (MC > 0).astype(np.uint8)

        if IMP is not None:
            if IMP.ndim == 4 and IMP.shape[-1] == 1:
                IMP = IMP[..., 0]
            IMP = (IMP > 0).astype(np.uint8)

        # Rebuild viewer
        self.viewer.layers.clear()

        # Base image (force MIP)
        img_layer = self.viewer.add_image(
            vol, name=os.path.basename(base_path),
            colormap="gray",
            interpolation2d="linear", interpolation3d="linear",
            blending="additive",
            rendering="mip",
            rgb=False
        )

        # Clear refs before (re)adding
        self._ab_layer = None
        self._mc_layer = None
        self._imp_layer = None

        # Add layers (AB first, Implant optional, MC last)
        self._ab_layer  = self._add_iso_image(AB.astype(np.float32),"final_AB","red","translucent_no_depth", iso_thr=self._ab_iso)
        if self._ab_layer is not None:
            self._ab_layer.visible = True

        if IMP is not None and np.any(IMP):
            self._imp_layer = self._add_iso_image(IMP.astype(np.float32), "final_implant", "green", "translucent", iso_thr=0.5)
            if self._imp_layer is not None:
                self._imp_layer.visible = True

        self._mc_layer  = self._add_iso_image(MC.astype(np.float32),  "final_MC","yellow", "translucent",iso_thr=self._mc_iso)
        if self._mc_layer is not None:
            self._mc_layer.visible = True

        # Re-apply iso thresholds
        try:
            if self._ab_layer is not None: self._ab_layer.iso_threshold = self._ab_iso
            if self._mc_layer is not None: self._mc_layer.iso_threshold = self._mc_iso
        except Exception:
            pass

        # Order: base, AB, implant (if any), MC
        try:
            order = [img_layer, self._ab_layer, self._imp_layer, self._mc_layer]
            layers = [lyr for lyr in order if lyr is not None]
            for idx, lyr in enumerate(layers):
                self.viewer.layers.move(self.viewer.layers.index(lyr), idx)
        except Exception:
            pass

        # 3D on/off as per UI and reset view
        self.viewer.dims.ndisplay = 3 if self.chk3D.isChecked() else 2
        self.viewer.reset_view()

    # ---- Iso threshold slots ----
    def _on_ab_iso_changed(self, val: float):
        self._ab_iso = float(val)
        self.lblAbPct.setText(f"{int(round(self._ab_iso*100))}%")
        if self._ab_layer is not None:
            try:
                self._ab_layer.iso_threshold = self._ab_iso
            except Exception:
                pass

    def _on_mc_iso_changed(self, val: float):
        self._mc_iso = float(val)
        self.lblMcPct.setText(f"{int(round(self._mc_iso*100))}%")
        if self._mc_layer is not None:
            try:
                self._mc_layer.iso_threshold = self._mc_iso
            except Exception:
                pass

    # ----------------------------
    # NEW: Implant options UI logic
    # ----------------------------
    def _populate_implant_options_ui(self):
        """Fill the combo with applicable implants and enable the chooser."""
        self.cmbImplantOptions.clear()
        if not self._candidate_implants:
            self.cmbImplantOptions.setEnabled(False)
            self.btnApplyImplant.setEnabled(False)
            return

        for idx, (rid, d, h, _x, _y, s, e, vol) in enumerate(self._candidate_implants, start=1):
            label = f"#{idx}  {rid} — {d:g}×{h:g} mm | vol={vol:.2f} | slices={s}..{e}"
            self.cmbImplantOptions.addItem(label)

        self.cmbImplantOptions.setCurrentIndex(0)
        self.cmbImplantOptions.setEnabled(True)
        self.btnApplyImplant.setEnabled(True)

    # ========== UPDATED: apply choice safely ==========
    def _save_implant_choice(self, candidate, ctx):
        """
        Rebuilds implant-only images/final_implant.npy using the chosen candidate,
        then refreshes the viewer *safely*. 'candidate' = (rid, d_mm, h_mm, x_px, y_px, start_i, end_i, volume)
        """
        import os
        import numpy as np
        import cv2

        # prevent re-entrant updates during vispy draws
        if getattr(self, "_is_updating_implant", False):
            return
        self._is_updating_implant = True

        try:
            rid, d_mm, h_mm, x_px, y_px, start_i, end_i, _vol = candidate

            # unpack context
            (patient_dir, alveolar_bone_path, mandibular_canal_path, implant_only_path,
             final_AB_path, final_MC_path, all_slices_path,
             start_filename, end_filename, get_image_resolution,
             convert_to_pixels, draw_implant_in_images, fill_black, copy_files,
             read_and_convert_images_to_npy) = (
                ctx["patient_dir"], ctx["alveolar_bone_path"], ctx["mandibular_canal_path"], ctx["implant_only_path"],
                ctx["final_AB_path"], ctx["final_MC_path"], ctx["all_slices_path"],
                ctx["start_filename"], ctx["end_filename"], ctx["get_image_resolution"],
                ctx["convert_to_pixels"], ctx["draw_implant_in_images"], ctx["fill_black"], ctx["copy_files"],
                ctx["read_and_convert_images_to_npy"]
            )

            patient_path_imp = os.path.join(patient_dir, 'final_implant.npy')
            w, h = get_image_resolution(alveolar_bone_path)

            # Clear and redraw implant-only for the selected candidate
            os.makedirs(implant_only_path, exist_ok=True)
            for name in os.listdir(implant_only_path):
                try:
                    os.remove(os.path.join(implant_only_path, name))
                except Exception:
                    pass

            # Absolute Z indices in the full stack
            abs_start = start_i + start_filename
            abs_end   = end_i   + start_filename

            draw_implant_in_images(
                [], d_mm, h_mm, x_px, y_px,
                implant_only_path, abs_start, abs_end, resolution=(w, h)
            )

            total_slices = int(np.load(all_slices_path).shape[0])
            fill_black(total_slices, abs_start, abs_end,
                       fmt="implant_{num}.png", out_folder=implant_only_path, resolution=(w, h))

            # Save final_implant.npy (Z,Y,X)
            read_and_convert_images_to_npy(implant_only_path, patient_path_imp)

            # ---- defer the viewer update to the next event-loop tick
            def _deferred_update():
                try:
                    # Prefer in-place layer update (no scene rebuild)
                    if not self._load_implant_layer_only():
                        # First-time visualization (no base/AB/MC yet)
                        self._show_final_results_in_napari()
                except Exception as e:
                    try:
                        QtWidgets.QMessageBox.critical(None, "Visualization error", f"Failed to visualize results:\n{e}")
                    except Exception:
                        pass
                finally:
                    self._is_updating_implant = False

            QtCore.QTimer.singleShot(0, _deferred_update)

        except Exception as e:
            self._is_updating_implant = False
            QtWidgets.QMessageBox.critical(None, "Apply error", f"Failed to apply selected implant:\n{e}")

    def _apply_selected_implant(self):
        """User picked another option; rebuild 'final_implant.npy' and refresh."""
        idx = self.cmbImplantOptions.currentIndex()
        if idx < 0 or idx >= len(self._candidate_implants):
            QtWidgets.QMessageBox.warning(None, "Selection", "Pick an applicable implant first.")
            return
        if not self._candidate_ctx:
            QtWidgets.QMessageBox.warning(None, "Context", "No recommendation context found.")
            return
        try:
            cand = self._candidate_implants[idx]
            self._save_implant_choice(cand, self._candidate_ctx)
            QtWidgets.QMessageBox.information(None, "Applied", "Selected implant applied (final_implant.npy updated).")
        except Exception as e:
            QtWidgets.QMessageBox.critical(None, "Apply error", f"Failed to apply selected implant:\n{e}")

    # ----------------------------
    # Other actions
    # ----------------------------
    def browse_directory(self):
        directory = QtWidgets.QFileDialog.getExistingDirectory(None, "Select Patient Folder", "")
        if directory:
            self.lineEdit.setText(directory)

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
