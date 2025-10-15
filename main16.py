# -*- coding: utf-8 -*-
"""
main.py
- Tab 7 mirrors "Selected CBCT: <folder>/d_side_volume/side_volume.npy"
- Embeds a Napari viewer and loads the CBCT volume as a 3D rendering
- 3D controls: rendering mode, 2D/3D toggle, reset view
- Textbox + two buttons to set implant angle for LEFT/RIGHT teeth
- "Missing teeth slice" row with start/end boxes + 2 buttons
- LEFT/RIGHT actions:
    * rotate side_volume.npy around Z-axis (axes=(1,0), reshape=True)
    * save rotated_expanded_zaxis.npy in the same folder
    * export rotated PNG slices into rotated_slices_left/ or rotated_slices_right/
    * reload rotated volume in Napari
- Reset View reloads side_volume.npy (original)
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
"""
from PyQt5 import QtCore, QtGui, QtWidgets
import subprocess, os, sys, importlib.util
import numpy as np

# ---- new: imports for auto classification / preprocessing ----
import traceback
import cv2  # OpenCV for preprocessing
# Torch/Torchvision are imported lazily in _load_classifier_model


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
        # --- Patch Napari's status checker to ignore known face_normal=None bug ---
        try:
            import napari._qt.threads.status_checker as sc

            _orig_calc_status = sc.StatusChecker.calculate_status

            def _safe_calc_status(self, *a, **kw):
                try:
                    return _orig_calc_status(self, *a, **kw)
                except TypeError as e:
                    # Ignore known "face_normal=None" crashes from Shapes get_status
                    if "'NoneType' object is not subscriptable" in str(e):
                        return None
                    raise
                except Exception:
                    # Avoid spamming console, keep status thread alive
                    return None

            sc.StatusChecker.calculate_status = _safe_calc_status
        except Exception:
            pass

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
        # TAB 5: Dental Implant Database (CRUD + CSV)
        # ----------------------------
        self.tab_6 = QtWidgets.QWidget()
        self.tab_6.setObjectName("tab_6")

        # Top form
        form = QtWidgets.QGridLayout()
        form.setHorizontalSpacing(12)
        form.setVerticalSpacing(8)

        self.label = QtWidgets.QLabel(self.tab_6); self.label.setText("ID")
        self.label_2 = QtWidgets.QLabel(self.tab_6); self.label_2.setText("Diameter (mm)")
        self.label_3 = QtWidgets.QLabel(self.tab_6); self.label_3.setText("Height (mm)")

        self.edId = QtWidgets.QLineEdit(self.tab_6); self.edId.setPlaceholderText("Model Number (unique)")
        self.edDia = QtWidgets.QLineEdit(self.tab_6); self.edDia.setPlaceholderText("e.g. 4.5")
        self.edHgt = QtWidgets.QLineEdit(self.tab_6); self.edHgt.setPlaceholderText("e.g. 11.5")

        # numeric validators for dia/height
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

        # Buttons
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

        # Table
        self.tableView = QtWidgets.QTableView(self.tab_6)
        self.tableView.setSelectionBehavior(QtWidgets.QAbstractItemView.SelectRows)
        self.tableView.setSelectionMode(QtWidgets.QAbstractItemView.SingleSelection)
        self.tableView.setEditTriggers(QtWidgets.QAbstractItemView.NoEditTriggers)
        self.tableView.horizontalHeader().setStretchLastSection(True)

        # Model
        self._db_model = QtGui.QStandardItemModel(self.tableView)
        self._db_model.setHorizontalHeaderLabels(["ID", "Diameter (mm)", "Height (mm)"])
        self.tableView.setModel(self._db_model)

        # Layout composition
        v6 = QtWidgets.QVBoxLayout(self.tab_6)
        v6.addLayout(form)
        v6.addLayout(btnRow)
        v6.addWidget(self.tableView)

        self.tabWidget.addTab(self.tab_6, "")

        # ----------------------------
        # STATE
        # ----------------------------
        self.implant_angle_deg = None              # last set (for compatibility)
        self.implant_angle_left_deg = None         # explicit left
        self.implant_angle_right_deg = None        # explicit right
        self.missing_slice_start = None            # manual start slice
        self.missing_slice_end = None              # manual end slice
        self._overlay_layer = None                 # labels overlay for missing slices
        self._boundary_layer = None                # shapes layer (rectangle outlines on first/last)

        # classifier cache/config (MATCH TRAINING)
        self._clf_model = None
        self._clf_device = "cpu"                   # use CPU by default
        self._clf_img_size = 224
        self._clf_band_frac = (0.0, 1.0)           # <-- full height (your training)
        self._clf_use_clahe = False                # <-- no CLAHE (your training had no_clahe=True)
        self._clf_top_cap_frac = 0.0
        self._clf_k = 2                            # context_k used in your training reference (=> 5 channels)
        self._clf_missing_label = 0                # by training: 0 = missing, 1 = non-missing

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

        # Missing teeth slice handlers
        self.btnSetMissingSlice.clicked.connect(self._set_missing_slice_manual)
        self.btnGetMissingAuto.clicked.connect(self._get_missing_slice_automatically)

        # DB signals
        self.btnAdd.clicked.connect(self._db_add_row)
        self.btnUpdate.clicked.connect(self._db_update_row)
        self.btnDelete.clicked.connect(self._db_delete_selected)
        self.btnSave.clicked.connect(self._db_save_csv)
        self.btnLoad.clicked.connect(self._db_load_csv)
        self.tableView.selectionModel().selectionChanged.connect(self._db_on_selection)

        # Load DB at startup (creates CSV with defaults if missing)
        self._db_load_csv()

        # Segmentation: wire the button
        self.btnGetImplantRec.clicked.connect(self._run_yolo_segmentation)

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
        # We always save Z-rotations to this file (as requested)
        return os.path.join(d, "rotated_expanded_zaxis.npy")

    # --------- find classification.pt in (1) script dir, (2) patient d_side_volume, (3) CWD ----------
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

    def _load_into_viewer(self, vol, name="CBCT"):
        # Clear and load volume with current rendering mode
        self.viewer.layers.clear()
        self.viewer.add_image(
            vol, name=name, colormap="gray",
            interpolation2d="linear",   # <-- deprecation-safe
            interpolation3d="linear",   # <-- deprecation-safe
            blending="additive",
            rendering=self.cmbRendering.currentText(), rgb=False
        )
        self.viewer.dims.ndisplay = 3 if self.chk3D.isChecked() else 2
        self.viewer.reset_view()
        self.viewer.camera.zoom = 1.0

    # ---------- keep overlays behind the CBCT image (buried inside) ----------
    def _bury_overlays(self, base_layer):
        try:
            layers = self.viewer.layers
            # Move base (CBCT image) to the very top (rendered last)
            if base_layer in layers:
                base_idx = list(layers).index(base_layer)
                layers.move(base_idx, len(layers) - 1)

            # Move overlay layers to the bottom (rendered first)
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
    # ------------------------------------------------------------------------

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
                    # Initialize default missing slice range (0 ... Z-1)
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
        Apply rendering mode only to Image layers.
        Labels/Shapes do not support 'mip' etc., so we leave them alone.
        """
        try:
            import napari
        except Exception:
            return

        for lyr in list(self.viewer.layers):
            try:
                # Only Image layers support the volume rendering modes in the combobox
                if isinstance(lyr, napari.layers.Image):
                    # guard: accept only valid image renderers
                    if mode in ("attenuated_mip", "mip", "translucent", "additive", "iso"):
                        lyr.rendering = mode
                elif isinstance(lyr, napari.layers.Labels):
                    if getattr(lyr, "rendering", None) not in ("translucent", "iso"):
                        try:
                            lyr.rendering = "translucent"
                        except Exception:
                            pass
                # Shapes layers don't have volume rendering at all; ignore them
            except Exception:
                continue

    def _reset_view(self):
        """Reload ORIGINAL side_volume.npy and reset camera."""
        side_path = self._path_side_volume()
        if side_path and os.path.isfile(side_path):
            try:
                vol = np.load(side_path)
                if vol.ndim == 3:
                    self._load_into_viewer(vol, name="CBCT")
                    # Also reset default missing slice range
                    zmax = int(vol.shape[0]) - 1
                    self.txtMissingStart.setText("0")
                    self.txtMissingEnd.setText(str(zmax))
                    # Remove overlay if present
                    if self._overlay_layer and self._overlay_layer in self.viewer.layers:
                        self.viewer.layers.remove(self._overlay_layer)
                        self._overlay_layer = None
                    # Remove boundary layer if present
                    if self._boundary_layer and self._boundary_layer in self.viewer.layers:
                        self.viewer.layers.remove(self._boundary_layer)
                        self._boundary_layer = None
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
    # Implant angle (LEFT / RIGHT) with rotation+save+reload (Z-axis)
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

    def _rotate_and_save_z(self, angle_deg: float, tag: str):
        side_path = self._path_side_volume()
        if not side_path or not os.path.isfile(side_path):
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
            patient_dir = os.path.dirname(side_path)
            volume = np.load(side_path)  # 3D array (Z, Y, X)
            if volume.ndim != 3:
                QtWidgets.QMessageBox.warning(None, "Unsupported",
                                              f"side_volume.npy has ndim={volume.ndim}, expected 3.")
                return

            # Rotate around Z-axis as requested (axes=(1,0))
            rotated_volume = ndimage.rotate(volume, angle=float(angle_deg), axes=(1, 0), reshape=True)

            # Create a directory for saving the rotated slices
            rotated_slices_dir = os.path.join(patient_dir, f'rotated_slices_{tag}')
            os.makedirs(rotated_slices_dir, exist_ok=True)

            # Save each slice as PNG
            for i in range(rotated_volume.shape[0]):
                slice_img = rotated_volume[i]

                if np.min(slice_img) == np.max(slice_img):
                    slice_img_normalized = np.zeros_like(slice_img, dtype=np.uint8)
                else:
                    slice_img_normalized = (255 * (slice_img - np.min(slice_img)) /
                                            (np.max(slice_img) - np.min(slice_img))).astype(np.uint8)

                from PIL import Image  # safe even if already imported
                img = Image.fromarray(slice_img_normalized)
                img.save(os.path.join(rotated_slices_dir, f'slice_{i:03d}.png'))

            # Save the rotated volume
            save_path = self._path_rotated_volume()  # rotated_expanded_zaxis.npy
            np.save(save_path, rotated_volume)

            # Reload into Napari
            self._load_into_viewer(rotated_volume, name=f"CBCT (rotated Z-axis, {tag})")

            # When a new volume is loaded, clear any old overlays/boundaries
            if self._overlay_layer and self._overlay_layer in self.viewer.layers:
                self.viewer.layers.remove(self._overlay_layer)
                self._overlay_layer = None
            if self._boundary_layer and self._boundary_layer in self.viewer.layers:
                self.viewer.layers.remove(self._boundary_layer)
                self._boundary_layer = None

            # Update default start/end for the new rotated volume
            zmax = int(rotated_volume.shape[0]) - 1
            self.txtMissingStart.setText("0")
            self.txtMissingEnd.setText(str(zmax))

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
        self._rotate_and_save_z(angle_deg=self.implant_angle_left_deg, tag="left")

    def _set_implant_angle_right(self):
        angle = self._parse_angle_from_box()
        if angle is None:
            return
        right_angle = (360.0 - angle) % 360.0
        self.implant_angle_right_deg = right_angle
        self.implant_angle_deg = right_angle
        self.lblImplantAngle.setText(f"Current implant angle (Right): {self.implant_angle_right_deg:.1f}°")
        self._rotate_and_save_z(angle_deg=self.implant_angle_right_deg, tag="right")

    # ----------------------------
    # Missing teeth slice (manual/auto)
    # ----------------------------
    def _set_missing_slice_manual(self):
        # Ensure we have a base CBCT Image layer (do NOT assume index 0)
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
            QtWidgets.QMessageBox.warning(None, "Range error",
                                          f"Slices must be within [0, {zmax}].")
            return
        if e_val < s_val:
            QtWidgets.QMessageBox.warning(None, "Range error", "End slice must be >= start slice.")
            return

        self.missing_slice_start = s_val
        self.missing_slice_end = e_val

        # Build a labels volume with ONLY the first/last slices colored
        labels = np.zeros(vol_shape, dtype=np.uint8)
        labels[s_val, :, :] = 2
        labels[e_val, :, :] = 3

        # Remove existing overlay if present
        if self._overlay_layer and self._overlay_layer in self.viewer.layers:
            self.viewer.layers.remove(self._overlay_layer)
            self._overlay_layer = None
        # Remove existing boundary lines if present
        if self._boundary_layer and self._boundary_layer in self.viewer.layers:
            self.viewer.layers.remove(self._boundary_layer)
            self._boundary_layer = None

        # Add labels overlay
        self._overlay_layer = self.viewer.add_labels(labels, name="Missing Slices", opacity=1.0)

        # Color map: endpoints solid red only
        try:
            self._overlay_layer.color = {
                2: (1.0, 0.0, 0.0, 0.95),  # first slice plane = solid red
                3: (1.0, 0.0, 0.0, 0.95),  # last slice plane  = solid red
            }
        except Exception:
            pass

        # Boundary rectangle outlines on the first & last slice (for crisp edges)
        try:
            Y, X = vol_shape[1], vol_shape[2]
            rect_yx = np.array([[0, 0], [0, X - 1], [Y - 1, X - 1], [Y - 1, 0], [0, 0]], dtype=float)

            shapes = []
            for z in (s_val, e_val):
                zcol = np.full((rect_yx.shape[0], 1), float(z))
                coords_zyx = np.concatenate([zcol, rect_yx], axis=1)  # (z, y, x)
                shapes.append(coords_zyx)

            # solution 3: thin polylines, not polygons; disable picking
            self._boundary_layer = self.viewer.add_shapes(
                shapes,
                shape_type="path",
                edge_color="red",
                edge_width=1,
                face_color="transparent",
                name="Missing Slice Boundaries",
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

        # bury overlays behind the CBCT image
        self._bury_overlays(base)

        QtWidgets.QMessageBox.information(None, "Set",
                                          f"Missing teeth slice set to [{s_val}, {e_val}].\n"
                                          f"Solid red planes drawn on first and last slice only (buried inside CBCT).")

    # ----------------------------
    # Auto classification helpers
    # ----------------------------
    def _load_classifier_model(self):
        """Load TorchScript or state_dict, rebuilding EfficientNet-B0 2-logit (2.5D) if needed."""
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

        # 1) Try TorchScript first
        try:
            m = torch.jit.load(model_path, map_location=self._clf_device)
            m.eval()
            self._clf_model = m
            return self._clf_model
        except Exception:
            pass  # fall back

        # 2) Raw object / checkpoint dict / state_dict
        obj = torch.load(model_path, map_location=self._clf_device)

        if hasattr(obj, "forward"):  # already a Module
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

        # strip common prefixes
        def strip_prefix(state_dict, prefixes=("net.", "module.", "model.")):
            new_sd = OrderedDict()
            for k, v in state_dict.items():
                nk = k
                for p in prefixes:
                    if nk.startswith(p):
                        nk = nk[len(p):]
                new_sd[nk] = v
            return new_sd

        sd = strip_prefix(sd)

        # rebuild EfficientNet-B0 2-logit with in_ch=2k+1
        in_ch = int(2 * self._clf_k + 1)
        base = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.IMAGENET1K_V1)

        old_first: nn.Conv2d = base.features[0][0]
        new_first = nn.Conv2d(in_ch, old_first.out_channels,
                              kernel_size=old_first.kernel_size,
                              stride=old_first.stride,
                              padding=old_first.padding,
                              bias=False)
        with torch.no_grad():
            w = old_first.weight  # [out,3,k,k]
            w_mean = w.mean(dim=1, keepdim=True)
            if in_ch == 3:
                new_first.weight.copy_(w)
            else:
                new_first.weight.copy_(w_mean.repeat(1, in_ch, 1, 1))
        base.features[0][0] = new_first

        in_feat = base.classifier[1].in_features
        base.classifier[1] = nn.Linear(in_feat, 2)

        # try strict, then filtered non-strict
        try:
            base.load_state_dict(sd, strict=True)
        except Exception as e_strict:
            model_keys = set(base.state_dict().keys())
            filtered = OrderedDict((k, v) for k, v in sd.items() if k in model_keys)
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
        """
        Preprocess one 2D slice like your training pipeline:
        - select vertical band (band_frac)
        - optional clahe
        - resize to img_size
        - normalize to [0,1], float32
        Returns (H,W) float32.
        """
        img = slc2d
        if img.dtype != np.uint8:
            vmin, vmax = float(np.min(img)), float(np.max(img))
            if vmax > vmin:
                img = ((img - vmin) / (vmax - vmin) * 255.0).astype(np.uint8)
            else:
                img = np.zeros_like(img, dtype=np.uint8)

        # band crop
        h = img.shape[0]
        y0 = int(h * self._clf_band_frac[0]); y1 = int(h * self._clf_band_frac[1])
        y0 = max(0, min(y0, h-1)); y1 = max(1, min(y1, h))
        if y1 <= y0: y0, y1 = 0, h
        roi = img[y0:y1, :]

        # top cap if needed
        if self._clf_top_cap_frac > 0:
            cap = int(roi.shape[0] * self._clf_top_cap_frac)
            if cap > 0:
                roi[:cap, :] = 0

        # CLAHE (disabled to match training unless you change _clf_use_clahe)
        if self._clf_use_clahe:
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
            roi = clahe.apply(roi)

        # resize
        roi = cv2.resize(roi, (self._clf_img_size, self._clf_img_size), interpolation=cv2.INTER_AREA)

        # normalize to [0,1]
        roi = roi.astype(np.float32)
        mn, mx = float(roi.min()), float(roi.max())
        roi = (roi - mn) / (mx - mn + 1e-6)
        return roi

    def _slice_stack_2p5d(self, vol: np.ndarray, idx: int) -> np.ndarray:
        """
        Build a 2.5D stack centered at idx with context k: (C,H,W) float32
        C = 2*k+1
        """
        k = int(self._clf_k)
        Z = vol.shape[0]
        chans = []
        for d in range(-k, k+1):
            j = min(max(idx + d, 0), Z - 1)
            s = vol[j]
            chans.append(self._prep_single_slice(s))
        stack = np.stack(chans, axis=0)  # (C,H,W)
        return stack

    def _classify_volume_missing_range(self, vol: np.ndarray):
        """
        Run model over all slices (0..Z-1), predict class argmax,
        and return (start_idx, end_idx) for the configured missing label.
        If none found, optionally flip label as a fallback.
        """
        import torch

        model = self._load_classifier_model()  # may raise with helpful message
        device = torch.device(self._clf_device)
        model.to(device)

        Z = vol.shape[0]
        preds = []
        with torch.inference_mode():
            for i in range(Z):
                x_np = self._slice_stack_2p5d(vol, i)          # (C,H,W)
                x = torch.from_numpy(x_np).unsqueeze(0)        # (1,C,H,W)
                x = x.to(device, dtype=torch.float32)
                logits = model(x)                               # (1,2)
                pred = int(torch.argmax(logits, dim=1).item())  # 0 or 1
                preds.append(pred)

        # quick debug: count classes
        cnt0 = sum(1 for p in preds if p == 0)
        cnt1 = Z - cnt0
        print(f"[Classifier] Pred counts -> class0: {cnt0}, class1: {cnt1}, Z={Z}")

        # expected missing label
        miss_lbl = int(self._clf_missing_label)

        indices_miss = [i for i, p in enumerate(preds) if p == miss_lbl]
        if not indices_miss:
            # Fallback: maybe labels were exported swapped (rare but happens)
            alt_lbl = 1 - miss_lbl
            indices_alt = [i for i, p in enumerate(preds) if p == alt_lbl]
            if indices_alt:
                QtWidgets.QMessageBox.information(
                    None, "Auto-detect",
                    "Model predicted no slices with the configured missing label.\n"
                    "Assuming the missing label is swapped; using the other class."
                )
                return min(indices_alt), max(indices_alt), preds
            # truly none
            return None, None, preds

        return min(indices_miss), max(indices_miss), preds

    def _get_missing_slice_automatically(self):
        """
        Classify slices from 0..Z-1 using classification.pt.
        Start = lowest index with 'missing' class. End = highest index with 'missing' class.
        Then draw the planes (buried inside CBCT).
        """
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

        # update UI and draw planes using the manual routine
        self.txtMissingStart.setText(str(start_idx))
        self.txtMissingEnd.setText(str(end_idx))
        self._set_missing_slice_manual()

    # ----------------------------
    # CSV-backed Implant Database (Tab index 4)
    # ----------------------------
    def _db_csv_path(self):
        try:
            base = os.path.dirname(os.path.abspath(sys.argv[0]))
        except Exception:
            base = os.getcwd()
        return os.path.join(base, "implant_database.csv")

    def _db_set_rows(self, rows):
        """rows: list of (id, dia, hgt) strings/floats"""
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
            # Seed with initial database (parsed from your table image)
            rows = [
                # --- GS series ---
                ("GS2M35080", 3.5, 8.5),
                ("GS2M35100", 3.5, 10),
                ("GS2M35110", 3.5, 11.5),
                ("GS2M35130", 3.5, 13),
                ("GS2M35150", 3.5, 15),
                ("GS2R40070", 4.0, 7),
                ("GS2R40080", 4.0, 8.5),
                ("GS2R40100", 4.0, 10),
                ("GS2R40110", 4.0, 11.5),
                ("GS2R40130", 4.0, 13),
                ("GS2R40150", 4.0, 15),
                ("GS2W45070", 4.5, 7),
                ("GS2W45080", 4.5, 8.5),
                ("GS2W45100", 4.5, 10),
                ("GS2W45110", 4.5, 11.5),
                ("GS2W45130", 4.5, 13),
                ("GS2W45150", 4.5, 15),
                ("GS2W50070", 5.0, 7),
                ("GS2W50080", 5.0, 8.5),
                ("GS2W50100", 5.0, 10),
                ("GS2W50110", 5.0, 11.5),
                ("GS2W50130", 5.0, 13),
                ("GS2W50150", 5.0, 15),
                # --- MSP series (partial) ---
                ("MSP25103R", 2.0, 10),
                ("MSP25104R", 2.0, 10),
                ("MSP25113R", 2.5, 11.5),
                ("MSP25114R", 2.5, 11.5),
                ("MSP25133R", 2.5, 13),
                ("MSP25134R", 2.5, 13),
                ("MSP25153R", 2.5, 15),
                ("MSP25154R", 2.5, 15),
                ("MSP30103R", 3.0, 10),
                ("MSP30104R", 3.0, 10),
                ("MSP30113R", 3.0, 11.5),
                ("MSP30114R", 3.0, 11.5),
                ("MSP30133R", 3.0, 13),
                ("MSP30134R", 3.0, 13),
                ("MSP30153R", 3.0, 15),
                ("MSP30154R", 3.0, 15),
            ]
            # Immediately save the seeded CSV
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

        # Prefer updating the selected row; fallback to ID match
        idxs = self.tableView.selectionModel().selectedRows()
        r = idxs[0].row() if idxs else self._db_find_row_by_id(rid)
        if r == -1:
            QtWidgets.QMessageBox.warning(None, "Update", "Select a row or ensure the ID exists.")
            return

        # If ID text changed to a new one that collides, block
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
        """Normalize a (H,W) slice to uint8 [0..255], then stack to (H,W,3) BGR for OpenCV/YOLO."""
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
        # 1) Get current volume + range
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

        # parse start/end from UI
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

        # 2) Where to save results
        patient_dir = self._cbct_dir()
        if not patient_dir:
            QtWidgets.QMessageBox.warning(None, "Folder", "Select a valid patient folder first.")
            return

        ab_dir = os.path.join(patient_dir, "segmented_AB")
        mc_dir = os.path.join(patient_dir, "segmented_MC")
        self._clear_and_prepare_dir(ab_dir)
        self._clear_and_prepare_dir(mc_dir)

        # 3) Find model (same folder as this script)
        model_path = self._find_segmentation_model_path()
        if not model_path:
            QtWidgets.QMessageBox.critical(None, "Model not found",
                                           "segmentation.pt not found in the same folder as main.py.")
            return

        # 4) Build inputs (resize each slice to 640x640 as in your reference)
        H, W = vol.shape[1], vol.shape[2]  # original size (y, x)
        imgs_640 = []
        src_indices = list(range(s_val, e_val + 1))
        for idx in src_indices:
            rgb = self._slice_to_rgb(vol[idx])                    # (H,W,3) uint8
            resized = cv2.resize(rgb, (640, 640), interpolation=cv2.INTER_AREA)
            imgs_640.append(resized)

        # 5) Run YOLOv8 (segment)
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

        # 6) Postprocess & save binary masks per slice
        import torch
        previous_mc_mask = None  # fallback for MC

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
                        # AB union
                        if ab_idx.numel() > 0:
                            ab_stack = mdata[ab_idx].cpu().numpy()
                            ab_union = (ab_stack > 0.5).any(axis=0).astype(np.uint8) * 255
                            # fill contours
                            _, ab_thresh = cv2.threshold(ab_union, 127, 255, cv2.THRESH_BINARY)
                            cnts, _ = cv2.findContours(ab_thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                            ab_filled_640 = np.zeros_like(ab_union)
                            cv2.drawContours(ab_filled_640, cnts, -1, 255, thickness=cv2.FILLED)
                            ab_bin = cv2.resize(ab_filled_640, (W, H), interpolation=cv2.INTER_NEAREST)

                        # MC union or fallback
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

            # Save images
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
