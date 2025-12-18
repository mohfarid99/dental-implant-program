# -*- mode: python ; coding: utf-8 -*-

from PyInstaller.utils.hooks import collect_all

hiddenimports = []
datas = []
binaries = []

# -------------------------
# napari
# -------------------------
napari_datas, napari_bins, napari_hidden = collect_all('napari')
datas += napari_datas
binaries += napari_bins
hiddenimports += napari_hidden

# -------------------------
# vispy
# -------------------------
vispy_datas, vispy_bins, vispy_hidden = collect_all('vispy')
datas += vispy_datas
binaries += vispy_bins
hiddenimports += vispy_hidden

# -------------------------
# torch
# -------------------------
torch_datas, torch_bins, torch_hidden = collect_all('torch')
datas += torch_datas
binaries += torch_bins
hiddenimports += torch_hidden

# -------------------------
# ultralytics
# -------------------------
ultra_datas, ultra_bins, ultra_hidden = collect_all('ultralytics')
datas += ultra_datas
hiddenimports += ultra_hidden

# -------------------------
# YOLO model files
# -------------------------
datas += [
    ('segmentation.pt', '.'),
    ('classification.pt', '.'),
]

a = Analysis(
    ['main42.py'],
    pathex=[],
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    runtime_hooks=[],
    excludes=['PyQt6'],  # 🔥 REQUIRED
    noarchive=False,
)

pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.datas,
    [],
    name='DentalImplantApp',
    console=False,
)
