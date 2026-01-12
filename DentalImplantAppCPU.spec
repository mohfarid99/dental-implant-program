# -*- mode: python ; coding: utf-8 -*-

import os
from PyInstaller.utils.hooks import collect_submodules, collect_data_files

block_cipher = None

# =====================================================
# 1. HIDDEN IMPORTS (NAPARI + TORCH + ULTRALYTICS)
# =====================================================
hiddenimports = (
    collect_submodules("napari") +
    collect_submodules("vispy") +
    collect_submodules("skimage") +
    collect_submodules("scipy") +
    collect_submodules("cv2") +
    collect_submodules("torch") +
    collect_submodules("ultralytics") +
    collect_submodules("qtpy") +
    collect_submodules("napari_plugin_engine") +
    collect_submodules("npe2") +
    collect_submodules("PIL")
)

# =====================================================
# 2. DATA FILES
# =====================================================
datas = []

# Napari & VisPy assets
datas += collect_data_files("napari")
datas += collect_data_files("vispy")

# Torch CPU runtime data
datas += collect_data_files("torch")
datas += collect_data_files("torchvision")

# Model files (harus ada di folder exe)
datas += [
    ("classification.pt", "."),
    ("segmentation.pt", "."),
]

# Qt Plugins (WAJIB untuk PyQt5)
datas += [
    (
        os.path.join(
            os.environ["CONDA_PREFIX"],
            "Library",
            "plugins"
        ),
        "PyQt5/Qt/plugins"
    )
]

# =====================================================
# 3. ANALYSIS
# =====================================================
a = Analysis(
    ["main46CPU.py"],
    pathex=[],
    binaries=[],
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    runtime_hooks=[],
    excludes=[
        "torch.cuda",
        "cupy",
        "jax",
        "tensorflow",
        "numba.cuda",
        "pytorch_lightning",
        "torchvision.io.image",
        "torchvision.io._image",
        "torchvision.image",
        "torchvision._image",
        "PyQt6",
        "PySide6",
        "PySide2",
    ],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
)

# =====================================================
# 4. PYZ
# =====================================================
pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

# =====================================================
# 5. EXE (WINDOWS GUI, NO CONSOLE)
# =====================================================
exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.zipfiles,
    a.datas,
    [],
    name="DentalImplantApp",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=False,   # ⬅️ GUI ONLY
)

# =====================================================
# 6. COLLECT
# =====================================================
coll = COLLECT(
    exe,
    a.binaries,
    a.zipfiles,
    a.datas,
    strip=False,
    upx=True,
    name="DentalImplantApp",
)
