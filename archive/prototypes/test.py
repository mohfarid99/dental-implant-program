import os
os.environ["QT_OPENGL"] = "angle"
os.environ["VISPY_FORCE_GL"] = "angle"
os.environ["VISPY_USE_APP"] = "pyqt5"

from PyQt5 import QtCore, QtGui, QtWidgets
import sys, os
qt_bin = os.path.join(os.path.dirname(QtCore.__file__), "Qt", "bin")
if hasattr(os, "add_dll_directory"):
    os.add_dll_directory(qt_bin)

fmt = QtGui.QSurfaceFormat()
fmt.setRenderableType(QtGui.QSurfaceFormat.OpenGLES)
fmt.setVersion(3, 0)
QtGui.QSurfaceFormat.setDefaultFormat(fmt)

from vispy import app, scene, sys_info
print(sys_info())

canvas = scene.SceneCanvas(keys='interactive', show=True)
view = canvas.central_widget.add_view()
view.camera = 'turntable'

axis = scene.visuals.XYZAxis(parent=view.scene)
app.run()
