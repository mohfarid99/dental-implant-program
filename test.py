from vispy import scene, app

canvas = scene.SceneCanvas(keys='interactive', show=True)
view = canvas.central_widget.add_view()
import numpy as np
vol = np.random.rand(64,64,64).astype('float32')
volume = scene.visuals.Volume(vol, parent=view.scene)
app.run()
