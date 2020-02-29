import numpy as np

from fb import Viewer

def update():
    image = np.random.random((600, 600, 3)) * 255.0
    image[:,:200,0] = 255.0
    image[:,200:400,1] = 255.0
    image[:,400:,2] = 255.0
    return image.astype('uint8')

viewer = Viewer(update, (600, 600))
viewer.start()