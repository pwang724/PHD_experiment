from skimage import io
import os
import numpy as np
import PIL.Image as Image

d = r'C:\Users\P\Desktop\movies'
x = ['pir.tif', 'ofc.tif', 'mpfc.tif']
images = []
multiplier = [16, 10, 10]
slice = [[300, 600],[0, 300],[650,950]]
for i, n in enumerate(x):
    im = io.imread(os.path.join(d,n))
    im = im[slice[i][0]:slice[i][1],:256,:256]
    im *= multiplier[i] # uint12 to uint16 conversion
    images.append(im)
    print(im.shape)

tiled_images = np.concatenate(images, axis=2)


out = os.path.join(d, 'stack.tif')
io.imsave(out, tiled_images)