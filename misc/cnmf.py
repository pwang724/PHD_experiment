import os
import glob
from skimage import io
import time
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np
from _CONSTANTS.config import Config
import _CONSTANTS.conditions as conditions
import tools.file_io

mouse = 'M232_ofc'
date = '05-30-2017'
reload = True

# load ROIs
data_directory = Config().LOCAL_DATA_PATH
data_path = os.path.join(data_directory, 'registration','CNMF', mouse, date)

if not reload:
    import matlab.engine
    eng = matlab.engine.start_matlab()
    ds = glob.glob(os.path.join('I:\IMPORTANT DATA\STORAGE_DATA', mouse + '_' + date + '*.mat'))
    assert len(ds) == 1
    temp = eng.load(ds[0])
    obj = list(temp.values())[0]
    obj_name = "obj"
    eng.workspace[obj_name] = obj
    signal = eng.eval(obj_name + ".roiCell('norm')")
    signal = np.asarray(signal).squeeze()
    roi = eng.eval(obj_name + ".roi")
    roi = np.asarray(roi).squeeze()
    roiT = roi.transpose(2, 0, 1)
    tools.file_io.save_numpy(data_path, 'roi', roiT)
    tools.file_io.save_numpy(data_path, 'signal', signal)
else:
    roiT = tools.file_io.load_numpy(os.path.join(data_path,'roi.npy'))
    signal = tools.file_io.load_numpy(os.path.join(data_path,'signal.npy'))
    print(roiT.shape)
    print(signal.shape)

# load images
d = os.path.join('I:\IMPORTANT DATA\DATA_X', mouse, date)
start_frame = 1
end_frame = 50

start_time = time.time()

raw_dir = glob.glob(os.path.join(d, '*z'))
d_raw = glob.glob(os.path.join(raw_dir[0], '*.tif'))[0]
raw = io.imread(d_raw)
print(raw.shape)

# d_bg = os.path.join(d, 'bg.tif')
# bg = io.imread(d_bg)

# d_residual = os.path.join(d, 'res.tif')
# residual = io.imread(d_residual)

# d_denoised = os.path.join(d, 'den_bg.tif')
# denoised = io.imread(d_denoised)

print('finished reading : {} s'.format(time.time() - start_time))

plt.figure(figsize=(15,3))
plt.plot(signal[0])

element_mult = np.multiply(raw, roiT[0])
raw_trace = np.mean(element_mult, axis=(1,2))
plt.figure(figsize=(15,3))
plt.plot(raw_trace)


#
#
#
#
# def ani_frame(fps = 30):
#     def plot_weights():
#         fig, axs = plt.subplots(2, 2, figsize=(8, 8))
#         plt.style.use('dark_background')
#
#         im0 = axs[0].imshow(raw[0], cmap='gray')
#         axs[0].axis('tight')
#         axs[0].axis('off')
#         axs[0].set_title('raw')
#         for loc in ['bottom', 'top', 'left', 'right']:
#             axs.spines[loc].set_visible(False)
#         return fig, im0
#
#     def update_img(n):
#         im0.set_data(raw[n,:,:])
#         return im0
#
#     fig, im0 = plot_weights()
#     ani = animation.FuncAnimation(fig, update_img, end_frame, interval=1)
#     writer = animation.writers['ffmpeg'](fps=fps)
#     dpi = 200
#     ani.save('help.mp4', writer=writer, dpi=dpi)
#     return ani
#
# ani_frame(fps=30)