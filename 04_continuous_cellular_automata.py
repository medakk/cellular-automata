import random
import time

import numpy as np
import pycuda.driver as cuda
import pycuda.autoinit
import pycuda.gpuarray as gpuarray
from pycuda.compiler import SourceModule
from PIL import Image

from fb import Viewer

N = 400
display_size = (900, 900)

cell_state = np.zeros((N, N), dtype=np.float32)
cell_state[0, N//2] = 1
mod = SourceModule('''
  __device__ float frac(float x) {
    return x - floorf(x);
  }
  __global__ void cell_step(float *A, float rule, int step, int n) {
    const int idx = threadIdx.x;
    const float a = A[(step-1)*n + idx - 1];
    const float b = A[(step-1)*n + idx];
    const float c = A[(step-1)*n + idx + 1];
    const float p = (a + b + c) / 3.0;

    A[step * n + idx] = frac(p + rule);
  }
  ''')
func = mod.get_function('cell_step')

step = 0
rule = 0
last_time = time.time()
def update():
    global step, rule, last_time
    step += 1
    curr_time = time.time()
    dt = curr_time - last_time
    last_time = curr_time
    fps = 1 / dt

    if step == N:
        rule = random.random()
        step = 1

    func(cuda.InOut(cell_state), np.float32(rule), np.int32(step), np.int32(N), block=(N, 1, 1))

    image = np.stack((cell_state*255.0,)*3, axis=-1).astype('uint8')
    image = np.array(Image.fromarray(image).resize(display_size, Image.BILINEAR))
    viewer.set_title(f'Cellular Automate: Rule {rule:.2f} [{fps:.1f} FPS]')
    return image

viewer = Viewer(update, display_size)
viewer.start()