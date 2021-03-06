import random
import time

import numpy as np
import pycuda.driver as cuda
import pycuda.autoinit
import pycuda.gpuarray as gpuarray
from pycuda.compiler import SourceModule
from PIL import Image

from fb import Viewer

N = 128*2
display_size = (1024, 1024)

cell_state = np.zeros((N, N), dtype=np.float32)
cell_state[0, N//2] = 1
mod = SourceModule('''
  __device__ float frac(float x) {
    return x - floorf(x);
  }
  __global__ void cell_step(float *A, float rule, int step, int n) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if(idx < 1 || idx >= n)
      return;

    const float a = A[(step-1)*n + idx - 1];
    const float b = A[(step-1)*n + idx];
    const float c = A[(step-1)*n + idx + 1];
    const float p = (a + b + c) / 3.0;

    A[step * n + idx] = frac(p + rule);
  }
  ''')
func = mod.get_function('cell_step')

step = 0
rule = random.random()
last_time = time.time()
def update(steps=10):
    global step, rule, last_time

    for _ in range(steps):
      step += 1
      if step >= N:
          # viewer.save_frame(f'out/rule_{rule:0.2f}.png')
          rule = random.random()
          step = 1
          break

      func(cuda.InOut(cell_state), np.float32(rule), np.int32(step), np.int32(N),
                block=(1024, 1, 1),
                grid=(max(1, N//1024), 1))


    image = np.stack((cell_state*255.0,)*3, axis=-1).astype('uint8')
    image = np.array(Image.fromarray(image).resize(display_size, Image.BILINEAR))
    viewer.set_title(f'Cellular Automate: Rule {rule:.2f}')
    return image

viewer = Viewer(update, display_size)
viewer.start()