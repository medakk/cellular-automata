from fb import Viewer
import numpy as np
import pycuda.driver as cuda
import pycuda.autoinit
import pycuda.gpuarray as gpuarray
from pycuda.compiler import SourceModule
from PIL import Image

N = 200
display_size = (900, 900)

cell_state = np.zeros((N, N), dtype=np.int32)
cell_state[0, N//2] = 0x80000001
mod = SourceModule('''
  __global__ void cell_step(int *A, int rule, int step, int n)
  {
    const int idx = threadIdx.x + 1;
    const int is_active = A[(step-1)*n + idx] & 0x80000000;
    if(is_active == 0) {
      A[step*n + idx] = A[(step-1)*n + idx];
      return;
    }

    const int a = A[(step-1)*n + idx - 1] & 1;
    const int b = A[(step-1)*n + idx] & 1;
    const int c = A[(step-1)*n + idx + 1] & 1;
    const int current_pattern = (a<<2) | (b<<1) | c;

    const int p          = ((rule & 0x00ff) & (1 << current_pattern)) >> current_pattern;
    const int new_active = (((rule & 0xff00)>>8) & (1 << current_pattern)) >> current_pattern;
    if(new_active == 0) {
      A[step * n + idx + 1] |= 0x80000000;
    } else {
      A[step * n + idx - 1] |= 0x80000000;
    }
    A[step * n + idx] = p;
  }
  ''')
func = mod.get_function('cell_step')

step = 0
rule = 0b1100011000100011
rule = 0b0011100100100011
def update():
    global step, rule
    step += 1

    if step == N:
        step = 1

    func(cuda.InOut(cell_state), np.int32(rule), np.int32(step), np.int32(N), block=(N-1, 1, 1))
    image = cell_state * rule
    image = np.array(Image.fromarray(image).resize(display_size, Image.NEAREST))
    return image

viewer = Viewer(update, display_size)
viewer.start()