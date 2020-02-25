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
cell_state[0, N//2] = 1
mod = SourceModule('''
  __global__ void cell_step(int *A, int rule, int step, int n)
  {
    const int idx = threadIdx.x;
    const int a = A[(step-1)*n + idx - 1];
    const int b = A[(step-1)*n + idx];
    const int c = A[(step-1)*n + idx + 1];
    const int current_pattern = (a<<2) | (b<<1) | c;

    const int p = (rule & (1 << current_pattern)) >> current_pattern;
    A[step * n + idx] = p;
  }
  ''')
func = mod.get_function('cell_step')

step = 0
rule = 0
def update():
    global step, rule
    step += 1

    if step == N:
        rule = (rule + 1) % 256
        step = 1

    func(cuda.InOut(cell_state), np.int32(rule), np.int32(step), np.int32(N), block=(N, 1, 1))
    image = np.array(Image.fromarray(cell_state * 255).resize(display_size, Image.NEAREST))
    return image, f'Cellular Automate: Rule {rule}'

viewer = Viewer(update, display_size)
viewer.start()