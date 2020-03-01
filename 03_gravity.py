import random

import numpy as np
import pycuda.driver as cuda
import pycuda.autoinit
import pycuda.gpuarray as gpuarray
from pycuda.compiler import SourceModule
from PIL import Image

from fb import Viewer

N = 200
display_size = (900, 900)

mod = SourceModule('''
    __global__ void cell_step(int *in, int *out, int n)
    {
        int i = threadIdx.x;

        for(int j=0; j<n; j++) {
            int b = in[i*n + j];
            out[i*n + j] = in[i*n + j];

            if(j != 0) {
                int a = in[i*n + j - 1];
                if(a == 1 && b == 0) {
                    out[i*n + j] = 1;
                }
            }

            if(j != (n-1)) {
                int c = in[i*n + j + 1];
                if(b == 1 && c == 0) {
                    out[i*n + j] = 0;
                }
            }
        }
    }

    __global__ void circle(int *out, int x, int y, int r, int set_to, int n) {
        int i = threadIdx.x;

        for(int j=0; j<n; j++) {
            int dx = i - x;
            int dy = j - y;
            if(dx * dx + dy * dy < r * r) {
                out[i*n + j] = set_to;
            }
        }
    }
  ''')
step = mod.get_function('cell_step')
circle = mod.get_function('circle')

A = (np.random.random((N, N))<0.35).astype(np.int32)
def update():
    global A

    step(cuda.In(A), cuda.Out(A), np.int32(N), block=(N, 1, 1))

    if random.random() < 0.1:
        x = np.int32(random.randrange(0, N))
        y = np.int32(random.randrange(0, N))
        r = np.int32(random.randrange(5, 12))
        set_to = np.int32(random.randrange(0, 2))
        circle(cuda.InOut(A), x, y, r, set_to, np.int32(N), block=(N, 1, 1))

    image = np.array(Image.fromarray(A * 255).resize(display_size, Image.NEAREST))
    return image

viewer = Viewer(update, display_size)
viewer.start()