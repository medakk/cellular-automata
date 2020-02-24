import taichi as ti

ti.init(arch=ti.cuda) # Run on GPU by default

n = 100
pixels = ti.var(dt=ti.i32, shape=(n, n))

@ti.kernel
def paint(t: ti.i32):
    for i, j in pixels:
        pixels[i, j] = 1

gui = ti.GUI("Automata", (n, n))
for t in range(n):
    paint(t)
    gui.set_image(pixels)
    gui.show()