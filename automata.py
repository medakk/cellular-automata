import taichi as ti

ti.init(arch=ti.cuda) # Run on GPU by default

n = 200
cells = ti.var(dt=ti.i32, shape=(n, n))

img_n = 700
pixels = ti.var(dt=ti.f32, shape=(img_n, img_n))

@ti.kernel
def gen_image():
    for i, j in pixels:
        x = (n * i) // img_n
        y = (n * j) // img_n
        pixels[i, j] = cells[x, y]

@ti.kernel
def init_cells():
    for i, j in cells:
        cells[i, j] = 0

    cells[0, n//2] = 1

@ti.kernel
def paint(t: ti.i32):
    for i in range(1, n-1):
        a = cells[t-1, i-1] 
        b = cells[t-1, i] 
        c = cells[t-1, i+1] 


        p = 0
        if a == 1:
            if b == 0 and c == 0:
                p = 1
            else:
                p = 0
        else:
            if b == 0 and c == 0:
                p = 0
            else:
                p = 1
        cells[t, i] = p

gui = ti.GUI("Automata", (img_n, img_n))

init_cells()
for i in range(1, n):
    paint(i)
    gen_image()
    gui.set_image(pixels)
    gui.show()

gui.set_image(pixels)
gui.wait_key()