from grid import Grid2D

def init_grid(n=None, inp=None):
    if n is not None and inp is not None:
        return Grid2D(n, inp)
        
    nx = 2
    ny = 2
    x = [0, 1, 0, 1]
    y = [0, 0, 1, 1]
    return Grid2D((nx,ny), (x,y))
