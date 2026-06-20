import numpy as np

class Grid2D:
    def __init__(self, filename: str):
        self.n, self.inp = self._read_file(filename)
        self.coords = self._define_coords()
        self.cell_volumes = self._calculate_volumes()
        self.cell_centers = self._calculate_cell_centers()
        self.cell_surface_areas = self._calculate_cell_surface_areas()
        self.airfoil_start, self.airfoil_end = self._get_airfoil_limits()
    
    def __init__(self, n: tuple, inp: tuple):
        self.n = n
        self.inp = inp
        self.coords = self._define_coords()
        self.cell_volumes = self._calculate_volumes()
        self.cell_centers = self._calculate_cell_centers()
        self.cell_surface_areas = self._calculate_cell_surface_areas()
        self.airfoil_start, self.airfoil_end = self._get_airfoil_limits()

    def _read_file(self, filename: str):
        '''
        this is a very hacky way of gathering the coordinates - initialize
        empty arrays and populate them dynamically by appending
        coordinate information. Removes the need to know array size
        info apriori, but is usually bad for big files (~50K+ lines)
        '''

        # again we use 1D arrays because they are slightly more efficient, but 
        # it's easy enough to make them 2D
        x = []
        y = []

        # open file and read line by line to assign to array
        with open(filename,"r") as fp:
            [nx, ny] = [int(m) for m in fp.readline().strip("\n").split(", ")]
            for line in fp:
                coords = line.strip("\n").split(", ")
                x.append(float(coords[0]))
                y.append(float(coords[1]))

        # note you can also use meshgrid(x,y) to create a mesh out of this!

        # when doing file I/O using "with open ..." you don't need to do file.close()
        # python automatically closes the file once the I/O processes are finished

        return (nx,ny), (x,y)
    
    def _define_coords(self):
        nx,ny = self.n
        x_inp, y_inp = self.inp
        
        x = np.zeros((nx+2, ny+2))
        y = np.zeros((nx+2, ny+2))
        x[1:nx+1, 1:ny+1] = np.asarray(x_inp).reshape((ny,nx)).T
        y[1:nx+1, 1:ny+1] = np.asarray(y_inp).reshape((ny,nx)).T

        y[1:nx+1, 0] = 2*y[1:nx+1,1] - y[1:nx+1,2]
        x[1:nx+1, 0] = x[1:nx+1, 1]

        y[0, :] = y[1, :]
        x[0, :] = 2*x[1, :] - x[2, :]

        y[nx+1, :] = y[nx, :]
        x[nx+1, :] = 2*x[nx, :] - x[nx-1, :]

        return (x,y)
    
    def _calculate_volumes(self):
        nx,ny = self.n
        x,y = self.coords

        vol = np.zeros((nx+1, ny+1))
        vol = 0.5*((x[1:,1:] - x[:-1,:-1]) * (y[:-1, 1:] - y[1:, :-1])
                    - (x[:-1, 1:] - x[1:, :-1]) * (y[1:, 1:] - y[:-1, :-1]))
        return vol
    
    def _calculate_cell_centers(self):
        x,y = self.coords

        xc = 0.25*(x[:-1, :-1] + x[1:, :-1]
                    + x[:-1, 1:] + x[1:, 1:])
        yc = 0.25*(y[:-1, :-1] + y[1:, :-1]
                    + y[:-1, 1:] + y[1:, 1:])
        return (xc, yc)

    def _calculate_cell_surface_areas(self):
        nx,ny = self.n
        x,y = self.coords

        s_xi = np.zeros((3, nx+1, ny))
        s_xi[0, 1:,1:] = y[1:nx+1, 2:ny+1] - y[1:nx+1, 1:ny]
        s_xi[1, 1:,1:] = x[1:nx+1, 1:ny] - x[1:nx+1, 2:ny+1]
        s_xi[2] = (s_xi[0]**2 + s_xi[1]**2)**0.5


        s_eta = np.zeros((3, nx, ny+1))
        s_eta[0, 1:,1:] = y[1:nx, 1:ny+1] - y[2:nx+1, 1:ny+1]
        s_eta[1, 1:,1:] = x[2:nx+1, 1:ny+1] - x[1:nx, 1:ny+1]
        s_eta[2] = (s_eta[0]**2 + s_eta[1]**2)**0.5

        return (s_xi, s_eta)

    def _get_airfoil_limits(self):
        x,_ = self.coords
        return np.argwhere(x[1:,1] == 0)[0,0] + 1, np.argwhere(x[1:,1] == 1)[0,0] + 1