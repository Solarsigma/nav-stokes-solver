from grid import Grid2D

## BOUNDARY CONDITIONS
def set_common_boundary_conditions(Q, Q0, grid : Grid2D):
    nx,ny = grid.n
    s_xi,s_eta = grid.cell_surface_areas
    airfoil_start = grid.airfoil_start
    airfoil_end = grid.airfoil_end

    ## INLET - free-stream
    Q[:, 0,:] = Q0[:, None]
    
    ## TOP WALL - inviscid adiabatic slip
    # rho and et are const
    Q[0, 1:nx,ny] = Q[0, 1:nx,ny-1]
    Q[3, 1:nx,ny] = Q[3, 1:nx,ny-1]

    # u and v - slip conditions
    n = s_eta[0:2, 1:nx,ny] / s_eta[2, 1:nx,ny]
    Q[1, 1:nx,ny] = (n[1]**2 - n[0]**2) * Q[1, 1:nx,ny-1] - 2*n[0]*n[1] * Q[2, 1:nx,ny-1]
    Q[2, 1:nx,ny] = -2*n[0]*n[1] * Q[1, 1:nx,ny-1] - (n[1]**2 - n[0]**2) * Q[2, 1:nx,ny-1]
    
    ## BOTTOM WALL - adiabatic inviscid slip
    # rho and et are const
    Q[0, 1:airfoil_start,0] = Q[0, 1:airfoil_start,1]
    Q[3, 1:airfoil_start,0] = Q[3, 1:airfoil_start,1]

    # u and v - slip conditions
    n = s_eta[0:2, 1:airfoil_start,1] / s_eta[2, 1:airfoil_start,1]
    Q[1, 1:airfoil_start,0] = (n[1]**2 - n[0]**2) * Q[1, 1:airfoil_start,1] - 2*n[0]*n[1] * Q[2, 1:airfoil_start,1]
    Q[2, 1:airfoil_start,0] = -2*n[0]*n[1] * Q[1, 1:airfoil_start,1] - (n[1]**2 - n[0]**2) * Q[2, 1:airfoil_start,1]
    
    ## BOTTOM WALL - adiabatic inviscid slip
    # rho and et are const
    Q[0, airfoil_end+1:nx,0] = Q[0, airfoil_end+1:nx,1]
    Q[3, airfoil_end+1:nx,0] = Q[3, airfoil_end+1:nx,1]

    # u and v - slip conditions
    n = s_eta[0:2, airfoil_end+1:nx,1] / s_eta[2, airfoil_end+1:nx,1]
    Q[1, airfoil_end+1:nx,0] = (n[1]**2 - n[0]**2) * Q[1, airfoil_end+1:nx,1] - 2*n[0]*n[1] * Q[2, airfoil_end+1:nx,1]
    Q[2, airfoil_end+1:nx,0] = -2*n[0]*n[1] * Q[1, airfoil_end+1:nx,1] - (n[1]**2 - n[0]**2) * Q[2, airfoil_end+1:nx,1]
    
    ## OUTLET - NVCBC's
    Q[:, nx,:] = Q[:, nx-1,:]

def set_euler_boundary_conditions(Q, Q0, grid : Grid2D):
    airfoil_start = grid.airfoil_start
    airfoil_end = grid.airfoil_end
    _,s_eta = grid.cell_surface_areas
    ## AIRFOIL_EDGE - adiabatic inviscid slip for Euler
    # rho and et are const
    Q[0, airfoil_start:airfoil_end+1,0] = Q[0, airfoil_start:airfoil_end+1,1]
    Q[3, airfoil_start:airfoil_end+1,0] = Q[3, airfoil_start:airfoil_end+1,1]

    n = s_eta[0:2, airfoil_start:airfoil_end+1,1] / s_eta[2, airfoil_start:airfoil_end+1,1]
    Q[1, airfoil_start:airfoil_end+1,0] = (n[1]**2 - n[0]**2) * Q[1, airfoil_start:airfoil_end+1,1] - 2*n[0]*n[1] * Q[2, airfoil_start:airfoil_end+1,1]
    Q[2, airfoil_start:airfoil_end+1,0] = -2*n[0]*n[1] * Q[1, airfoil_start:airfoil_end+1,1] - (n[1]**2 - n[0]**2) * Q[2, airfoil_start:airfoil_end+1,1]

def set_nav_stokes_boundary_conditions(Q, Q0, grid : Grid2D):
    airfoil_start = grid.airfoil_start
    airfoil_end = grid.airfoil_end
    _,s_eta = grid.cell_surface_areas
    ## AIRFOIL_EDGE - adiabatic no-slip
    # rho and et are const
    Q[0, airfoil_start:airfoil_end+1,0] = Q[0, airfoil_start:airfoil_end+1,1]
    Q[3, airfoil_start:airfoil_end+1,0] = Q[3, airfoil_start:airfoil_end+1,1]

    n = s_eta[0:2, airfoil_start:airfoil_end+1,1] / s_eta[2, airfoil_start:airfoil_end+1,1]
    Q[1, airfoil_start:airfoil_end+1,0] = - Q[1, airfoil_start:airfoil_end+1,1]
    Q[2, airfoil_start:airfoil_end+1,0] = - Q[2, airfoil_start:airfoil_end+1,1]