from fluxes.inviscid import calculate_E_xi, calculate_F_eta
from boundary_conditions import set_common_boundary_conditions, set_euler_boundary_conditions
from config import CFL_MAX
from constants.fluid_properties import gamma
import numpy as np

class EulerSolver:
    def __init__(self, grid : Grid2D, Q_init):
        self.grid = grid
        self.Q_init = Q_init
        self.initialize_Q()
    
    def set_boundary_conditions(self):
        set_common_boundary_conditions(self.Q[-1], self.Q_init, self.grid)
        set_euler_boundary_conditions(self.Q[-1], self.Q_init, self.grid)

    def initialize_Q(self):
        vol = self.grid.cell_volumes
        nx,ny = self.grid.n

        self.Q = [np.zeros(self.Q_init.shape + vol.shape)]

        self.Q[0][:, 1:nx,1:ny] = self.Q_init[:,None,None]
        self.set_boundary_conditions()

    def calculate_time_step_cap(self):
        nx,ny = self.grid.n
        s_xi,s_eta = self.grid.cell_surface_areas

        p_cell = (gamma - 1) * (self.Q[-1][3, 1:nx,1:ny] - 0.5*(self.Q[-1][1, 1:nx,1:ny]**2 + self.Q[-1][2, 1:nx,1:ny]**2) / self.Q[-1][0, 1:nx,1:ny])
        c_cell = np.sqrt(gamma * p_cell / self.Q[-1][0, 1:nx,1:ny])

        n_xi_cell = 0.5*(s_xi[:2, 1:nx,1:ny]/s_xi[2, 1:nx,1:ny] + s_xi[:2, 2:nx+1,1:ny]/s_xi[2, 2:nx+1,1:ny])
        s_xi_cell = 0.5 * (s_xi[2, 1:nx,1:ny] + s_xi[2, 2:nx+1,1:ny])

        U_xi_cell = self.Q[-1][1, 1:nx,1:ny]*n_xi_cell[0] + self.Q[-1][2, 1:nx,1:ny]*n_xi_cell[1]

        n_eta_cell = 0.5*(s_eta[:2, 1:nx,1:ny]/s_eta[2, 1:nx,1:ny] + s_eta[:2, 1:nx,2:ny+1]/s_eta[2, 1:nx,2:ny+1])
        s_eta_cell = 0.5 * (s_eta[2, 1:nx,1:ny] + s_eta[2, 1:nx,2:ny+1])

        V_eta_cell = self.Q[-1][1, 1:nx,1:ny]*n_eta_cell[0] + self.Q[-1][2, 1:nx,1:ny]*n_eta_cell[1]

        rho_xi = (abs(U_xi_cell) + c_cell) * s_xi_cell
        rho_eta = (abs(V_eta_cell) + c_cell) * s_eta_cell

        return np.min(CFL_MAX / (rho_xi + rho_eta), axis=0)

    def advance_time_step(self):
        nx,ny = self.grid.n
        s_xi,s_eta = self.grid.cell_surface_areas
        vol = self.grid.cell_volumes

        E_xi_curr = calculate_E_xi(self.Q[-1], self.grid)
        F_eta_curr = calculate_F_eta(self.Q[-1], self.grid)

        delta_tau_cap = self.calculate_time_step_cap()

        Q_diff = np.zeros(self.Q[-1].shape)

        Q_diff[:, 1:nx,1:ny] = - (delta_tau_cap) * ( \
            ((E_xi_curr[:, 2:nx+1,1:ny]) * s_xi[2, 2:nx+1,1:ny]
                - (E_xi_curr[:, 1:nx,1:ny]) * s_xi[2, 1:nx,1:ny])
            + ((F_eta_curr[:, 1:nx,2:ny+1]) * s_eta[2, 1:nx,2:ny+1]
                - (F_eta_curr[:, 1:nx,1:ny]) * s_eta[2, 1:nx,1:ny])
            )
        
        self.Q.append(self.Q[-1] + Q_diff)
        self.Q_diff_curr = Q_diff
        self.set_boundary_conditions()

        ## OUTLET - first order interp
        self.Q[-1][:, nx,1:ny] = self.Q[-2][:, nx-1,1:ny]