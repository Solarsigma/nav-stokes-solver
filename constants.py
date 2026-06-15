## LIMITER TYPES
NONE_LIMITER = 'none'
MINMOD_LIMITER = 'minmod'
SUPERBEE_LIMITER = 'superbee'

## CONFIG RELATED
ORDER = 3
CFL_MAX = 0.4
VNN_MAX = 0.5
# LIMITER = 'none'
LIMITER = MINMOD_LIMITER
# LIMITER = 'superbee'
ENV = 'prod'
GRID_FILENAME = 'g65x65s.dat'
FILE_NAME_SUFFIX = f'{ENV}_nav_stokes_order_{ORDER}_limiter_{LIMITER}'

if ORDER == 1:
    MUSCL_EPS = 0
    MUSCL_K = 1
elif ORDER == 2:
    MUSCL_EPS = 1
    MUSCL_K = -1
elif ORDER == 3:
    MUSCL_EPS = 1
    MUSCL_K = 1/2
else:
    MUSCL_EPS = 0
    MUSCL_K = 1


## PROPERTIES OF FLUID/SYSTEM
R = 287 # J/(kg-K)
Cp = 1005 # J/(kg-K)
Pr = 0.72
gamma = 1.4
M = 2
c = 347.2 # m/s
u_ref = 694.4 # m/s
p_ref = 101325 # Pa
T_ref = 300 # K

