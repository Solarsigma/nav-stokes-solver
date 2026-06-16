from constants.limiter_types import *

ORDER = 2
CFL_MAX = 0.4
VNN_MAX = 0.5
LIMITER = MINMOD_LIMITER
ENV = 'debug'
GRID_FILEPATH = 'mesh'
GRID_FILENAME = 'g65x65s.dat'
OUTP_FILEPATH = 'data'
OUTP_FILE_NAME_SUFFIX = f'{ENV}_nav_stokes_order_{ORDER}_limiter_{LIMITER}'