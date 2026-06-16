import numpy as np
from constants.limiter_types import *

class FluxLimiterFactory:
    def __init__(self):
        self._limiters = {}
        self._limiters[NONE_LIMITER] = DefaultLimiter
        self._limiters[MINMOD_LIMITER] = MinmodLimiter
        self._limiters[SUPERBEE_LIMITER] = SuperbeeLimiter

    def register_limiter(self, limiter_type: str, limiter: FluxLimiter):
        self._limiters[limiter_type] = limiter

    def get_limiter(self, limiter_type: str) -> FluxLimiter:
        limiter = self._limiters.get(limiter_type)
        if not limiter:
            raise UserWarning(f"Flux Limiter of type {limiter_type} does not exist. Using DefaultLimiter instead.")
            return DefaultLimiter()
        return limiter()


class FluxLimiter:
    def apply(self, r):
        pass

class DefaultLimiter(FluxLimiter):
    def apply(self, r):
        return r

class MinmodLimiter(FluxLimiter):
    def apply(self, r):
        return np.fmin(np.fmax(r, 0), 1)

class SuperbeeLimiter(FluxLimiter):
    def apply(self, r):
        return np.fmax(0, np.fmax(np.fmin(2*r, 1), np.fmin(r, 2)))