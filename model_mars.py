import numpy as np

class MarsEnv:
    def __init__(self, R, g0, rho_surf, H):
        self.R = float(R)
        self.g0 = float(g0)
        self.rho_surf = float(rho_surf)
        self.H = float(H)

    def g(self, h, vary=False):
        if not vary:
            return self.g0
        r = self.R + max(h, 0.0)
        return self.g0 * (self.R / r)**2

    def density(self, h):
        # exponential scale-height
        return self.rho_surf * np.exp(-max(h, 0.0) / self.H)

    def mu(self):
        # Use g0 * R^2 (consistent with chosen g0)
        return self.g0 * self.R**2
