### helper functions to find a root of a function roughly

import numpy as np

import itertools
from scipy.interpolate import UnivariateSpline
import scipy.optimize as opt


def spline(x,v,axis = 0):
    spline = UnivariateSpline(x,v)

    return ()



def old_opt(self, Ts,Ps):
    opt_x = []

    def target_fun(x, T, P): return float(self.energy_free_x(T, P, x))

    Ts = np.asarray(Ts)
    if Ts.shape == ():
        Ts = [Ts]
    Ps = np.asarray(Ps)
    if Ps.shape == ():
        Ps = [Ps]

    for T, P in itertools.product(Ts, Ps):
        #self.set_bmixer(T, P)
        # print(T,P)
        # bounds = (0,1) because x only in this range
        opt_x.append(opt.minimize_scalar(target_fun, args=(
            T, P), method='bounded', bounds=(0, 1)).x)

    return np.array(opt_x).reshape(len(Ts), len(Ps))
