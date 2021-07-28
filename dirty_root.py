### helper functions to find a root of a function roughly

# from gibbs import GenericMixingGibbs
from os import pardir
import numpy as np
import numba as nb


import itertools
from scipy.interpolate import UnivariateSpline
import scipy.optimize as opt


from joblib import Parallel, delayed

import psutil

def spline(x,v,axis = 0):
    spline = UnivariateSpline(x,v)

    return ()



def old_opt(self, Ts,Ps):
    opt_x = []

    #def target_fun(x, T, P): return float(self.energy_free_x(T, P, x))


    def target_fun(x, T, P): return float(self.energy_x_grad(T, P, x))
    
    ## Can implement this to change the upper limit
    ## Higher P -> 
    ## Higher T -> lower x

    x_upper = 1

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
        #opt_x.append(opt.minimize_scalar(target_fun, args=(
        #    T, P), method='bounded', bounds=(0, 1)).x)

        opt_x.append(opt.brentq(target_fun,1e-7,1-1e-7,args=(T,P)))


    return np.array(opt_x).reshape(len(Ts), len(Ps))


def brent_opt(func,T:np.float64,P:np.float64,lower = 0,upper = 1,eps = 1e-7):


    return opt.brentq(func,lower+eps,upper-eps,args = (T,P))

    pass


def para_opt(self,Ts,Ps, Ncores = -1,verbose = 0):


    Ts = np.asarray(Ts)
    if Ts.shape == ():
        Ts = [Ts]
    Ps = np.asarray(Ps)
    if Ps.shape == ():
        Ps = [Ps]

    func = self.energy_x_grad

    # @nb.vectorize
    def target_fun(x, T, P): return float(func(T, P, x))

    # @nb.vectorize([nb.float64(nb.float64, nb.float64)]) 
    def opt_one(T,P)->nb.float64:
        return brent_opt(target_fun,T,P)
    
    par = Parallel(n_jobs = Ncores,verbose=verbose)

    results = par(delayed(opt_one)(T,P) for T in Ts for P in Ps)


    return np.array(results).reshape(len(Ts),len(Ps))