### helper functions to find a root of a function roughly

# from gibbs import GenericMixingGibbs
from os import pardir
import numpy as np
import numba as nb


import itertools
from scipy.interpolate import UnivariateSpline
import scipy.optimize as opt

## TODO: perform checks to see if this package is avaliable, if not remove and use no sped up version 
from joblib import Parallel, delayed,Memory

import psutil


from lib import energy_x_grad_general




###

###setting up memorizer

meme = Memory(location='__gibbs_cache__')


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

#@meme.cache
def brent_opt(func,T:np.float64,P:np.float64,*args,lower = 0,upper = 1,eps = 1e-7):
    '''
        TODO: Add try except statements for the situations where the surface is very flat (T -> infty), meaning the endpoints have the same sign!!!!
    '''
    
    try:
        return opt.brentq(func,lower+eps,upper-eps,args = (T,P,*args))

    except ValueError:
        # Handling the cases that bot have the same sign
        # Account for the fact the system must have 0 or 1 
        f_a = func(lower+eps,T,P,*args)
        f_b = func(upper-eps,T,P,*args)
        if f_a <= f_b:
            return lower+eps
        else:
            return upper-eps
    
     

    pass

## for caching the output of this function
@meme.cache
def para_opt(func,Ts,Ps,g,omega, Ncores = -1,verbose = 0,*args):


    Ts = np.asarray(Ts)
    if Ts.shape == ():
        Ts = [Ts]
    Ps = np.asarray(Ps)
    if Ps.shape == ():
        Ps = [Ps]
    g = np.asarray(g)
    
    # reshape into (1,1) if only a scalar
    if g.shape == ():
        g = g*np.ones((1,1)) ## make into a 1*1 array of flots
    omega = np.asarray(omega)
    if omega.shape == ():
        omega = omega*np.ones((1,1))
    # Reshape into a 2d vector if only a 1d vector
    
    if len(g.shape) ==1:
        g = g.reshape((len(Ts),len(Ps)))
    if len(omega.shape) ==1:
        omega = omega.reshape((len(Ts),len(Ps)))

    # assert g.shape == omega.shape
    # assert omega.shape == (*Ts.shape, *Ps.shape)

    #func = self.energy_x_grad

    # @nb.vectorize
    def target_fun(x, T, P,g,omega): return float(func(T, P, x,g,omega ))

    # @nb.vectorize([nb.float64(nb.float64, nb.float64)]) 
    def opt_one(T,P,g,omega)->nb.float64:
        return brent_opt(target_fun,T,P,g,omega)
    
    par = Parallel(n_jobs = Ncores,verbose=verbose)

    
    
    results = par(delayed(opt_one)(Ts[i],Ps[j],g[i,j],omega[i,j] ) for i in range(len(Ts)) for j in range(len(Ps)))


    return np.array(results).reshape(len(Ts),len(Ps))


if __name__ == '__main__':

    from gibbs import BiddleFreeEn




    bid : BiddleFreeEn = BiddleFreeEn()

    bid.omega
    