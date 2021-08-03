from scipy.constants import gas_constant
from abc import ABC, abstractmethod
import numpy as np

from joblib import Memory

import numba as nb

from numba.experimental import jitclass


from poly2d import Poly2D

import scipy.optimize as opt
#import itertools

import dirty_root

import lib


class FreeEnergy(ABC):

    @abstractmethod
    def energy(self):
        pass


class IdealGibbs(FreeEnergy):
    '''calculate the ideal solution gibbs free energy, takes x as a parameter
'''

    def __init__(self, omega=0):

        # set the non-deality parameter
        self.omega = omega

        self.bmixer = None

    def energy(self, T, x):
        '''
            Evaluate Free Energy of the nearly ideal liqud
            given by 
            # unitless
            G/(RT_c) = T^ (xln(x) + (1-x)ln(x) + omega*x*(1-x))
        '''
        return T * (x*np.log(x) + (1-x)*np.log(1-x) + self.omega*x*(1-x))

    def opt_x(self, T):
        '''Find the optimal value(s) of x for a given temperature'''

        pass


class BMixingGibbs(IdealGibbs):
    def __init__(self, G_BA: np.array, omega: np.array):
        '''
            G_BA :  array_like, can be 2d array of values corresponding to grid of pressure/temp values


        '''

        # pass the value'''' of omega to the level above
        super().__init__(omega)
        
        self.omega = omega
        self.G_BA = G_BA  # assign the coefficient for G_BA

    def energy(self, T, x):
        ''' As with ideal depends only on the free energy of the the above'''

        # x_e = func(x)

        A = self.G_BA * x
        B = (x*np.log(x) + (1-x)*np.log(1-x))
        C = self.omega*x*(1-x) 
        near_ideal_shape = (B+C).shape
        # print(near_ideal_shape)
        # double transpose multiplies along temp axis
        return A + (T*(B + C).T).T

    def energy_x_grad(self, T, x):
        ###
        # work out the energy gradient
        # criteria for minimum =0
        # TODO check this is vectorised correctly
        # TODO it seems just adding G_BA and Omega gets a square matrix rather than a vector (if they're both vectors.)
        # TODO make a rank-3 tensor like the energy is?

        A = self.G_BA
        B = np.log(x/(1-x))
        C = self.omega * (1-2*x)

        # if a matrix, make a vector## to deal with the cases where C is a N by 1 matrix, rather than a vector
        A = A.reshape(C.shape)
        return A + (T*(B + C).T).T

    def x_equib(self, T, grid=1000):
        '''
            Overwritten in inherited classes. Not implemented correctly
        
        '''
        x = np.linspace(0, 1, grid)

        def grad(x):
            return np.gradient(self.energy(T, x), x, axis=-1)

        def target_fun(x): return self.energy(T, x)

        # spline_approx = #

        # grad(x)

        return opt.minimize(target_fun, x0=0.5)

        ''' TODO implement'''

        # find the value of x nearest the

        # Quick and dirty, worry about vectorising for different pressures later...add()

        raise NotImplementedError

# @jitclass()


class GenericMixingGibbs(FreeEnergy):
    '''composes a BMixingGibbs object calculates G_BA on the fly?? - depends on pressure'''

    def __init__(self, poly_B: Poly2D, omega_func, Pc_hat, ncores=-1):
        '''
                poly_B = the polynomial form for the G_BA coefficient. 2d polynomial

                omega_func : function of T and deltaP, dimensionless, that gives omega

                Dimensionless:
                Tc : float
                    Critical temperature for l-l transiotn
                Pc : float
                    critical pressure for l-l transiton
                ncores : paralised by using ncores. specify 1 to not use
        '''

        self.poly_B = poly_B
        self.omega_func = omega_func
        self.Pc_hat = Pc_hat
        # redundant Tc_hat is by definition 1, only really need for Pc_hat because
        # P_hat =/= P/P_c
        # P_hat = P/(R T_c rho_c)

        self.Tc_hat = 1
        self.ncores = ncores
        # initialise the bmixer subobject

        self.bmixer = BMixingGibbs(self.poly_B.grid(
            1, Pc_hat), self.omega_func(1, 0))

        # self.cache = cache

    def energy(self, T, P):
        x = self.x_equib(T, P)

        return self.energy_free_x(T, P, x)

    def energy_free_x(self, T_hat, P_hat, x):
        '''
            evaluate the temperature and pressure for a free x of the free energy
        '''

        # initialise BMixingGibbs object with G_ba and omega

        self.set_bmixer(T_hat, P_hat)
        #self.bmixer.x_equib(T_hat, P_hat)

        return self.bmixer.energy(T_hat, x)

    def energy_x_grad(self, T, P, x):
        '''
            TODO, parse the omega and gab functions here directly
        '''
        self.set_bmixer(T, P)

        delP = P-self.Pc_hat
        delT = T-1

        gBA = self.poly_B.grid(delT, delP)

        omega = self.omega_func(T, delP)

        return lib.energy_x_grad_general(T, P, x, gBA, omega)

        # self.set_bmixer(T,P)
        # return self.bmixer.energy_x_grad(x)

    def x_equib(self, Ts, Ps):
        '''
            TODO: To do, change the optimisation procedure to find roots instead!!!!
        '''

        '''
        def grad(x): 
            return self.energy_grad(Ts,Ps,x)

        x0 = 0.5

        sol = opt.root(grad,)
        
        return dirty_root.old_opt(self,Ts,Ps)
        '''
        #raise NotImplementedError

        # Pre evaluate the array for calculating the energies
    
        
        self.set_bmixer(Ts,Ps)

        delP = Ps-self.Pc_hat

        delT = Ts - 1

        gBA = self.poly_B.grid(delT, delP)

        omega = self.omega_func(Ts, delP)

        # sol.x
        return dirty_root.para_opt(lib.energy_x_grad_general, Ts, Ps, gBA, omega, Ncores=self.ncores)

        # def grad(x):
        #     return np.gradient(self.energy(T, x), x, axis=-1)

        # spline_approx = #

        # grad(x)

        # return opt.minimize(target_fun, x0=1)

        # return self.bmixer.x_equib(T)

    def set_bmixer(self, T, P) -> None:
        deltaT = T - self.Tc_hat
        deltaP = P-self.Pc_hat
        # create an array of values for this
        G_BA = self.poly_B.grid(deltaT, deltaP)
        omega = self.omega_func(T, deltaP)
        if self.bmixer is None:
            self.bmixer: BMixingGibbs = BMixingGibbs(G_BA, omega)
        else:
            self.bmixer.omega = omega
            self.bmixer.G_BA = G_BA


class FinalMixingGibbs(GenericMixingGibbs):

    def __init__(self, coefs_2D, omega_0, Pc_hat, ncores=-1) -> None:
        '''
            Set up Mixing of structure free energies, as given in 

        '''

        self.b_coefs = coefs_2D

        self.omega0 = omega_0

        def omega_func(T, delP): return lib.bid_func_omega(T, delP, omega_0)

        super().__init__(Poly2D(coefs_2D), omega_func, Pc_hat, ncores=ncores)


class Cached_Mixer(FinalMixingGibbs):

    def __init__(self, coefs_2D, omega_0, Pc_hat, mem: Memory, ncores=-1):
        ''' Applying the caching by overwriting at innit 
            NOTE This might be a poor usage. The docs suggest caching methods is not reccommended and should really only be used for pure functions

        '''
        super().__init__(coefs_2D, omega_0, Pc_hat, ncores=ncores)

        self.mem: Memory = mem
        self.energy = self.mem.cache(self.energy)
        self.x_equib = self.mem.cache(self.x_equib)

    # def energy(self, T, P):
        # apply decorator to cache the output

     #   return self.mem.cache(super().energy(T, P))

    # def x_equib(self, Ts, Ps):
    #    return self.mem.cache(super().x_equib(Ts, Ps))


class GibbsPoly(FreeEnergy):

    def __init__(self, coefs):
        self.coefs = coefs
        self.poly = Poly2D(coefs)

    def energy(self, T, P):
        '''
            Calculate grid of energy values, for provided pressures and temps. 
            T : unit less temperature property
            P : unitless Pressure property. NB. may need to convert to deltas, depending on implementation - done in child classes, e.g. GibbsA
        '''
        return self.poly.grid(T, P)


class GibbsA(GibbsPoly):

    def __init__(self, coefs, Pc):
        #pc is unitless
        self.Pc = Pc
        super().__init__(coefs)

    def energy(self, T, P):

        # reduced pressure and temp differences
        deltaT = T-1
        deltaP = P-self.Pc
        return super().energy(deltaT, deltaP)


class GibbsSpin(FreeEnergy):
    def __init__(self, coef_A, coef_Ps):

        self.polyA = np.polynomial.Polynomial(
            coef_A)  # Linear polynomial in deltaT
        self.polyPs = np.polynomial.Polynomial(
            coef_Ps)  # Quadratic polynomial in delta T

    def energy(self, T, P):
        '''
            TODO Energy function is broken, doesn't work with vectors of both T and P, of different shapes 

        '''
        delT = T-1

        #  doesn't work

        A = self.polyA(delT)  # linear in del T

        if np.shape(T) == ():
            T = [T]
        if np.shape(P) == ():
            P = [P]

        # For making properly sized arrays so that they play nicely
        P_ghost = np.ones(np.shape(P))
        T_ghost = np.ones(np.shape(T))

        # reshaping so each vector can be deducted
        # np.outer(T_ghost, P) makes an array of len(T),len(P) in shape
        B = (np.outer(T_ghost, P)-np.outer(self.polyPs(delT), P_ghost))**1.5

        #assert B[0,0] == (P - self.polyPs(delT[0]))**1.5

        return (B.T*A).T
        #spin.energy(Ts, Ps)

        # return np.outer(self.polyA(delT), (P-self.polyPs(delT))**1.5)


class FinalRedUnits(FreeEnergy):
    def __init__(self, Pc_hat: float, coef_GAB, omega_0, coef_A, coef_Ps, coef_GA, cache_dir='./__gibbs_cache__', ncores=-1, fast=False):
        '''
            Coefficients in the order they are in the Table I in the paper 
            except lambda is multiuplied with the coefficients a,b,d,f



            cache_dir = './__gibbs_cache__' Set to none to not use caching
        '''
        self.Pc_hat = Pc_hat

        if cache_dir is None:
            self.mixer: FinalMixingGibbs = FinalMixingGibbs(
                coef_GAB, omega_0, Pc_hat, ncores=ncores)
        else:
            self.mixer: FinalMixingGibbs = Cached_Mixer(
                coef_GAB, omega_0, Pc_hat, mem=Memory(cache_dir), ncores=ncores)

        self.spinner: GibbsSpin = GibbsSpin(coef_A, coef_Ps)
        self.gibbs_A: GibbsA = GibbsA(coef_GA, Pc_hat)
        self.fast = False

    def energy(self, T, P):

        if not self.fast:
            mix_eng = self.mixer.energy(T, P)
        else:
            mix_eng = 0

        spin_eng = self.spinner.energy(T, P)
        A_eng = self.gibbs_A.energy(T, P)

        return A_eng + spin_eng + mix_eng

    def vol(self, T, P):
        return np.gradient(self.energy(T, P), P, axis=1)

    def rho(self, T, P):
        return 1/self.vol(T, P)

    def alpha(self, T, P):

        vol = self.vol(T, P)

        dVdT = np.gradient(vol, T, axis=0)

        return dVdT


class RealGibbs(FinalRedUnits):

    def __init__(self, Tc, Pc: float, rhoc, coef_GAB, omega_0, coef_A, coef_Ps, coef_GA, fast=False, mol_mass=18.015, cache_dir='./__gibbs_cache__', ncores=-1):
        '''
            Tc in K
            Pc in bar
            rho_c in g cm^-3
        '''
        self.mol_mass = mol_mass
        self.R = gas_constant
        self.Tc = Tc
        self.Pc = Pc
        self.rhoc = rhoc
        Pc_hat = self.convert_P(Pc) #*1e5/self.R/self.rhoc_mol_m3/self.Tc  # convert to Pa then divide

        self.mol_mass = mol_mass

        super().__init__(Pc_hat, coef_GAB, omega_0, coef_A, coef_Ps,
                         coef_GA, fast=fast, cache_dir=cache_dir, ncores=ncores,)

    def convert_T(self, T):
        return T/self.Tc

    def convert_rho(self, rho):
        return rho / self.rhoc

    @property
    def rhoc_mol_m3(self):
        return self.rhoc * 1e6/(self.mol_mass)  # mol m^-3

    def convert_P(self, P):
        # convert from bar to pascal and divide by quantity with units pascal (J m^-3)
        return P*1e5/(self.Tc*self.R*self.rhoc_mol_m3)

    def convert_TP(self, T, P):
        '''convert the temperature and pressure to their dimensionless forms'''

        return self.convert_T(T), self.convert_P(P)

    def energy(self, T, P):
        '''
            The molar volume of the system, J mol^-1
        '''
        return super().energy(*self.convert_TP(T, P))*self.Tc*self.R

    def vol(self, T, P):
        '''
            The molar volume of the system, in m^3/mol
        '''
        return super().vol(T, P)/self.rhoc_mol_m3

    def rho(self, T, P):
        '''
            The density of the system, in g cm^-3

        '''

        return super().rho(T, P)  # *self.rhoc calls self.vol, so not needed...

    def alpha(self, T, P):
        '''
            Thermal expansion coefficient of the system in K^-1

        '''
        return super().alpha(T, P)  # /self.Tc  # divide by Tc to get right units

    def x(self, T, P):
        '''
            Find the optimal x for the supplied real temperatures and pressures
            Is this near zero?
            can be used to see if the mixing terms are needed for the states investigated.
        '''
        return self.mixer.x_equib(*self.convert_TP(T, P))


crit_params = {'Tc': 182, 'Pc': 1700, 'rhoc': 1.017}

lamb = 1.55607
a, b, d, f = [0.154014, 0.125093, 0.00854418, 1.14576]
coef_GAB = lamb * np.array([[0, a, d],
                            [1, b, 0],
                            [f, 0, 0]])


omega_0 = 0.03
coef_A = [-0.0547873, -0.0822462]
coef_Ps = [-5.40845, 5.56087, -2.5205]
coef_GA = np.array([[0, 0, -0.00261876, 0.000605678],
                    [0, 0.257249, 0.0248091, -0.000994166],
                    [-6.30589, -0.0400033, -0.00840543, 0],
                    [2.18819, 0.0719058, 0, 0],
                    [-0.256674, 0, 0, 0]])

non_crit = {'coef_GAB': coef_GAB, 'omega_0': omega_0,
            'coef_A': coef_A, 'coef_Ps': coef_Ps, 'coef_GA': coef_GA}

biddle_params = {**crit_params, **non_crit}


class BiddleFreeEn(RealGibbs):
    def __init__(self, cache_dir=None, ncores=-1):
        super().__init__(**biddle_params, cache_dir=cache_dir, ncores=ncores)
