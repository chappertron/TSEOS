from scipy.constants import gas_constant
from abc import ABC, abstractmethod
import numpy as np



import numba as nb

from numba.experimental import jitclass


from poly2d import Poly2D

import scipy.optimize as opt
#import itertools

import dirty_root

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
        self.G_BA = G_BA  # assign the coefficient for G_BA

    def energy(self, T, x):
        ''' As with ideal depends only on the free energy of the the above'''

        # x_e = func(x)

        A = self.G_BA * x
        B = (x*np.log(x) + (1-x)*np.log(1-x))
        C = self.omega*x*(1-x)  # np.multiply.outer(self.omega,x*(1-x))
        near_ideal_shape = (B+C).shape
        # print(near_ideal_shape)
        # double transpose multiplies along temp axis
        return A + (T*(B + C).T).T

    def energy_x_grad(self,x):
        ###
            # work out the energy gradient
            # criteria for minimum =0 
        ### TODO check this is vectorised correctly
        ### TODO it seems just adding G_BA and Omega gets a square matrix rather than a vector (if they're both vectors.)
        ### TODO make a rank-3 tensor like the energy is? 


        A = self.G_BA
        B = np.log(x/(1-x)) 
        C = self.omega * (1-2*x)


        A = A.reshape(C.shape) ## if a matrix, make a vector## to deal with the cases where C is a N by 1 matrix, rather than a vector
        return A + B + C


    def x_equib(self, T, grid=1000):

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

#@jitclass()
class GenericMixingGibbs(FreeEnergy):
    '''composes a BMixingGibbs object calculates G_BA on the fly?? - depends on pressure'''

    def __init__(self, poly_B: Poly2D, omega_func, Pc_hat):
        '''
                poly_B = the polynomial form for the G_BA coefficient. 2d polynomial

                omega_func : function of T and deltaP, dimensionless, that gives omega

                Dimensionless:
                Tc : float
                    Critical temperature for l-l transiotn
                Pc : float
                    critical pressure for l-l transiton
        '''

        self.poly_B = poly_B

        self.omega_func = omega_func
        self.Pc_hat = Pc_hat
        # redundant Tc_hat is by definition 1, only really need for Pc_hat because
        # P_hat =/= P/P_c
        # P_hat = P/(R T_c rho_c)

        self.Tc_hat = 1
        # initialise the bmixer subobject

        self.bmixer = BMixingGibbs(self.poly_B.grid(
            1, Pc_hat), self.omega_func(1, 0))

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

    def energy_x_grad(self, T,P, x):
        self.set_bmixer(T,P)
        return self.bmixer.energy_x_grad(x)

    def x_equib(self, Ts, Ps,old = False):
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

        return  dirty_root.old_opt(self,Ts,Ps) #sol.x


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

    def __init__(self, coefs_2D, omega_0, Pc_hat) -> None:
        '''
            Set up Mixing of structure free energies, as given in 

        '''

        def func_omega(t, del_p): return np.outer((1/t), (2 + omega_0*del_p))

        super().__init__(Poly2D(coefs_2D), func_omega, Pc_hat)


class GibbsPoly(FreeEnergy):

    def __init__(self, coefs):
        self.coefs = coefs
        self.poly = Poly2D(coefs)

    def energy(self, T, P):
        '''
            Calculate grid of energy values, for provided pressures and temps. 
            T : unit less temperature property
            P : unitless Pressure property. NB. may need to convert to deltas, depending on implementation
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

        A = self.polyA(delT)

        if np.shape(T) == (): T= [T]
        if np.shape(P) == (): P= [P]

        P_ghost = np.ones(np.shape(P))
        T_ghost = np.ones(np.shape(T))

        ## reshaping so each vector can be deducted 
        B = (np.outer(T_ghost, P)-np.outer(self.polyPs(delT), P_ghost))**1.5


        return (B.T*A).T
        #spin.energy(Ts, Ps)

        #return np.outer(self.polyA(delT), (P-self.polyPs(delT))**1.5)


class FinalRedUnits(FreeEnergy):
    def __init__(self, Pc_hat: float, coef_GAB, omega_0, coef_A, coef_Ps, coef_GA, fast=False):
        '''
            Coefficients in the order they are in the Table I in the paper 
            except lambda is multiuplied with the coefficients a,b,d,f

        '''
        self.mixer: FinalMixingGibbs = FinalMixingGibbs(
            coef_GAB, omega_0, Pc_hat)
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
        return np.gradient(self.energy(T, P), P, axis=-1)

    def rho(self, T, P):
        return 1/self.vol(T, P)

    def alpha(self, T, P):

        vol = self.vol(T, P)

        dVdT = np.gradient(vol, T, axis=0)

        return dVdT


class RealGibbs(FinalRedUnits):

    def __init__(self, Tc, Pc: float, rhoc, coef_GAB, omega_0, coef_A, coef_Ps, coef_GA, fast=False, mol_mass=18.01):
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
        self.Pc_hat = Pc*1e5/self.R/self.rhoc_mol_m3/self.Tc ## convert to Pa then divide 

        self.mol_mass = mol_mass

        super().__init__(self.Pc_hat, coef_GAB, omega_0, coef_A, coef_Ps, coef_GA, fast=fast)

    def convert_T(self, T):
        return T/self.Tc

    def convert_rho(self, rho):
        return rho / self.rhoc

    @property
    def rhoc_mol_m3(self):
        return self.rhoc * 1e6/(self.mol_mass)  # mol m^-3

    def convert_P(self, P):

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
        return super().vol(*self.convert_TP(T, P))/self.rhoc_mol_m3

    def rho(self, T, P):
        '''
            The density of the system, in g cm^-3

        '''

        return super().rho(*self.convert_TP(T, P))*self.rhoc

    def alpha(self, T, P):
        '''
            Thermal expansion coefficient of the system in K^-1

        '''
        return super().alpha(*self.convert_TP(T, P))/self.Tc  # divide by Tc to get right units

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
    def __init__(self,):
        super().__init__(**biddle_params)
