import numpy as np

from numpy.polynomial import Polynomial
from numpy.polynomial.polynomial import polyval2d, polyvander2d

import numpy.polynomial.polynomial as poly


class Poly2D:
    def __init__(self, c):

        if len(c.shape) == 2:
            self.coef = c
            self.c = self.coef
        else:
            raise ValueError('Polynomial coefficients need to be a 2d array')

    def val(self, x, y, **kwargs):
        return poly.polyval2d(x, y, self.coef, **kwargs)

    def deriv(self, m=1, scl=1, axis=0):
        '''
            m : int, optional
                order of differntiation
            scl : scalar, optional
                scaling factor
            axis : int
                index of axis of differntiation
        '''
        return Poly2D(poly.polyder(self.coef, m=m, scl=scl, axis=axis))

    def integ(self, m=1, k=[], lbnd=0, scl=1, axis=0):
        '''
            m : int, optional
                Integrate this many times
            k : array_like, optional
                constants of integration
            axis = axis of integration
        '''

        return Poly2D(poly.polyint(self, m=m, k=k, lbnd=lbnd, scl=scl, axis=axis))

    def grid(self, x, y):
        return poly.polygrid2d(x, y, self.coef)

    def __call__(self, x, y):
        return self.val(x, y)

    def __repr__(self):
        return '\n'.join(['2D - Polynomial with coefficients: ', self.coef.__repr__()])

    ## Multiplying and adding not implemented
    def __add__(self, other):

        return Poly2D(self.coef+other.coef)

    def __sub__(self, other):
        return Poly2D(self.coef-other.coef)

    def __mul__(self, other):
        raise NotImplementedError(
            '2D polynomial multiplication not implemented')
        #return Poly2D(poly.polymul(self.coef,other.coef))

    def __div__(self, other):
        raise NotImplementedError('2D polynomial division not implemented')

        #return #Poly2D(poly.polydiv(self.coef,other.coef))
