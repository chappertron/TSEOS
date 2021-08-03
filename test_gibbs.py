from typing import Final
import unittest

from numpy.lib import meshgrid
from numpy.random import random

import poly2d

import gibbs
from gibbs import GenericMixingGibbs, BMixingGibbs, GibbsPoly, GibbsSpin
from gibbs import biddle_params
import numpy as np


class testBMixingGibbs(unittest.TestCase):
    x_shape = (7, 12)
    temp_Press_size = (7, 12)

    # create an array that has the size of temp rpess
    G_BA = np.random.random(size=temp_Press_size)

    temp_vect = 2*np.random.random(size=temp_Press_size[0])

    x_vect = np.random.random(size=x_shape)

    omega = np.random.random(size=temp_Press_size)

    def test_energy(self):

        test_obj = BMixingGibbs(self.G_BA, self.omega)

        self.assertEqual(test_obj.energy(
            self.temp_vect, self.x_vect).shape, self.temp_Press_size)

        def rand_in(n): return np.random.randint(0, n)

        random_index = (rand_in(self.temp_Press_size[0]), rand_in(
            self.temp_Press_size[1]))

        x0 = self.x_vect[random_index[0], random_index[1]]
        G0 = self.G_BA[random_index[0], random_index[1]]
        T0 = self.temp_vect[random_index[0]]
        omega0 = self.omega[random_index[0], random_index[1]]

        left_most = G0*x0+T0*(x0*np.log(x0)+(1-x0) *
                              np.log(1-x0) + omega0*x0*(1-x0))

        self.assertAlmostEqual(test_obj.energy(
            self.temp_vect, self.x_vect)[random_index], left_most)

        pass


class testGenericMixingGibbs(unittest.TestCase):

    coef_test = np.array([[0, 2, 4],
                          [1, 3, 0],
                          [5, 0, 0]])

    def test_energy(self):
        Temps = np.linspace(0.0001, 100, 99)
        Ps = np.linspace(0.0001, 100)

        poly_B = poly2d.Poly2D(self.coef_test)

        omega_0 = 0.003
        # vectorised gives 2d array of coefficients for all
        def omega_func(t, p): return np.outer((1/t), (2 + omega_0*p))

        # print(omega_func(Temps,Ps))
        # print(omega_func(Temps,Ps).shape)

        #print(poly_B.grid(Temps, Ps))
        print(poly_B.grid(Temps, Ps).shape)
        self.assertEqual(omega_func(Temps, Ps).shape,
                         poly_B.grid(Temps, Ps).shape)

        testGibbs = GenericMixingGibbs(poly_B, omega_func, 1)

        Tv, Pv = np.meshgrid(Temps, Ps, indexing='ij')

        print(Tv.shape)

        self.assertEqual(testGibbs.energy(Temps, Ps).shape, Tv.shape)

    def test_min_x(self):

        Temps = np.linspace(0.0001, 100, 10)
        Ps = np.linspace(0.0001, 100, 10)

        poly_B = poly2d.Poly2D(self.coef_test)

        omega_0 = 0.003
        # vectorised gives 2d array of coefficients for all
        def omega_func(t, p): return np.outer((1/t), (2 + omega_0*p))

        # print(omega_func(Temps,Ps))
        # print(omega_func(Temps,Ps).shape)

        #print(poly_B.grid(Temps, Ps))
        print(poly_B.grid(Temps, Ps).shape)
        self.assertEqual(omega_func(Temps, Ps).shape,
                         poly_B.grid(Temps, Ps).shape)

        testGibbs = GenericMixingGibbs(poly_B, omega_func, 1)
        x = testGibbs.x_equib(Temps, Ps)
        print(x.shape)
        self.assertEqual(x.shape, np.outer(Temps, Ps).shape)

        pass

    pass


class TestSpinner(unittest.TestCase):

    def test_energy(self):

        spin = GibbsSpin(biddle_params['coef_A'], biddle_params['coef_Ps'])

        temps = np.random.random(420)
        presses = np.random.random(69)

        randi, randj = np.random.randint(0, 420), np.random.randint(0, 69)

        delT = temps-1

        eman = spin.polyA(delT[randi]) * \
            (presses[randj] - spin.polyPs(delT[randi]))**1.5

        e_imp = spin.energy(temps[randi], presses[randj])

        self.assertEqual(eman, e_imp)


class TestGibbsPoly(unittest.TestCase):

    def test_energy(self):

        temps = np.arange(0, 69, 324)
        presses = np.arange(0, 420, 8432)

        test_coefs = np.array([[12, 69, 21], [13, 42, 19], [69, 42, 91]])

        poly = poly2d.Poly2D(test_coefs)
        gibbbs_poly = GibbsPoly(test_coefs)

        random_i = np.random.randint(len(temps))
        random_j = np.random.randint(len(presses))

        x = temps[random_i]
        y = presses[random_j]
        manual_poly = test_coefs[0, 0] + test_coefs[1, 0] * x + test_coefs[2, 0]*x**2 + test_coefs[0, 1]*y + test_coefs[0,
                                                                                                                        2] * y**2 + test_coefs[1, 1]*x*y + test_coefs[2, 1]*x**2*y+test_coefs[1, 2]*x*y**2 + test_coefs[2, 2]*x**2*y**2



        self.assertAlmostEqual(poly.grid(temps,presses),gibbbs_poly.energy(temps,presses))

        self.assertAlmostEqual(manual_poly,gibbbs_poly.energy(temps,presses)[random_i,random_j])