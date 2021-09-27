import numpy as np

import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
from scipy import stats
from scipy import optimize
from scipy import special
from scipy.stats import chi2
import sys
import gc
import config as cfg
import argparse
import pygmo as pg

class objective_function:
        def fitness(self, x):
            def scalingFunc(T, L, Tc, nu, y, A, b, c):

                #A
                #T is a vector of all our data
                #L is data, fixed
                #all else is fit
                t = (T - Tc) / Tc  # This is filled with values for different L

                #2D matrix, m_r X #datapoints
                powers_of_t_chi = np.power(np.column_stack([t] * (m_R)),
                                        np.arange(1, m_R + 1).transpose())  # (powers of t, L data)
                # [t] * (m_R) = [t, t, t, t] (m_R=4 times)
                # [t, t^2, t^3, t^4]

                #same sort of thing but easier than chi
                chi = np.dot(powers_of_t_chi, b)
                chi_vec = np.power(chi * (L ** (1 / nu)), np.column_stack(
                    [np.arange(0, n_R + 1)] * len(t)))  # vector containing 1, chi*L**1/nu, chi**2*L**2/nu, ...


                if n_I > 0:

                    #A when L is big the irrelevant var is useless
                    powers_of_t_psi = np.power(np.column_stack([t] * (m_I + 1)),
                                            np.arange(0, m_I + 1).transpose())  # Always has 1 in the first column
                    psi = np.dot(powers_of_t_psi, c)
                    psi_vec = np.power(psi * (L ** y), np.column_stack([np.arange(0, n_I + 1)] * len(t)))

                    #A matrix product of psi * A * chi
                    A = np.insert(A, [1, int(n_R)], 1.0)  # set F01=F10=1.0
                    A = A.reshape(n_I + 1, n_R + 1)
                    A = np.matmul(A, chi_vec)

                    lam = np.multiply(psi_vec, A)  # elementwise multiply
                    lam = np.sum(lam, axis=0)
                else:
                    A = np.insert(A, 1, 1.0)#A sets the energy scale, aka all other parameters are wrt element 1 
                    lam = np.matmul(A, chi_vec)#computing dot product

                return lam

            #the actual function being optimized
            def objective(args):
                # this is where we specify the loss function
                Tc = args[0]
                nu = args[1]
                y = args[2]
                the_rest = args[3:]
                if n_I > 0:
                    A = the_rest[:int((n_I + 1) * (n_R + 1) - 2)]
                    b = the_rest[int((n_I + 1) * (n_R + 1) - 2):int((n_I + 1) * (n_R + 1) - 2 + m_R)]
                    c = the_rest[int((n_I + 1) * (n_R + 1) - 2 + m_R):]
                else:
                    A = the_rest[:int(n_R)]
                    b = the_rest[int(n_R):int(n_R + m_R)]
                    c = the_rest[int(n_R + m_R):]  # will not matter. Set m_I to 0 if n_I is 0
                Lambda_scaled = scalingFunc(Tvar, L, Tc, nu, y, A, b, c)

                #A the cost thats being tested
                c2 = np.sum(np.abs(Lambda - Lambda_scaled)**2)


                dof = len(Tvar) - numParams

                # c2, p = stats.chisquare(f_obs=Lambda_scaled, f_exp=Lambda, ddof=dof)

                return c2

            return [objective(x)]

        def get_bounds(self):
            bounds = ([[crit_bound_lower, crit_bound_upper],
                            [nu_bound_lower, nu_bound_upper],
                            [y_bound_lower, y_bound_upper]]
                            + [[param_bound_lower, param_bound_upper]] * (numParams))
            bounds = np.transpose(bounds)
            return (bounds[0,:], bounds[1,:])