import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
from scipy import stats
from scipy import optimize
from scipy import special
from scipy.stats import chi2
import pygmo as pg
import warnings
from multiprocessing import Pool
import time

def openfile(filename):
    f = open(filename, "r")
    outputlst = []
    for line in f:
        if line:
            strlist = line.split()
            L = float(strlist[4][:-1])
            W = float(strlist[5][:-1])
            c = float(strlist[7][:-1])
            lyap = float(strlist[10][1:])
            sem = float(strlist[11][:-1])
            outputlst.append([L, W, c, lyap, sem])
    return outputlst
def openfileZeke(filename):
    f = open(filename, "r")
    outputlst = []
    for i, line in enumerate(f):
        strlist = line.split()
        c = float(0.8-0.025*i)
        for j, word in enumerate(strlist):
            L = float(5+j)
            W = float(0.0)
            g = float(word)
            outputlst.append([L, W, c, g])
    return outputlst

def bootVals(Lambda,L,Tvar,sigma,n):
    output=np.array([])
    Lambdanew=np.array([])
    Lnew=np.array([])
    Tvarnew=np.array([])
    sigmanew=np.array([])

    if n==1:
        #don't do any sampling
        return np.array([1]), Lambda, L, Tvar, sigma

    for i in range(0,n):
        outputtemp=(np.random.choice(len(Lambda), int(len(Lambda)), replace=True))
        outputtemp=np.sort(outputtemp)
        if len(output)==0:
            output=outputtemp
            Lambdanew=Lambda[outputtemp]
            Lnew=L[outputtemp]
            Tvarnew=Tvar[outputtemp]
            sigmanew=sigma[outputtemp]
        else:
            output=np.vstack([output,outputtemp])
            Lambdanew=np.vstack([Lambdanew,Lambda[outputtemp]])
            Lnew=np.vstack([Lnew,L[outputtemp]])
            Tvarnew=np.vstack([Tvarnew,Tvar[outputtemp]])
            sigmanew=np.vstack([sigmanew,sigma[outputtemp]])
    return output,Lambdanew,Lnew,Tvarnew,sigmanew

fig1, (ax1, ax3) = plt.subplots(nrows=1, ncols=2, figsize=(11, 6), sharey=True)
fs = 18 #font size
filename="../data/offdiagE2W6.txt"
minL = 8


# orders of expansion
n_R = 3
n_I = 1
m_R = 2
m_I = 1

resamplesize = 60  # number of resamples
numCPUs = 6  # number of processors to use. Each one will do a resample

window_width = 0.06 #width of window
window_offset = 0.0  #  distance from window center to near edge of window
window_center = 0.06

input = np.array(openfile(filename))
Lrange = np.unique(input[:, 0])
Wrange = np.unique(input[:, 1])
crange = np.unique(input[:, 2])


data = input[:, 0:5]  # L, W, c, LE
data[:, 3] = 1 / (data[:, 0] * data[:, 3])  # L, W, c, normalized localization length

# sort according to L
data = data[np.argsort(data[:, 0])]
#omit L less than minL to control finite size effects
data = data[data[:,0]>=minL]
data = data[np.abs(data[:,2]-window_center)<=window_offset+window_width]
data = data[np.abs(data[:,2]-window_center)>=window_offset]

Lambda = data[:, 3]
L = data[:, 0]
W = data[:, 1]
c = data[:, 2]
sigma = data[:, 4] #uncomment for MacKinnon

crit_bound_lower, crit_bound_upper = 0.06, 0.064  # critical value bounds
#crit_bound_lower, crit_bound_upper = min(c), max(c) # critical value bounds
nu_bound_lower, nu_bound_upper = 0.8, 1.9  # nu bounds
y_bound_lower, y_bound_upper = -10.0, -0.1  # y bounds
param_bound_lower, param_bound_upper = -100.0, 100.1  # all other bounds
use_bounds = False

# set the driving parameter
Tvar = c
#print(L)
#print(Tvar)

np.seterr(all='raise')

if n_I > 0:
    numParams = (n_I + 1) * (n_R + 1) + m_R + m_I - 1
else:
    numParams = n_R + m_R



if __name__ == '__main__':

    class objective_function:
        def __init__(self,Lval,Tvarval,Lambda):
            self.L=Lval
            self.Tvar=Tvarval
            self.Lambda=Lambda
        def fitness(self, x):
            def scalingFunc(T, Lval, Tc, nu, y, A, b, c):

                t = (T - Tc) / Tc  # This is filled with values for different L

                powers_of_t_chi = np.power(np.column_stack([t] * (m_R)),
                                           np.arange(1, m_R + 1).transpose())  # (powers of t, L data)
                # [t] * (m_R) = [t, t, t, t] (m_R=4 times)
                # [t, t^2, t^3, t^4]

                chi = np.dot(powers_of_t_chi, b)
                chi_vec = np.power(chi * (self.L ** (1 / nu)), np.column_stack(
                    [np.arange(0, n_R + 1)] * len(t)))  # vector containing 1, chi*L**1/nu, chi**2*L**2/nu, ...

                if n_I > 0:
                    powers_of_t_psi = np.power(np.column_stack([t] * (m_I + 1)),
                                               np.arange(0, m_I + 1).transpose())  # Always has 1 in the first column
                    psi = np.dot(powers_of_t_psi, c)
                    psi_vec = np.power(psi * (self.L ** y), np.column_stack([np.arange(0, n_I + 1)] * len(t)))

                    A = np.insert(A, [1, int(n_R)], 1.0)  # set F01=F10=1.0
                    A = A.reshape(n_I + 1, n_R + 1)
                    A = np.matmul(A, chi_vec)

                    lam = np.multiply(psi_vec, A)  # elementwise multiply
                    lam = np.sum(lam, axis=0)
                else:
                    A = np.insert(A, 1, 1.0)
                    lam = np.matmul(A, chi_vec)

                return lam

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
                Lambda_scaled = scalingFunc(self.Tvar, self.L, Tc, nu, y, A, b, c)
                c2 = np.sum(np.abs(self.Lambda - Lambda_scaled)**2)
                dof = len(self.Tvar) - numParams

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

            # bootstrap method  generate n sets of randints.


    randints, bLambda, bL, bTvar, bsigma = bootVals(Lambda, L, Tvar, sigma, resamplesize)  # getting new variables with prefix b to designate bootstrap resamples


    print(str(numParams) + "+3 parameters")

    Nurange=np.array([])
    Tcrange=np.array([])

    if resamplesize == 1 and bL.ndim == 1:  # fix edge case
        bL = np.expand_dims(L, 0)
        bTvar = np.expand_dims(Tvar, 0)
        bLambda = np.expand_dims(Lambda, 0)

    #list containing problem descriptions for each bootstrap resample
    problist = [pg.problem(objective_function(bL[i, :], bTvar[i, :], bLambda[i, :])) for i in range(resamplesize)]
    probbackup = problist


    #definition of the algorithm
    algo = pg.algorithm(pg.cmaes(gen=1000, force_bounds=use_bounds, ftol=1e-8))
    pg.mp_island.init_pool(processes=numCPUs)

    #print("Computing main fit")
    #pop = pg.population(prob=objective_function(L, Tvar, Lambda), size=500)
    #algo.evolve(pop)

    #best_main = pop.champion_x


    print("Starting bootstrap")
    islands = [pg.island(algo=algo, prob=problist[i], size=100, udi=pg.mp_island()) for i in range(resamplesize)]


    _ = [isl.evolve() for isl in islands]
    solutions = np.array([])
    while solutions.shape[0]<resamplesize:
        for ind, isl in enumerate(islands):
            if isl.status==pg.evolve_status.idle_error or isl.status==pg.evolve_status.busy_error:  # something blew up, try again
                islands[ind] = pg.island(algo=algo, prob=probbackup[ind], size=100, udi=pg.mp_island())
                islands[ind].evolve()
                print('Island %i had an exception, restarting...' % ind)
            if isl.status==pg.evolve_status.idle:  # completed normally, store the result
                solution = isl.get_population().champion_x
                score = isl.get_population().champion_f
                Tcrange = np.append(Tcrange, solution[0])
                Nurange = np.append(Nurange, solution[1])
                if len(solutions)==0:
                    solutions = np.expand_dims(solution, axis=0)
                else:
                    solutions = np.vstack([solutions, solution])
                print('Bootstrapping {}% Tc: {}, nu: {} from island {}'.format(round(len(solutions) / resamplesize * 100),
                                                                round(solution[0], 3),
                                                                round(solution[1], 3), ind))

                #finally, remove the island from the array so its not counted twice (or three times, ...)
                del islands[ind]
        time.sleep(1)

    _ = [isl.wait() for isl in islands]

    print(Nurange)
    print(Tcrange)


    TcChoke = np.percentile(Tcrange, [37.5, 62.5], interpolation='lower') # tight range
    Tcrangef = Tcrange[Tcrange >= TcChoke[0]]
    Nurangef = Nurange[Tcrange >= TcChoke[0]]
    Nurangef = Nurangef[Tcrangef <= TcChoke[1]]
    Tcrangef = Tcrangef[Tcrangef <= TcChoke[1]]

    nu_1CI = np.percentile(Nurange, [2.5, 97.5], interpolation='lower')
    TcCI = np.percentile(Tcrange, [2.5, 97.5], interpolation='lower')
    Tcfinal = np.median(Tcrange)
    Nufinal = np.median(Nurange) # Don't take Nufinal if you are pregnant or nursing
    print('50%ile Tc range: {}, {}'.format(round(min(Tcrangef),3), round(max(Tcrangef),3)))
    print('50%ile nu range: {}, {}'.format(round(min(Nurangef),3), round(max(Nurangef),3)))
    print('95%ile Tc range: {}, {}'.format(round(TcCI[0], 3), round(TcCI[1], 3)))
    print('95%ile nu range: {}, {}'.format(round(nu_1CI[0], 3), round(nu_1CI[1], 3)))
    print('Cc: {} [{}, {}]'.format(round(Tcfinal,3), round(TcCI[0], 3), round(TcCI[1], 3)))
    print('nu: {} [{}, {}]'.format(round(Nufinal,3), round(nu_1CI[0], 3), round(nu_1CI[1], 3)))
    #solution = best_main

    solution = np.median(solutions, axis=0)
    print("Solution"+str(solution))


    print('n_R, n_I, m_R, m_I = {}, {}, {}, {}'.format(n_R, n_I, m_R, m_I))


    plt.figure()
    plt.hist(Nurange, label=r'$\nu$', color='#1a1af980')
    plt.hist(Tcrange, label=r'$c_c$', color='r')
    plt.xlabel(r'$\nu$')
    plt.ylabel('counts')
    def plotScalingFunc(T, L, args):
        Tc = args[0]
        nu = args[1]
        y = args[2]
        the_rest = args[3:]
        A = the_rest[:int((n_I + 1) * (n_R + 1) - 2)]
        b = the_rest[int((n_I + 1) * (n_R + 1) - 2):int((n_I + 1) * (n_R + 1) - 2 + m_R)]
        c = the_rest[int((n_I + 1) * (n_R + 1) - 2 + m_R):]

        t = (T - Tc) / Tc
        powers_of_t_chi = np.power(np.column_stack([t] * (m_R)), np.arange(1, m_R + 1).transpose())  # (powers of t, L data)
        chi = np.dot(powers_of_t_chi, b)
        chi_exp_L = (np.abs(chi) ** (nu)) * L
        chi_L_exp = chi * L ** (1 / nu)
        ax1.scatter(chi_exp_L, Lambda)
        #ax2.scatter(chi_L_exp, Lambda)
        ax1.set_ylabel(r'$\Lambda$', fontsize=fs)
        # ax1.set_ylabel(r'$g$', fontsize=fs)
        ax1.set_xlabel(r'$|\chi|^\nu*L$', fontsize=fs)
        ax1.set_xscale('log')
        #ax2.set_xlabel(r'$\chi L^{1/\nu}$', fontsize=fs)
        #ax2.set_xscale('linear')

        ax1.set_yscale('log')
        #ax2.set_yscale('log')


    plotScalingFunc(Tvar, L, solution)

    # Plot raw data
    for T in np.unique(Tvar)[::-1]:
        toPlotX = []
        toPlotY = []
        for i, Tv in enumerate(Tvar):
            if T == Tv:
                toPlotY.append(Lambda[i])
                toPlotX.append(L[i])
        npX = np.array(toPlotX)
        npY = np.array(toPlotY)
        inds = npX.argsort()  # sort the data according to L to make sure it plots right
        npX = npX[inds]
        npY = npY[inds]
        if np.array_equal(Tvar,W):
            lbl='W='
        else:
            lbl='c='
        ax3.semilogy(npX, npY, 'o-', label=lbl + str(round(T,2)))
    ax3.set_xlabel('L', fontsize=fs)
    ax3.set_xscale('log')
    Lrange = np.arange(min(Lrange),max(Lrange)+1)
    ax3.set_xticks(Lrange)
    ax3.set_xticklabels(list(map(int, Lrange)))
    ax3.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1))


    #fig2, ax4 = plt.subplots(nrows=1, ncols=1, figsize=(8, 5), sharey=True)

    '''
    #Plot heatmap
    fig2, ax2 = plt.subplots()
    Tcrange = np.linspace(crit_bound_lower,crit_bound_upper,50)
    nurange = np.linspace(nu_bound_lower,nu_bound_upper,50)
    Prange = np.linspace(param_bound_lower, param_bound_upper, 50)
    Pindex = 3
    Z = np.zeros((len(Prange),len(nurange)))
    # Tc, nu, y, A0, A2, ... , b0, b1, b2, ...
    #  0,  1, 2,  3,  4
    axamp = plt.axes([0.25, .03, 0.50, 0.02])
    samp = Slider(axamp, 'Parameter', 0.0, numParams+3.1)
    def update(val):
        Pindex = int(min(val, len(solution)-1))
        for i, P in enumerate(Prange):
            for j, nu in enumerate(nurange):
                candidate = solution
                candidate[Pindex] = P
                Z[i,j] = np.log10(objective_function.fitness(objective_function,candidate))
        plt.sca(ax2)
        plt.cla()
        ax2.imshow(Z.transpose(),cmap='hot', extent=[min(Prange),max(Prange),min(nurange),max(nurange)], aspect='auto', origin='lower')
        #ax2.set_xlabel('Parameter', fontsize=fs)
        ax2.set_ylabel(r'$\nu$', fontsize=fs)
        #ax4.set_ylabel(r'$s$', fontsize=fs)
        ax2.scatter(champs[:,Pindex],champs[:,1],marker='o')
    samp.on_changed(update)
    '''



    plt.show()
