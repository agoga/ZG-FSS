import numpy as np
import matplotlib.pyplot as plt
import pygmo as pg

from scipy import stats
from scipy import optimize
from scipy import special
from scipy.stats import chisquare

import gc

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
if __name__ == '__main__':
    fig1, (ax1, ax3) = plt.subplots(nrows=1, ncols=2, figsize=(11, 6), sharey=True)
    fs = 18 #font size
    filename="offdiagE4W14.txt"
    minL = 8
    input = np.array(openfile(filename))
    #input = np.array(openfileZeke("gcL.txt"))
    Lrange = np.unique(input[:, 0])
    Wrange = np.unique(input[:, 1])
    crange = np.unique(input[:, 2])


    data = input[:, 0:5]  # L, W, c, LE
    data[:, 3] = 1 / (data[:, 0] * data[:, 3])  # L, W, c, normalized localization length

    # sort according to L
    data = data[np.argsort(data[:, 0])]
    #omit L less than minL to control finite size effects
    data = data[data[:,0]>=minL]

    Lambda = data[:, 3]
    L = data[:, 0]
    W = data[:, 1]
    c = data[:, 2]
    sigma = data[:, 4] #uncomment for MacKinnon
    # set the driving parameter
    Tvar = c

    crit_bound_lower1, crit_bound_upper1 = min(c), max(c) # critical value bounds
    #crit_bound_lower, crit_bound_upper =0.6, 0.62
    crit_bound_lower2, crit_bound_upper2 = min(c), max(c)
    nu1_bound_lower, nu1_bound_upper = 1.57, 1.59  # nu1 bounds
    nu2_bound_lower, nu2_bound_upper = 1.1, 1.7  # nu2 bounds
    y_bound_lower, y_bound_upper = -10.0, -0.1   # y bounds
    param_bound_lower, param_bound_upper = -10.0, 10.1  # all other bounds

    # orders of expansion
    n_R = 1
    n_I = 0
    m_R = 3
    m_I = 0

    #numBoot = len(Lambda)//2
    numBoot = 1

    if n_I > 0:
        numParams = (n_I + 1) * (n_R + 1) + (n_I + 1) * (n_R + 1) - 1 + 2*m_R + m_I
    else:
        numParams = 2*n_R + 2*m_R + 2
    print(str(numParams) + "+4 parameters")


    if numBoot==1:
        bootSize=1
    else:
        bootSize = 0.9 #fraction of data to keep
    bootResults=[]
    Lambda_restart = Lambda
    L_restart = L
    Tvar_restart = Tvar
    sigma_restart = sigma

    class objective_function:

        def fitness(self, x):
            def scalingFunc(T, L, Tc1, Tc2, nu1, nu2, y, A1, b1, A2, b2, c):
                t1 = (T - Tc1) / Tc1  # This is filled with values for different L
                t2 = (T - Tc2) / Tc2

                powers_of_t_chi1 = np.power(np.column_stack([t1] * (m_R)),
                                            np.arange(1, m_R + 1).transpose())  # (powers of t, L data)
                chi1 = np.dot(powers_of_t_chi1, b1)
                chi1_vec = np.power(chi1 * L ** (1 / nu1), np.column_stack(
                    [np.arange(0, n_R + 1)] * len(t1)))  # vector containing 1, chi*L**1/nu, chi**2*L**2/nu, ...

                powers_of_t_chi2 = np.power(np.column_stack([t2] * (m_R)),
                                            np.arange(1, m_R + 1).transpose())  # (powers of t, L data)
                chi2 = np.dot(powers_of_t_chi2, b2)
                chi2_vec = np.power(chi2 * L ** (1 / nu2), np.column_stack(
                    [np.arange(0, n_R + 1)] * len(t2)))  # vector containing 1, chi*L**1/nu, chi**2*L**2/nu, ...
                if n_I > 0:
                    powers_of_t_psi = np.power(np.column_stack([t1] * (m_I + 1)),
                                               np.arange(0, m_I + 1).transpose())  # Always has 1 in the first column
                    psi = np.dot(powers_of_t_psi, c)
                    psi_vec = np.power(psi * L ** y, np.column_stack([np.arange(0, n_I + 1)] * len(t1)))

                    A1 = np.insert(A1, [1, int(n_R)], 1.0)  # set F01=F10=1.0
                    A1 = A1.reshape(n_I + 1, n_R + 1)
                    A1 = np.matmul(A1, chi1_vec)

                    A2 = A2.reshape(n_I + 1, n_R + 1)
                    A2 = np.matmul(A2, chi2_vec)

                    lam1 = np.multiply(psi_vec, A1)  # elementwise multiply
                    lam2 = np.multiply(psi_vec, A2)
                    lam = np.sum(lam1, axis=0) + np.sum(lam2, axis=0)
                else:
                    A1 = np.insert(A1, 1, 1.0)
                    lam = np.matmul(A1, chi1_vec) + np.matmul(A2, chi2_vec)
                return lam

            def objective(args):
                # this is where we specify the loss function
                Tc1 = args[0]
                Tc2 = args[1]
                nu1 = args[2]
                nu2 = args[3]
                y = args[4]
                the_rest = args[5:]
                if n_I > 0:
                    A1 = the_rest[:int((n_I + 1) * (n_R + 1) - 2)]
                    b1 = the_rest[int((n_I + 1) * (n_R + 1) - 2):int((n_I + 1) * (n_R + 1) - 2 + m_R)]
                    A2 = the_rest[
                         int((n_I + 1) * (n_R + 1) - 2 + m_R):int((n_I + 1) * (n_R + 1) - 2 + (n_I + 1) * (n_R + 1) + m_R)]
                    b2 = the_rest[int((n_I + 1) * (n_R + 1) - 2 + (n_I + 1) * (n_R + 1) + m_R):int(
                        (n_I + 1) * (n_R + 1) - 2 + (n_I + 1) * (n_R + 1) + 2 * m_R)]
                    c = the_rest[int((n_I + 1) * (n_R + 1) - 2 + (n_I + 1) * (n_R + 1) + 2 * m_R):]

                else:
                    A1 = the_rest[:int(n_R)]
                    b1 = the_rest[int(n_R):int(n_R + m_R)]
                    A2 = the_rest[int(n_R + m_R):int(2 * n_R + m_R + 1)]
                    b2 = the_rest[int(2 * n_R + m_R + 1):int(2 * n_R + 2 * m_R + 1)]
                    c = the_rest[int(2 * n_R + 2 * m_R + 1):]  # will not matter. Set m_I to 0 if n_I is 0
                Lambda_scaled = scalingFunc(Tvar, L, Tc1, Tc2, nu1, nu2, y, A1, b1, A2, b2, c)

                c2 = np.sum((Lambda-Lambda_scaled)**2)
                return c2

            return [objective(x)]

        def get_bounds(self):
            bounds = ([[crit_bound_lower1, crit_bound_upper1],
                       [crit_bound_lower2, crit_bound_upper2],
                               [nu1_bound_lower, nu1_bound_upper],
                               [nu2_bound_lower, nu2_bound_upper],
                               [y_bound_lower, y_bound_upper]]
                              + [[param_bound_lower, param_bound_upper]] * (numParams))
            bounds = np.transpose(bounds)
            return (bounds[0,:], bounds[1,:])


    Lambda = Lambda_restart
    L = L_restart
    Tvar = Tvar_restart

    #bootstrap method
    #randInds = np.random.choice(len(Lambda), int(len(Lambda)*bootSize), replace=False)
    #Lambda=Lambda[randInds]
    #L = L[randInds]
    #Tvar = Tvar[randInds]



    prob = pg.problem(objective_function())

    algo = pg.algorithm(pg.pso(gen=3000, eta1=1.1, eta2=3.0))
    #algo = pg.algorithm(pg.bee_colony(gen=500, limit=20))
    pop = pg.population(prob, 100)
    algo.set_verbosity(100)
    #archi = pg.archipelago(n=1,algo=algo,prob=prob, pop_size=50)
    pop = algo.evolve(pop)
    print(pop)
    #archi.evolve()
    #print(archi)
    #archi.wait()

    #result = optimize.shgo(objective, bounds, iters=3, sampling_method='sobol', n=15, options={'disp': True})
    #result = optimize.dual_annealing(objective, bounds, maxiter=15000, initial_temp=10000)
    #result = optimize.differential_evolution(objective, bounds, maxiter=5000, disp=True, strategy='best1exp', mutation=0.2)
    #result = optimize.basinhopping(objective, bounds, niter=100, disp=True)


    #print(str(t+1)+"/"+str(numBoot))
    #bootResults.append(result['x'])
    gc.collect()


    Lambda = Lambda_restart
    sigma = sigma_restart
    L = L_restart
    Tvar = Tvar_restart
    '''
    bootResults = np.stack(bootResults, axis=0)
    chi2Results = np.zeros(len(bootResults))
    for i in range(len(chi2Results)):
        chi2Results[i] = objective(bootResults[i])
    minInd=np.argmin(chi2Results)
    minChi2 = chi2Results[minInd]
    '''

    #solution = np.mean(bootResults,axis=0)

    #best_solution = archi.get_champions_x()[int(np.amin(archi.get_champions_f()))]
    best_solution = pop.get_x()[pop.best_idx()]
    print("Best solution"+str(best_solution))

    #nu1CI = np.percentile(bootResults[...,1], [2.5, 97.5], interpolation='lower')
    #nu2CI = np.percentile(bootResults[...,2], [2.5, 97.5], interpolation='lower')
    #TcCI = np.percentile(bootResults[...,0], [2.5, 97.5])
    #medianNu = np.median(bootResults[...,0])
    #medianTc = np.median(bootResults[...,1])
    #champs = np.array(archi.get_champions_x())
    champs = np.array(pop.get_x())
    Tc1s = champs[:,0]
    Tc2s = champs[:,1]
    nu_1s = champs[:,2]
    nu_2s = champs[:,3]
    print('nu_1: '+str(best_solution[2])+' ('+str(nu_1s)+')')
    print('nu_2: '+str(best_solution[3])+' ('+str(nu_2s)+')')

    print('Tc_1: %f (%f, %f)' % (best_solution[0], np.min(Tc1s), np.max(Tc1s)))
    print('Tc_2: %f (%f, %f)' % (best_solution[1], np.min(Tc2s), np.max(Tc2s)))
    print('nu_1: %f (%f, %f)' % (best_solution[2], np.min(nu_1s), np.max(nu_1s)))
    print('nu_2: %f (%f, %f)' % (best_solution[3], np.min(nu_2s), np.max(nu_2s)))

    '''
    print('n_R, n_I, m_R, m_I = {}, {}, {}, {}'.format(n_R, n_I, m_R, m_I))
    print('Critical value: %f' % (pop.champion_x[0]))
    print('nu_1: %f' % (pop.champion_x[1]))
    print('nu_2: %f' % (pop.champion_x[2]))
    print('File: '+filename)
    '''

    def plotScalingFunc(T, L, args):
        Tc1 = args[0]
        Tc2 = args[1]
        nu1 = args[2]
        nu2 = args[3]
        y = args[4]
        the_rest = args[5:]

        A1 = the_rest[:int((n_I + 1) * (n_R + 1) - 2)]
        b1 = the_rest[int((n_I + 1) * (n_R + 1) - 2):int((n_I + 1) * (n_R + 1) - 2 + m_R)]
        A2 = the_rest[int((n_I + 1) * (n_R + 1) - 2 + m_R):int((n_I + 1) * (n_R + 1) - 2 + (n_I + 1) * (n_R + 1) - 2 + m_R)]
        b2 = the_rest[int((n_I + 1) * (n_R + 1) - 2 + (n_I + 1) * (n_R + 1) - 2 + m_R):int(
            (n_I + 1) * (n_R + 1) - 2 + (n_I + 1) * (n_R + 1) - 2 + 2 * m_R)]
        c = the_rest[int((n_I + 1) * (n_R + 1) - 2 + (n_I + 1) * (n_R + 1) - 2 + 2 * m_R):]
        t1 = (T - Tc1) / Tc1
        t2 = (T - Tc2) / Tc2

        powers_of_t_chi1 = np.power(np.column_stack([t1] * (m_R)), np.arange(1, m_R + 1).transpose())  # (powers of t, L data)
        powers_of_t_chi2 = np.power(np.column_stack([t2] * (m_R)), np.arange(1, m_R + 1).transpose())  # (powers of t, L data)
        chi1 = np.dot(powers_of_t_chi1, b1)
        chi2 = np.dot(powers_of_t_chi2, b2)

        chi_L_exp = chi1 * L ** (1 / nu1) + chi2 * L ** (1/nu2)
        chi_exp_L = (np.abs(chi1) ** (nu1)) * L + (np.abs(chi2) ** (nu2)) * L

        ax1.scatter(chi_exp_L, Lambda)
        #ax2.scatter(chi_L_exp, Lambda)
        ax1.set_ylabel(r'$\Lambda$', fontsize=fs)
        # ax1.set_ylabel(r'$g$', fontsize=fs)
        ax1.set_xlabel(r'$(|\chi_1|^{\nu_1}+|\chi_2|^{\nu_2})*L$', fontsize=fs)
        ax1.set_xscale('log')
        #ax2.set_xlabel(r'$\chi L^{1/\nu}$', fontsize=fs)
        #ax2.set_xscale('linear')

        ax1.set_yscale('log')
        #ax2.set_yscale('log')


    plotScalingFunc(Tvar, L, best_solution)

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


    #Plot heatmap
    '''
    Tcrange = np.linspace(crit_bound_lower,crit_bound_upper,50)
    nurange = np.linspace(nu_bound_lower,nu_bound_upper,50)
    
    Z = np.zeros((len(Tcrange),len(nurange)))
    
    for i, Tc in enumerate(Tcrange):
        for j, nu in enumerate(nurange):
            Z[i,j] = np.log10(objective(np.concatenate([np.array([Tc, nu]), solution[2:]])))
    
    heatmap = ax4.imshow(Z.transpose(),cmap='hot', extent=[min(Tcrange),max(Tcrange),min(nurange),max(nurange)], aspect='auto', origin='lower')
    
    ax4.set_xlabel('critical W', fontsize=fs)
    ax4.set_ylabel(r'$\nu$', fontsize=fs)
    #ax4.set_ylabel(r'$s$', fontsize=fs)
    ax4.scatter(solution[0],solution[1],marker='o')
    plt.colorbar(heatmap, ax=ax4)
    '''
    '''
    ax4.hist(chi2Results)
    ax4.set_ylabel('Count', fontsize=fs)
    ax4.set_xlabel(r'$\chi^2$', fontsize=fs)
    '''
    plt.show()
