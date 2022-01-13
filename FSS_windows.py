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
import time
from datetime import datetime

#testing

now = datetime.now()
startt=time.time()

#parser = argparse.ArgumentParser(description='Window args')
#parser.add_argument("in_wc",type=int,...,required=False)
#parser.add_argument("in_ww",type=int,required=False)
#parser.add_argument("in_wo",type=int,required=False)
#args=parser.parse_args()


def openfile(filename):
    f = open(filename, "r")
    outputlst = []
    for line in f:
        if line:
            strlist = line.split()
            slen = len(strlist)
            L = float(strlist[4][:-1])
            W = float(strlist[5][:-1])

            if slen == 16:
                c = float(strlist[10][:-1])
                gaus = float(strlist[7][:-2])
                lyap = float(strlist[13][1:])
                sem = float(strlist[14][:-1])
            outputlst.append([L, W, c, lyap, sem, gaus])
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

def showplt(plt, show):
    if show:
        plt.show()
fs = 18 #font size


#filename="offdiagE6W10.txt"#dataset from localization script, up of box

#A 
#small L and finite size effect
minL = 8#slice off lowest L, tip of the data
#@TODO adam fix a loop for ranges 
repeats=1000

show=False
forceBounds=False

if show is False:
    plt.ioff()
#crit_bound_lower, crit_bound_upper = 16.0, 17.0  # critical value bounds
#A looking for transition from lower to upper
crit_bound_lower, crit_bound_upper = 0.01, 1 # critical value bounds
nu_bound_lower, nu_bound_upper = 1.05, 2.1  # nu bounds
y_bound_lower, y_bound_upper = -10.0, -0.1  # y bounds
param_bound_lower, param_bound_upper = -10.0, 10.1  # all other bounds

# orders of expansion
n_R = 3
n_I = 1
m_R = 2
m_I = 1

#@TODO allow multiple files and parameters so we can do logarithmic analysis
#same W but more precise c values
datafile='E0W10normal_07_01.txt'

window_center = 1
window_offset = 0.00#  distance from window center to near edge of window
window_width = 1 # width of window

#@TODO fire myself for param loop 
if len(sys.argv) > 1:
    datafile=str(sys.argv[1])
    window_center = float(sys.argv[2])
    window_offset = float(sys.argv[3])
    window_width = float(sys.argv[4])


filename  = cfg.datafilename(datafile)#wrapper required
input = np.array(openfile(filename))

Lrange = np.unique(input[:, 0])
Wrange = np.unique(input[:, 1])
crange = np.unique(input[:, 2])
gausrange= np.unique(input[:, 5])

data = input[:, 0:6]  # L, W, c, LE
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
gaus = data[:,5]

# set the driving parameter
Tvar = gaus

#when L is

numBoot = len(Lambda)//4
numBoot = 1





if n_I > 0:
    numParams = (n_I + 1) * (n_R + 1) + m_R + m_I - 1
else:
    numParams = n_R + m_R
#print(str(numParams) + "+3 parameters")


if numBoot==1:
    bootSize=1
else:
    bootSize = 0.9 #fraction of data to keep
bootResults=[]
Lambda_restart = Lambda
L_restart = L
Tvar_restart = Tvar
sigma_restart = sigma


if __name__ == '__main__':# or len(sys.argv) > 1:
    print("center:%f offset: %f - width: %f" % (window_center, window_offset,window_width))
    fig1, (ax1, ax3) = plt.subplots(nrows=1, ncols=2, figsize=(11, 6), sharey=True)
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

    '''
    Lambda = Lambda_restart
    L = L_restart
    Tvar = Tvar_restart
    sigma = sigma_restart

    #bootstrap method
    randInds = np.random.choice(len(Lambda), int(len(Lambda)*bootSize), replace=False)
    Lambda=Lambda[randInds]
    L = L[randInds]
    Tvar = Tvar[randInds]
    sigma = sigma[randInds]
    '''

    prob = pg.problem(objective_function())

    #algo = pg.algorithm(pg.pso(gen=1000, eta1=3.2, eta2=2.0))
    #algo = pg.algorithm(pg.de(gen=1000, F=0.5, CR=0.9, variant=4, ftol=1e-06))

    #A force_bounds ignores the boundaries
    #cmaes is the gold standard of fitting
    
    algo = pg.algorithm(pg.cmaes(gen=repeats, force_bounds=forceBounds))

    #pop = pg.population(prob, 100)
    #algo.set_verbosity(100)

    #@TODO Adam
    #A n is the number of threads
    archi = pg.archipelago(n=1, algo=algo,prob=prob, pop_size=100)
    #pop = algo.evolve(pop)

    archi.evolve()
    archi.wait()
    #print(pop)

    #solution = pop.get_x()[pop.best_idx()]

    solution = archi.get_champions_x()[int(np.amin(archi.get_champions_f()))]


    #the full set of nu
    fullsolutions = archi.get_champions_x()

    #champs = np.array(pop.get_x())
    champs = np.array(archi.get_champions_x())
    Lambda = Lambda_restart
    sigma = sigma_restart
    L = L_restart
    Tvar = Tvar_restart

    costpp = np.min(archi.get_champions_f())/len(Lambda)
    print("Best solution"+str(solution))
    #print("Best solution cost/pt: "+str(pop.get_f()[pop.best_idx()]/len(Lambda)))
    print("Best solution cost/pt: "+str(costpp))


    print('n_R, n_I, m_R, m_I = {}, {}, {}, {}'.format(n_R, n_I, m_R, m_I))

    Tcs = champs[:,0]
    TcCI = np.percentile(Tcs, [2.5, 97.5], interpolation='lower')
    nu_1s = champs[:,1]
    nu_1CI = np.percentile(nu_1s, [2.5, 97.5], interpolation='lower')
    print('File: '+filename)
    print('Tc: %f [%f, %f]' % (solution[0], np.min(Tcs), np.max(Tcs)))
    print('nu: %f [%f, %f]' % (solution[1], np.min(nu_1s), np.max(nu_1s)))


    #histogram of nu values
    plt.figure()
    plt.hist(nu_1s, label=r'$\nu$', color='#1a1af980')
    plt.xlabel(r'$\nu$')
    plt.ylabel('counts')
    #@TODO Adam

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


    #plt.savefig(cfg.outputfilename('histogram'))
    #fig2, ax4 = plt.subplots(nrows=1, ncols=1, figsize=(8, 5), sharey=True)



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





#output
    endt=time.time()
    exet=str("%.2fs" % (endt-startt))
    print('execution time: ', exet)

    nuval=round(solution[1],2)
    cc=round(solution[0],2)
    print(nuval)
    #plt.title


    #cost per point @todo
    minL = 8#slice off lowest L, tip of the data


    
    ostr=str(float(window_offset)).split('.')[1]
    wstr=str(float(window_width)).split('.')[1]
    if nuval > 1:
        nustr=str(round(nuval,3)).replace('.','_')
    else:
        nustr=str(round(nuval,3)).split('.')[1]

    fname='nu_%s--O_%s-W_%s' % (nustr,ostr,wstr)
    

    if window_offset != 0:
        cc='--'
    fig1.suptitle(datafile.removesuffix('.txt')+ "  cpp:%.2f   %s\n\nCc: %s - nu: %f - offset: %.2f - width: %.2f" % (costpp,exet,cc, nuval, window_offset,window_width))
    run_file_name=cfg.runfilename(fname + '.pdf')
    fig1.savefig(run_file_name)


    #datastring='%f, %f, %f, %f, %f' % (solution[0], solution[1], window_center, window_width, window_offset)
    csv_column_names=['Date','Run time','Cc','Nu','Cost per point','Window center',
                    'Window width','Window offset','Repeats','Forced bounds?','Crit bound lower',
                    'Crit bound upper','Nu bound lower','Nu bound upper','n_R','n_I','m_R','m_I','Plot pdf']

    datacsv=[now.strftime("%D %H:%M"), exet.removesuffix('s'), cc, nuval, costpp, window_center, window_width, 
    window_offset,repeats,forceBounds,crit_bound_lower,crit_bound_upper,nu_bound_lower,nu_bound_upper,n_R,n_I,m_R,m_I,run_file_name]
    cfg.savecsv(datacsv,csv_column_names)


    showplt(plt,show)