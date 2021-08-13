import numpy as np
import matplotlib.pyplot as plt

from scipy import stats
from scipy import optimize
from scipy import special

from scipy.stats import chi2
import gc
import pygmo as pg

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
    print("openfileZeke")
    print(outputlst)
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

    print("openfileZeke")
    print(outputlst)
    return outputlst




fig1, (ax1, ax3) = plt.subplots(nrows=1, ncols=2, figsize=(11, 6), sharey=True)
fs = 18 #font size
filename="offdiagE4W14.txt"
minL = 8
window_width = 0.1 #width of window
window_offset = 0.00 #distance from window center to near edge of window
window_center = 0.6

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
data = data[np.abs(data[:,2]-window_center)<=window_offset+window_width]
data = data[np.abs(data[:,2]-window_center)>=window_offset]

Lambda = data[:, 3]
L = data[:, 0]
W = data[:, 1]
c = data[:, 2]
sigma = data[:, 4] #uncomment for MacKinnon
# set the driving parameter
Tvar = c

#crit_bound_lower, crit_bound_upper = 16.0, 17.0  # critical value bounds
crit_bound_lower, crit_bound_upper = 0.6, 0.62 # critical value bounds
nu_bound_lower, nu_bound_upper = 1.05, 2.0  # nu bounds
y_bound_lower, y_bound_upper = -10.0, -0.1  # y bounds
param_bound_lower, param_bound_upper = -10.0, 10.1  # all other bounds

# orders of expansion
n_R = 3
n_I = 1
m_R = 2
m_I = 1

numBoot = len(Lambda)//4
numBoot = 1

if n_I > 0:
    numParams = (n_I + 1) * (n_R + 1) + m_R + m_I - 1
else:
    numParams = n_R + m_R
print(str(numParams) + "+3 parameters")


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
        def scalingFunc(T, L, Tc, nu, y, A, b, c):
            t = (T - Tc) / Tc  # This is filled with values for different L

            powers_of_t_chi = np.power(np.column_stack([t] * (m_R)),
                                       np.arange(1, m_R + 1).transpose())  # (powers of t, L data)
            chi = np.dot(powers_of_t_chi, b)
            chi_vec = np.power(chi * L ** (1 / nu), np.column_stack(
                [np.arange(0, n_R + 1)] * len(t)))  # vector containing 1, chi*L**1/nu, chi**2*L**2/nu, ...
            if n_I > 0:
                powers_of_t_psi = np.power(np.column_stack([t] * (m_I + 1)),
                                           np.arange(0, m_I + 1).transpose())  # Always has 1 in the first column
                psi = np.dot(powers_of_t_psi, c)
                psi_vec = np.power(psi * L ** y, np.column_stack([np.arange(0, n_I + 1)] * len(t)))

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
            Lambda_scaled = scalingFunc(Tvar, L, Tc, nu, y, A, b, c)
            c2 = np.sum(np.abs(Lambda - Lambda_scaled))
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

algo = pg.algorithm(pg.pso(gen=3000, eta1=1.1, eta2=3.0))
#algo = pg.algorithm(pg.bee_colony(gen=500, limit=20))
pop = pg.population(prob, 100)
algo.set_verbosity(100)
#archi = pg.archipelago(n=1,algo=algo,prob=prob, pop_size=50)
pop = algo.evolve(pop)
print(pop)

solution = pop.get_x()[pop.best_idx()]


Lambda = Lambda_restart
sigma = sigma_restart
L = L_restart
Tvar = Tvar_restart







print("Best solution"+str(solution))
print("Best solution normalized difference squared: "+str(pop.get_f()[pop.best_idx()]/len(Lambda)))
'''
nuCI = np.percentile(bootResults[...,1], [2.5, 97.5], interpolation='lower')
TcCI = np.percentile(bootResults[...,0], [2.5, 97.5])
medianNu = np.median(bootResults[...,0])
medianTc = np.median(bootResults[...,1])
'''


print('n_R, n_I, m_R, m_I = {}, {}, {}, {}'.format(n_R, n_I, m_R, m_I))
champs = np.array(pop.get_x())
Tcs = champs[:,0]
nu_1s = champs[:,1]
print('Tc: %f (%f, %f)' % (solution[0], np.min(Tcs), np.max(Tcs)))
print('nu: %f (%f, %f)' % (solution[1], np.min(nu_1s), np.max(nu_1s)))
print('File: '+filename)

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
