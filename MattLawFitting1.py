import numpy as np, scipy.optimize as so, csv, matplotlib.pyplot as plt
import pandas as pd 
import os

scriptdir=os.getcwd() #os.path.dirname(__file__) 
datadir= os.path.join(scriptdir, 'data\\')#static ..\data\

filename='holemobilityvals.csv'

file = pd.read_csv(datadir+filename, sep=',',header=1)
data=file.to_numpy(dtype='float')

T=data[:,0]
grains=data[:,1:]

'''
def f(T,Pre,Ea,C,D):
    #kbev=1000*8.617333262*10**-5 #meV/K
    val=Pre*np.exp(-Ea/T)/T+C*(1-np.sqrt(D/T))
    return val
'''

def g(T,sm,sh,t0,a):
    m=1
    s=1.1
    sigma=sm*(sh*np.exp(-t0/T)/sm)**s*(1/(1+a/(sh*np.exp(-t0/T)/sm)**m))
    return sigma

def h(T,pre,t0):
    rho=pre*np.exp((T/t0)**(-1/2))
    cond=1/rho
    return cond
def hh(T,pre,t0):
    rho=pre*np.exp((t0/T)**0.5)
    return 1/rho

def f(T,sio,smo,E,val,t0,a,n,s):
    if s is None:
        s=1
    if n is None:
        n=3
    kbev=1000*8.617333262*10**-5 #meV/K
    si=sio*np.exp(-E/(kbev*T))
    sm=smo/(1+a*(T/t0)**n)
    sigma=sm**(1-s)*(si**s+val*sm**s)
    return sigma

nmax=np.inf
smin=None
smax=None


vals1,errs1=so.curve_fit(f,T,grains[:,0],bounds=    ((0,0,0,0,0,smin,1,0.5),
                                                    (np.inf,np.inf,np.inf,np.inf,np.inf,smax,nmax,2)))
vals2,errs2=so.curve_fit(f,T,grains[:,1],bounds=((0,0,0,0,0,smin,1,0.5),(np.inf,np.inf,np.inf,np.inf,np.inf,smax,nmax,2)))
vals3,errs3=so.curve_fit(f,T,grains[:,2],bounds=((0,0,0,0,0,smin,1,0.5),(np.inf,np.inf,np.inf,np.inf,np.inf,smax,nmax,2)))
vals4,errs4=so.curve_fit(f,T,grains[:,3],bounds=((0,0,0,0,0,smin,1,0.5),(np.inf,np.inf,np.inf,np.inf,np.inf,smax,nmax,2)))


print(vals1)
print(vals2)
print(vals3)
print(vals4)

perr1 = np.sqrt(np.diag(errs1))
perr2 = np.sqrt(np.diag(errs2))
perr3 = np.sqrt(np.diag(errs3))
perr4 = np.sqrt(np.diag(errs4))
'''
vals1,errs1=so.curve_fit(f,T,grains[:,0],p0=[300,290,1,0.5],bounds=(0,np.inf))
vals2,errs2=so.curve_fit(f,T,grains[:,1],p0=[300,290,1,0.5],bounds=(0,np.inf))
vals3,errs3=so.curve_fit(f,T,grains[:,2],p0=[300,290,1,0.5],bounds=(0,np.inf))
vals4,errs4=so.curve_fit(f,T,grains[:,3],p0=[300,290,1,0.5],bounds=(0,np.inf))

perr1 = np.sqrt(np.diag(errs1))
perr2 = np.sqrt(np.diag(errs2))
perr3 = np.sqrt(np.diag(errs3))
perr4 = np.sqrt(np.diag(errs4))

'''

fs=18
plt.close()
plt.plot(T, grains[:,0], 'r', label='Grain 1')
plt.plot(T, grains[:,1], 'b', label='Grain 2')
plt.plot(T, grains[:,2], 'g', label='Grain 3')
plt.plot(T, grains[:,3], 'k', label='Grain 4')

'''
plt.plot(T, f(T, *vals1), 'r--',label='Fit 1, Ea={:.2f} meV'.format(Ea1))
plt.plot(T, f(T, *vals2), 'b--',label='Fit 2, Ea={:.2f} meV'.format(Ea2))
plt.plot(T, f(T, *vals3), 'g--',label='Fit 3, Ea={:.2f} meV'.format(Ea3))
plt.plot(T, f(T, *vals4), 'k--',label='Fit 4, Ea={:.2f} meV'.format(Ea4))
'''

plt.plot(T, f(T, *vals1), 'r--',label='Fit 1, n={}'.format(vals1[-2]))
plt.plot(T, f(T, *vals2), 'b--',label='Fit 2, n={}'.format(vals2[-2]))
plt.plot(T, f(T, *vals3), 'g--',label='Fit 3, n={}'.format(vals3[-2]))
plt.plot(T, f(T, *vals4), 'k--',label='Fit 4, n={}'.format(vals4[-2]))
plt.yscale('log')
plt.title('Fit with Gergely Notes',fontsize=24)
plt.legend(loc='lower center',ncol=2)
plt.ylabel(r'$\mu$',fontsize=fs)
plt.xlabel(r'T (K)',fontsize=fs)

plt.show()
print('done')