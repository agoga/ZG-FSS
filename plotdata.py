#Mike Kovtun's plotting only code
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
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

fig1, ax3 = plt.subplots(nrows=1, ncols=1, figsize=(11, 6), sharey=True)
fs = 18 #font size
filename="data\\offdiagE2W8.txt"
minL = 8


input = np.array(openfile(filename))
Lrange = np.unique(input[:, 0])
Wrange = np.unique(input[:, 1])
crange = np.unique(input[:, 2])


data = input[:, 0:5]  # L, W, c, LE
data[:, 3] = 1 / (data[:, 0] * data[:, 3])  # L, W, c, normalized localization length

# sort according to L
data = data[np.argsort(data[:, 0])]


Lambda = data[:, 3]
L = data[:, 0]
W = data[:, 1]
c = data[:, 2]
sigma = data[:, 4] #uncomment for MacKinnon
# set the driving parameter
Tvar = c

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
    ax3.semilogy(npX, npY, 'o-', label=lbl + str(round(T,3)))
ax3.set_xlabel('L', fontsize=fs)
ax3.set_ylabel(r'$\Lambda$',fontsize=fs)
ax3.set_xscale('log')
ax3.set_title('E2W8',fontsize=fs*1.5)
Lrange = np.arange(min(Lrange),max(Lrange)+1)
ax3.set_xticks(Lrange)
ax3.set_xticklabels(list(map(int, Lrange)))
ax3.legend()


plt.show()
