import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
fs = 18
                    # E, W, nu_lowerCI, nu, nu_upperCI, C_crit
all_data = np.array([  [2, 6, 0.985, 1.015, 1.044, 0.061],
                       [2, 10, 1.117, 1.142, 1.167, 0.31],
                       [2, 12, 1.150, 1.177, 1.206, 0.44],
                       [2, 14, 1.196, 1.253, 1.306, 0.586],
                       [2, 16, 1.187, 1.368, 1.528, 0.764],
                       [4, 10, 1.160, 1.182, 1.205, 0.386],
                       [4, 12, 1.174, 1.201, 1.230, 0.492],
                       [4, 14, 1.134, 1.225, 1.271, 0.621],
                       [4, 16, 1.390, 1.484, 1.588, 0.791],
                       [6, 10, 1.302, 1.356, 1.398, 0.635],
                       [6, 12, 1.261, 1.326, 1.371, 0.64],
                       [6, 14, 1.272, 1.382, 1.445, 0.743],
                       [6, 15, 1.331, 1.443, 1.523, 0.80],
                       [6, 16, 1.574, 1.702, 1.891, 0.883],
                       [5, 10, 1.165, 1.209, 1.246, 0.46]])

Erange = np.unique(all_data[:,0])
Wrange = np.unique(all_data[:,1])
fig1, (ax1, ax2, ax3) = plt.subplots(nrows=1, ncols=3, figsize=(16, 6))
fig2 = plt.figure(figsize=(10,8))
fig2.tight_layout()
ax4 = fig2.add_subplot(111, projection='3d')

for E in Erange:
    Edata = all_data[all_data[:,0]==E]
    Edataerrors = np.abs([Edata[:,2]-Edata[:,3], Edata[:,4]-Edata[:,3]])
    legendlabel = 'E='+str(E)
    ax1.plot(Edata[:,1], Edata[:,5],'o-',label=legendlabel)
    ax2.errorbar(Edata[:, 1], Edata[:, 3], yerr=Edataerrors, fmt='o-', capsize=3, capthick=2, label=legendlabel)
    ax3.errorbar(Edata[:,5], Edata[:,3],yerr=Edataerrors, fmt='o-',capsize=3,capthick=2,label=legendlabel)

ax1.set_ylabel(r'$C_{c}$', fontsize=fs)
ax1.set_xlabel('W', fontsize=fs)
ax1.legend()
ax2.set_ylabel(r'$\nu$', fontsize=fs)
ax2.set_xlabel('W', fontsize=fs)
ax3.set_ylabel(r'$\nu$', fontsize=fs)
ax3.set_xlabel(r'$C_{c}$', fontsize=fs)

A = np.zeros((Wrange.size, Erange.size))
for i,W in enumerate(Wrange):
    for j,E in enumerate(Erange):
        mask = np.logical_and((all_data[:,0]==E),(all_data[:,1]==W))
        if np.squeeze(all_data[mask]).any():
            A[i,j]=np.squeeze(all_data[mask])[3]

Wplt, Eplt = np.meshgrid(Wrange,Erange, indexing='xy')

stretch_factor = (Erange.max()-Erange.min())/(Wrange.max() - Wrange.min())
size=1.0

#Colors
nonzeroA = A[A!=0]
leastnonzero = np.amin(nonzeroA)
norm = cm.colors.Normalize(vmin=leastnonzero, vmax=np.amax(A), clip=True)
mapper = cm.ScalarMappable(norm=norm, cmap=cm.terrain)
barcolors = mapper.to_rgba(A.T).squeeze()

barcolors = barcolors.reshape(-1,barcolors.shape[0])

B=A.T
#B[B==0]=leastnonzero
B=B
bottom = np.zeros_like(A.T)
print(B)
print(bottom)

bars = ax4.bar3d(x=Eplt.ravel(), y=Wplt.ravel(), z=bottom.ravel(), dx=size*stretch_factor, dy=size, dz=B.ravel(),
           shade=True, color=barcolors)
ax4.set_xlabel('E', fontsize=fs)
ax4.set_ylabel('W', fontsize=fs)
ax4.set_zlabel(r'$\nu$', fontsize=fs)
#ax4.set_zticks(np.linspace(leastnonzero,np.amax(A),5).round(2))
#for i, txt in enumerate(all_data[:,3]):
#    ax4.annotate(str(txt), (all_data[i,1], all_data[i,0]))
#ax4.imshow(A, interpolation=None, extent=[Wrange.min(), Wrange.max(), Erange.min(), Erange.max()])
fig2.colorbar(mapper, ax=ax4)
plt.show()