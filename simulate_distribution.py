import numpy as np
import matplotlib.pyplot as plt

num_points = 100000

W = 10
eps_i = W*(np.random.random(num_points)-0.5)

mu = 0.25
sigma = 0.03
t1 = np.random.normal(mu, sigma, num_points)

c = 0.4
th = 1
tl = 0.3
t2 = np.zeros_like(eps_i)
for i in range(num_points):
    if np.random.random() < c:
        t2[i] = tl
    else:
        t2[i] = th
normal = eps_i/t1
kindis = eps_i/t2
anderson = eps_i/1.0
plt.hist(normal, bins='auto', label='t: normal dist', alpha=0.8, range=[-30, 30])
plt.hist(kindis, bins='auto', label='t: kinetic dist', alpha=0.8, range=[-30, 30])
plt.hist(anderson, bins='auto', label='t: 1 (Anderson)', alpha=0.5, range=[-30, 30])
plt.title("W=%.1f, normal: mu=%.2f, sigma=%.2f, kinetic: c=%.2f, t_h=%.1f, t_l=%.1f" % (W, mu, sigma, c, th, tl))
maxnum = 3000
plt.vlines(8.25, 0, maxnum, colors='r', linestyles='dashed')
plt.vlines(-8.25, 0, maxnum, colors='r', linestyles='dashed')

nfrac = np.sum(np.abs(normal)>8.25)/len(normal)
kfrac = np.sum(np.abs(kindis)>8.25 )/len(kindis)
afrac = np.sum(np.abs(anderson)>8.25)/len(anderson)

print("fraction of normal dist > W_c: "+str(nfrac))
print("fraction of kindis dist > W_c: "+str(kfrac))
print("fraction of anderson dist > W_c: "+str(afrac))

plt.legend()
plt.xlabel("W/t")
plt.show()