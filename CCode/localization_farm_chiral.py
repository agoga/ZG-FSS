import numpy as np
import time
import sys
import datetime
import matplotlib.pyplot as plt
from scipy.linalg import circulant
from scipy.optimize import curve_fit
from scipy import stats
from numpy import linalg as la

def save(par, return_value, avg, name):
	now = datetime.datetime.now()
	f = open(str(name),"a")
	line = str(now)+" "+str(par)+" "+str(return_value)+" "+str(avg)+"\n"
	f.write(line)
	
def scaling_func(tup,*params):
	W=[x[0] for x in tup]
	L=[x[1] for x in tup]
	
	#params: Wc, a00, alpha1, b10, b11, b12, alpha2, b20, b21, b22
	#one relevant scaling variable (2nd order), one irrelevant (2nd order)
	(Wc, a00, alpha1, b10, b11, b12, alpha2, b20, b21, b22) = params
	w=(W-Wc)/Wc
	psi_1 = L**alpha1*(b10+b11*w+b12*w**2)
	psi_2 = L**alpha2*(b20+b21*w+b22*w**2)
	return a00+psi_1+psi_2 #linear for now. 


def makeHamiltonian2D(L,V,W):
	#Generates the Hamiltonian (incl. site disorder) in 2D
	H=np.zeros((L,L))
	np.fill_diagonal(H,W*(np.random.rand(1,L)-0.5)) #Random site energies
	for i in range(L):
		if i==0:
			H[i,i+1]=V
		elif i==L-1:
			H[i,i-1]=V
		else:
			H[i,i+1]=V
			H[i,i-1]=V
	H[L-1,0]=V
	H[0,L-1]=V #periodic boundary conditions
	return H

def makeHamiltonian3D(L,V,W):
	#Generates the Hamiltonian (incl. site disorder) in 3D
	#Uses a clever trick to form the matrix
	H=np.zeros((L*L,L*L))
	np.fill_diagonal(H,W*(np.random.rand(1,L*L)-0.5)) #Random site energies
	v=np.zeros(L)
	v[1]=1
	v[L-1]=1
	C=circulant(v)
	
	H=H+V*(np.kron(np.eye(L),C)+np.kron(C,np.eye(L)))
	return H

def makeTM(H,E):
	#Makes transfer matrix from Hamiltonian. No hopping disorder or inter-slice Hamiltonian
	Hlen=H.shape[0]
	T=np.block([[E*np.eye(Hlen)-H,np.eye(Hlen)],[-1*np.eye(Hlen),np.zeros(H.shape)]])
	return T.astype(np.float64)


def Create_Transfer_Matrix(coupling_matrix_down, W, t_low, c, L, E, dim):
	# P(t)= c*delta(t-t_h) + (1-c)*delta(t-t_l)
	# fraction=c in Z group
	# Dont use dim==2!
	# W does nothing
	N = L * L
	if dim == 3:
		# generate a diagonal of 1s
		W_strip = np.ones(N)

		coupling_matrix_up = np.diag(np.asarray(W_strip))
		coupling_up_inv = np.linalg.inv(coupling_matrix_up)

		# generate the intra-strip hamiltonian
		minilist = np.zeros(L)
		minilist[1] = 1
		minilist[-1] = 1
		offdi = circulant(minilist)
		I = np.eye(L)
		inner_strip_matrix = np.kron(np.asarray(offdi), I) + np.kron(I, np.asarray(offdi))  # magic!
		inner_strip_matrix = np.triu(inner_strip_matrix)  # so the energies are symmetric
		# Find the ones
		ones_indices = np.nonzero(inner_strip_matrix)
		ones_indices = np.array(ones_indices)
		# Choose the random indices
		ones_range = len(ones_indices[0])
		for ind in range(ones_range):
			if np.random.rand() > c:
				inner_strip_matrix[ones_indices[0, ind], ones_indices[1, ind]] = t_low

		# Transpose it over
		inner_strip_matrix = inner_strip_matrix + np.transpose(inner_strip_matrix)

		# Now add the diagonal disorder
		inner_strip_matrix = inner_strip_matrix + np.diag(W_strip)

		upper_left = np.matmul(coupling_up_inv, np.eye(N)*E-inner_strip_matrix)
		upper_right = -np.matmul(coupling_up_inv, coupling_matrix_down)
		lower_left = np.eye(N)
		lower_right = np.zeros((N, N))

		transfer_matrix = np.vstack([np.hstack([upper_left, upper_right]), np.hstack([lower_left, lower_right])])

	return [transfer_matrix, coupling_matrix_up]


def doCalc(eps,min_Lz,L,W,t_low,c,E,dim):
	#eps: desired error
	#min_Lz: At least this many slices will be computed, no matter what
	#L, V, W: size, potential, diagonal disorder
	
	#P(t)= c*delta(t-t_hi) + (1-c)*delta(t-t_low)
	#t_hi is fixed at 1
	#c: driving parameter. should be 0<c<1
	#t_low: sets low off diagonal disorder value. Needs to be low enough to be in conducting regime
	
	
	
	#housekeeping
	np.set_printoptions(linewidth=200)
	if dim==2:
		N=L
	else:
		N=L*L
	
	
	#Performs the actual localization length and conductance calculations
	eps_N=100000000000
	Lz=0 #running length
	n_i=5 #number of steps between orthogonalizations. Only an initial value, will change dynamically
	n_i_min=5 #Set n_i_min=n_i if you want to force n_i
	Nr=1000 #number of T matrices to generate Q0 with
	
	

	max_sum_deviation = 10**(-9) #we check this condition every n_i steps
	
	w=0
	coupling_down=[]
	for i in range(N):
		if np.random.random()<c:
			random=w*np.random.random()-.5*w
			coupling_down.append(random+1)
		else:
			small_w=w/10
			random_num=np.random.random()*small_w-small_w/2
			coupling_down.append(t_low+random_num)

	coupling_matrix_down = np.diag(coupling_down)
	#Generate Q0
	Q0=np.random.rand(2*N,N)-0.5
	Q0 = Q0.astype(np.float64)
	
	Q0, r = np.linalg.qr(Q0)
	
	for i in range(Nr):
		T, coupling_matrix_down =Create_Transfer_Matrix(coupling_matrix_down,W,t_low,c,L,E,dim)
		
		Q0=np.matmul(T,Q0)
		if i%n_i==0:
			Q0, r = np.linalg.qr(Q0)
	
	Q0, r = np.linalg.qr(Q0)
	
	d_a=np.zeros(N,dtype=np.float64)
	e_a=np.zeros(N,dtype=np.float64)
	
	lya=list()
	glst=list()
	Umat=Q0
	
	cnt=0
	while eps_N>eps or Lz<min_Lz:
		Umatbackup = Umat #In case error gets too big
		Lzbackup = Lz
		coupling_matrix_down_backup = coupling_matrix_down
		
		M_ni, coupling_matrix_down = Create_Transfer_Matrix(coupling_matrix_down,W,t_low,c,L,E,dim)
		
		for i in range(n_i):
			T, coupling_matrix_down = Create_Transfer_Matrix(coupling_matrix_down,W,t_low,c,L,E,dim)
			M_ni=np.matmul(M_ni,T)
		Umat=np.matmul(M_ni,Umat)
		
		Umat, r = np.linalg.qr(Umat)
		
		
		w_a_norm = np.abs(np.diagonal(r))
		d_a=d_a+np.log(w_a_norm)
		e_a=e_a+np.square(np.log(w_a_norm))
		
		#D_i.append(1/n_i*np.log(w_a_norm))
		
		
		Lz=Lz+n_i
		xi_a=d_a/Lz #these are the lyapunov exponents
		nu_a=e_a/Lz
		eps_a=np.sqrt(nu_a-xi_a**2)
		
		sum_xi=np.sum(xi_a)
		

		lya.append(xi_a[N-1])
		glst.append(np.log(np.sum(1/np.cosh(L*xi_a)**2)))
		#eps_N = eps_a[N-1]
		'''
		if sum_xi>max_sum_deviation and c==1:
			#revert and start the iteration over
			d_a=d_a-np.log(w_a_norm)
			e_a=e_a-np.square(np.log(w_a_norm))
			Umat=Umatbackup
			Lz=Lzbackup
			coupling_matrix_down=coupling_matrix_down_backup
			del lya[-1]
			del glst[-1]
			
			if n_i>n_i_min:
				n_i=n_i-1 #reduce number of steps between orthogonalizations
				print('Reducing n_i to '+str(n_i)+', sum of LEs got too big: '+str(sum_xi))
		'''
		cnt=cnt+1
		
		if cnt%1000==0:
			'''
			print('xi_N='+str(xi_a[N-1]))
			print('Lz='+str(Lz))
			print('LEs sum='+str(sum_xi))
			print('Lya mean='+str(np.mean(lya)))
			print('Lya std error='+str(stats.sem(lya)))
			print('eps_N='+str(eps_a[N-1]))
			print('g_avg='+str(np.mean(glst)))
			print('#################')
			'''
			
		if len(lya)<=1:
			eps_N = 1.0 #avoid problems that occur if n_i is too big on the first go-through
		else:
			eps_N = stats.sem(lya)/np.mean(lya) #standard error
		
	smallestLya = np.mean(lya)
	
	g=np.exp(np.mean(glst))
	
	

	return np.array([float(smallestLya), np.std(lya), g],dtype=object)



### Calculate localization length

eps=float(sys.argv[1])
min_Lz=float(sys.argv[2])
L=int(sys.argv[3])
W=float(sys.argv[4])
t_low=float(sys.argv[5])
c=float(sys.argv[6])
E=float(sys.argv[7])
dim=int(sys.argv[8])
avg=int(sys.argv[9])
name=sys.argv[10]


params=(eps,min_Lz,L,W,t_low,c,E,dim)
B = np.array([doCalc(*params) for x in range(avg)],dtype=object) #do the calculation and the averaging
ret=np.array([np.mean(B[:,0]),np.sqrt(np.sum(B[:,1]**2)),np.mean(B[:,2])],dtype=object) #avg lambda, avg g
save(params,ret,avg,name)
