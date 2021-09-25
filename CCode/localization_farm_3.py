#ultimately multiplies random matricies and gets their eigenvalue(Lyapunov exponent)

import numpy as np
import concurrent.futures
import time
import sys
import datetime
import matplotlib.pyplot as plt
from scipy.linalg import circulant
from scipy.optimize import curve_fit
from scipy import stats
from numpy import linalg as la

def save(par, return_value, name):
	now = datetime.datetime.now()
	f = open(str(name),"a")
	line = str(now)+" "+str(par)+" "+str(return_value)+"\n"
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

def Create_Transfer_Matrix(coupling_matrix_down,W,t_low,fraction,size,E,dim):
	
	'''
	Inputs:
		
	coupling_matrix_down: The coupling between the nth and n-1th bars/strips.
	W: Diagonal disorder
	size: the width of the strip or bar. Currently only square bars are supported
	fraction: the fraction of links that are "good", AKA necking fraction
	E: The fermi level, 0 represents the center of the band
	
	Returns:
	transfer_matrix: The transfer matrix between the n and n+1th layers
	coupling_matrix_up: The coupling matrix between the n and n+1th layers. Will be used in the next iteration
	as the coupling down matrix for self consistency
	'''
	#P(t)= c*delta(t-t_h) + (1-c)*delta(t-t_l)
	c=1 #c=t_hi=1
	#fraction=c in Z group
	w=0 #width of off diagonal disorder distributions. In binary model, this is 0
	if dim==2:
		
		#coupling_up is used to form the matrix that couples the nth strip or bar to the n+1th strip or bar
		coupling_up=[]
		for i in range(size):
			#fraction determines the connectivity (fraction of good links) on the lattice
			if np.random.random()<fraction:
				random_num=np.random.random()*w-w/2
				coupling_up.append(c+random_num)
			else:
				small_w=w/10
				random_num=np.random.random()*small_w-small_w/2
				coupling_up.append(t_low+random_num)
				
		coupling_matrix_up = np.diag(coupling_up)
		
		coupling_up_inv=la.inv(coupling_matrix_up)
		inner_strip_matrix=np.zeros((size,size))
		
		#generate the intra-strip hamiltonian
		for i in range(size):
			for j in range(size):
				#First the on-site, diagonal pieces
				if i==j:
					inner_strip_matrix[i][j]=E-(np.random.random()*W-W/2)
				#inner_strip_matrix[i][j]=E-(np.random.normal(0,np.sqrt(W/2)))
				#Next, the offdiagonal pieces
				if (i==(j+1) and (i)%size!=0) or (i==(j-1) and (i+1)%size!=0):
					if np.random.random()<fraction:
						if inner_strip_matrix[i][j]==0 and size!=2:
							random_num=np.random.random()*w-w/2
							inner_strip_matrix[i][j]=-(c+random_num)
							inner_strip_matrix[j][i]=-(c+random_num)
						if inner_strip_matrix[i][j]==0 and size==2:
							random_num=np.random.random()*w-w/2
							inner_strip_matrix[i][j]=-(2*(c+random_num))
							inner_strip_matrix[j][i]=-(2*(c+random_num))
					else:
						if inner_strip_matrix[i][j]==0:
							small_w=w/10
							random_num=np.random.random()*small_w-small_w/2
							inner_strip_matrix[i][j]=-t_low+random_num
							inner_strip_matrix[j][i]=-t_low+random_num
				#This last one ensures periodic boundary conditions
				if i==(j+size-1) or i==(j-size+1):
					if np.random.random()<fraction:
						if inner_strip_matrix[i][j]==0:
							random_num=np.random.random()*w-w/2
							inner_strip_matrix[i][j]=-(c+random_num)
							inner_strip_matrix[j][i]=-(c+random_num)
	
					else:
						if inner_strip_matrix[i][j]==0:
							small_w=w/10
							random_num=np.random.random()*small_w-small_w/2
							inner_strip_matrix[i][j]=-t_low+random_num
							inner_strip_matrix[j][i]=-t_low+random_num
							
				
		#These four pieces combine into the transfer matrix
		upper_left=np.matmul(coupling_up_inv,inner_strip_matrix)
		upper_right=-np.matmul(coupling_up_inv,coupling_matrix_down)
		lower_left=np.identity(size)
		lower_right=np.zeros((size,size))
		
		transfer_matrix=np.block([[upper_left,upper_right],[lower_left,lower_right]])
		
	if dim==3:
		#To do the same thing in 3D, we just make a bar as the intra-strip hamiltonian rather than a strip
		coupling_up=[]
		for i in range(size**2):
			if np.random.random()<fraction:
				random_num=np.random.normal(0,w)
				coupling_up.append(c+random_num)
			else:
				small_w=w/10
				random_num=np.random.random()*small_w-small_w/2
				coupling_up.append(t_low+random_num)
		
		coupling_matrix_up = np.diag(coupling_up)
		coupling_up_inv=la.inv(coupling_matrix_up)
		inner_strip_matrix=np.zeros((size**2,size**2))
		
		#generate the intra-strip hamiltonian
		for i in range(size**2):
			for j in range(size**2):
				#First the on-site, diagonal pieces
				if i==j:
					inner_strip_matrix[i][j]=E-(np.random.random()*W-W/2)
	#				 inner_strip_matrix[i][j]=E-(np.random.normal(0,np.sqrt(W/2)))
				#Next, the offdiagonal pieces
				if (i==(j+1) and (i)%size!=0) or (i==(j-1) and (i+1)%size!=0):
					if np.random.random()<fraction:
						if inner_strip_matrix[i][j]==0 and size!=2:
							random_num=np.random.random()*w-w/2
							inner_strip_matrix[i][j]=-(c+random_num)
							inner_strip_matrix[j][i]=-(c+random_num)
						if inner_strip_matrix[i][j]==0 and size==2:
							random_num=np.random.random()*w-w/2
							inner_strip_matrix[i][j]=-(2*(c+random_num))
							inner_strip_matrix[j][i]=-(2*(c+random_num))
					else:
						if inner_strip_matrix[i][j]==0:
							small_w=w/10
							random_num=np.random.random()*small_w-small_w/2
							inner_strip_matrix[i][j]=-t_low + random_num
							inner_strip_matrix[j][i]=-t_low + random_num
	#			 This last one ensures periodic boundary conditions
				if (i+size-1)==j and i%size==0:
					if np.random.random()<fraction:
						if inner_strip_matrix[i][j]==0:
							random_num=np.random.random()*w-w/2
							inner_strip_matrix[i][j]=-(c+random_num)
							inner_strip_matrix[j][i]=-(c+random_num)
	
					else:
						if inner_strip_matrix[i][j]==0:
							small_w=w/10
							random_num=np.random.random()*small_w-small_w/2
							inner_strip_matrix[i][j]=-t_low+random_num
							inner_strip_matrix[j][i]=-t_low+random_num
							
				if i==(j+size) or i==(j-size):
					if np.random.random()<fraction:
						if inner_strip_matrix[i][j]==0:
							random_num=np.random.random()*w-w/2
							inner_strip_matrix[i][j]=-(c+random_num)
							inner_strip_matrix[j][i]=-(c+random_num)
	
					else:
						if inner_strip_matrix[i][j]==0:
							small_w=w/10
							random_num=np.random.random()*small_w-small_w/2
							inner_strip_matrix[i][j]=-t_low + random_num
							inner_strip_matrix[j][i]=-t_low + random_num
							
				if i==(j+size**2-size) or i==(j-(size**2-size)):
					if np.random.random()<fraction:
						if inner_strip_matrix[i][j]==0:
							random_num=np.random.random()*w-w/2
							inner_strip_matrix[i][j]=-(c+random_num)
							inner_strip_matrix[j][i]=-(c+random_num)
	
					else:
						if inner_strip_matrix[i][j]==0:
							small_w=w/10
							random_num=np.random.random()*small_w-small_w/2
							inner_strip_matrix[i][j]=-t_low + random_num
							inner_strip_matrix[j][i]=-t_low+random_num
							
							

		upper_left=np.matmul(coupling_up_inv,inner_strip_matrix)
		upper_right=-np.matmul(coupling_up_inv,coupling_matrix_down)
		lower_left=np.identity(size**2)
		lower_right=np.zeros((size**2,size**2))

		
		transfer_matrix=np.block([[upper_left,upper_right],[lower_left,lower_right]])
	
	
	return [transfer_matrix,coupling_matrix_up]

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
	Nr=100 #number of T matrices to generate Q0 with
	
	

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
	Q0=np.random.rand(2*N,2*N)-0.5
	Q0 = Q0.astype(np.float64)
	
	Q0, r = np.linalg.qr(Q0)
	
	for i in range(Nr):
		T, coupling_matrix_down =Create_Transfer_Matrix(coupling_matrix_down,W,t_low,c,L,E,dim)
		
		Q0=np.matmul(T,Q0)
		if i%n_i==0:
			Q0, r = np.linalg.qr(Q0)
	
	Q0, r = np.linalg.qr(Q0)
	
	d_a=np.zeros(2*N,dtype=np.float64)
	e_a=np.zeros(2*N,dtype=np.float64)
	
	lya=list()
	glst=list()
	Umat=Q0
	
	cnt=0
	while eps_N>eps or Lz<min_Lz:
		#@TODO ADAM this is the meat of the code
		Umatbackup = Umat #In case error gets too big
		Lzbackup = Lz
		coupling_matrix_down_backup = coupling_matrix_down
		
		M_ni=Umat
		
		#@TODO this is notes sept 14
		for i in range(n_i):
			T, coupling_matrix_down = Create_Transfer_Matrix(coupling_matrix_down,W,t_low,c,L,E,dim)
			M_ni=np.matmul(M_ni,T)
		
		Umat=np.matmul(M_ni,Umat)
		
		Umat, r = np.linalg.qr(Umat)
		
		
		w_a_norm = np.abs(np.diagonal(r))
		d_a=d_a+np.log(w_a_norm)
		e_a=e_a+np.square(np.log(w_a_norm))
		
		#D_i.append(1/n_i*np.log(w_a_norm))

		#@TODO this is the meat
		
		Lz=Lz+n_i
		xi_a=d_a/Lz #these are the lyapunov exponents
		nu_a=e_a/Lz
		eps_a=np.sqrt(nu_a-xi_a**2)
		
		sum_xi=np.sum(xi_a)
		

		lya.append(xi_a[N-1])
		glst.append(np.log(np.sum(2/(np.cosh(xi_a*n_i)**2))))
		#eps_N = eps_a[N-1]
		
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
			eps_N = stats.sem(lya) #standard error
		
	#smallestLya = xi_a[N-1]
	smallestLya = np.mean(lya)
	
	g=np.exp(np.mean(glst))
	
	
	#A=np.matmul(pmat,A)
	A=np.eye(2*N)
	return np.array([float(smallestLya), g],dtype=object)

def threadHelper(pack):
	#Arguments have to be passed in a list for now. Please improve
	eps=pack[0]
	min_Lz=pack[1]
	L=pack[2]
	V=pack[3]
	W=pack[4]
	t_low=pack[5]
	c=pack[6]
	E=pack[7]
	dim=pack[8]
	av=pack[9]
	#av=number of trials to average over for conductance and lambda calculations
	B = np.array([doCalc(eps,min_Lz,L,V,W,t_low,c,E,dim) for x in range(av)],dtype=object)
	#print('L='+str(L)+' W='+str(W)+' E='+str(E)+' c='+str(c)+' t_low='+str(t_low)+' is done.')
	
	return np.array([np.mean(B[:,0]),np.mean(B[:,1]),B[0][2],np.std(B[:,0])],dtype=object)



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
ret=np.array([np.mean(B[:,0]),np.mean(B[:,1])],dtype=object) #avg lambda, avg g
save(params,ret,name)