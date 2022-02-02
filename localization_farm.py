from glob import iglob
from stat import IO_REPARSE_TAG_APPEXECLINK
import numpy as np
#import concurrent.futures
import time
import sys
import datetime
import matplotlib.pyplot as plt
from scipy.linalg import circulant
from scipy.optimize import curve_fit
from scipy import stats
from numpy import linalg as la


global distList, ranList, id, ir, ld, lr


def save(par, return_value, avg, name):
	now = datetime.datetime.now()
	f = open(str(name),"a")
	line = str(now)+" "+str(par)+" "+str(return_value)+" "+str(avg)+"\n"
	f.write(line)

def threadHelper(pack):
	#Arguments have to be passed in a list for now. Please improve
	eps=pack[0]
	min_Lz=pack[1]
	L=pack[2]
	V=pack[3]
	W=pack[4]
	t_low=pack[5]
	c=pack[6]
	#@TODO sigma
	E=pack[7]
	dim=pack[8]
	av=pack[9]
	#av=number of trials to average over for conductance and lambda calculations
	B = np.array([doCalc(eps,min_Lz,L,V,W,t_low,c,E,dim) for x in range(av)],dtype=object)
	#print('L='+str(L)+' W='+str(W)+' E='+str(E)+' c='+str(c)+' t_low='+str(t_low)+' is done.')
	
	return np.array([np.mean(B[:,0]),np.mean(B[:,1]),B[0][2],np.std(B[:,0])],dtype=object)

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

def genQ0(coupling_matrix_down, W, tDist, c, L, E,dim,Nr,n_i):
	N=L*L
	#Generate Q0
	Q0=np.random.rand(2*N,N)-0.5
	Q0 = Q0.astype(np.float64)
	
	Q0, r = np.linalg.qr(Q0)
	
	for i in range(Nr):
		T, coupling_matrix_down =Create_Transfer_Matrix_Old(coupling_matrix_down,W,tDist,c,L,E,dim)
		
		Q0=np.matmul(T,Q0)
		if i%n_i==0:
			Q0, r = np.linalg.qr(Q0)
	
	Q0, r = np.linalg.qr(Q0)
	return Q0, coupling_matrix_down

def makeTM(H,E):
	#Makes transfer matrix from Hamiltonian. No hopping disorder or inter-slice Hamiltonian
	Hlen=H.shape[0]
	T=np.block([[E*np.eye(Hlen)-H,np.eye(Hlen)],[-1*np.eye(Hlen),np.zeros(H.shape)]])
	return T.astype(np.float64)

def newTRan(tDist,gaussian=True,testing=False):
	if testing:
		tL=.5
		tH=.5
		return [tL,tH]

	if gaussian:
		tL=np.random.normal(tDist[0][0],tDist[0][1])
		tH=np.random.normal(tDist[1][0],tDist[1][1])
	else:
		if tDist[0][0]==tDist[0][1]:
			tL=tDist[0][0]
		else:
			tL=np.random.uniform(tDist[0][0],tDist[0][1])

		if tDist[1][0]==tDist[1][1]:
			tH=tDist[1][0]
		else:
			tH=np.random.uniform(tDist[1][0],tDist[1][1])

	return [tL,tH]
#newTRan.ranCounter=0

def Create_Hamiltonian(W, tDist, c, L):
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
		tL,tH=newTRan(tDist)
		if np.random.rand() > c:
			inner_strip_matrix[ones_indices[0, ind], ones_indices[1, ind]] = tL
		else:
			inner_strip_matrix[ones_indices[0, ind], ones_indices[1, ind]] = tH

	# Transpose it over
	inner_strip_matrix = inner_strip_matrix + np.transpose(inner_strip_matrix)
	
	# Flip the signs
	inner_strip_matrix = -1 * inner_strip_matrix



	return inner_strip_matrix


def Create_Transfer_Matrix(coupling_matrix_down, W, tDist, c, L, E, dim):
	# P(t)= c*delta(t-t_h) + (1-c)*delta(t-t_l)
	# fraction=c in Z group
	# Dont use dim==2!
	# W does nothing
	N = L * L
	if dim == 3:
		# generate a diagonal of 1s
		W_strip = W*(np.random.rand(N)-0.5)
		#make <\eps_i>=0 exactly
		#W_strip = W_strip - np.mean(W_strip)

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
			tL,tH=newTRan(tDist)
			if np.random.rand() > c:
				inner_strip_matrix[ones_indices[0, ind], ones_indices[1, ind]] = tL
			else:
				inner_strip_matrix[ones_indices[0, ind], ones_indices[1, ind]] = tH

		# Transpose it over
		inner_strip_matrix = inner_strip_matrix + np.transpose(inner_strip_matrix)
		
		# Flip the signs
		inner_strip_matrix = -1 * inner_strip_matrix

		# Now add the diagonal disorder
		inner_strip_matrix = inner_strip_matrix - np.diag(W_strip)

		upper_left = np.matmul(coupling_up_inv, np.eye(N)*E-inner_strip_matrix)

		upper_right = -np.matmul(coupling_up_inv, coupling_matrix_down)
		lower_left = np.eye(N)
		lower_right = np.zeros((N, N))

		transfer_matrix = np.vstack([np.hstack([upper_left, upper_right]), np.hstack([lower_left, lower_right])])

	return [transfer_matrix, coupling_matrix_up]


def Create_Hamiltonian_Old(W, tDist, fraction, size):

	inner_strip_matrix=np.zeros((size**2,size**2))


	#generate the intra-strip hamiltonian
	for i in range(size**2):
		for j in range(size**2):
			#First the on-site, diagonal pieces
			#if i==j:
				#inner_strip_matrix[i][j]=E-(np.random.random()*W-W/2)
		#		inner_strip_matrix[i][j]=E-(np.random.normal(0,np.sqrt(W/2)))
			#Next, the offdiagonal pieces
			if (i==(j+1) and (i)%size!=0) or (i==(j-1) and (i+1)%size!=0):
				#TODO maybe new ran instead of below

				#checking if we have a tL link or tH link
				if np.random.random()<fraction:

					#@TODO NEW RAN
					tL,tH=newTRan(tDist)
					if inner_strip_matrix[i][j]==0 and size!=2:
						random_num=0#np.random.random()*w-w/2

						#Where we have [i][j] we are fixing hopping from i->j same as j->i
						inner_strip_matrix[i][j]=-(tH+random_num)
						inner_strip_matrix[j][i]=-(tH+random_num)

					if inner_strip_matrix[i][j]==0 and size==2:

						random_num=0#np.random.random()*w-w/2
						inner_strip_matrix[i][j]=-(2*(tH+random_num))
						inner_strip_matrix[j][i]=-(2*(tH+random_num))
				else:
					#@TODO NEW RAN
					tL,tH=newTRan(tDist)
					if inner_strip_matrix[i][j]==0:

						small_w=w/10
						random_num=0#np.random.random()*small_w-small_w/2
						inner_strip_matrix[i][j]=-tL + random_num
						inner_strip_matrix[j][i]=-tL + random_num
		#			 This last one ensures periodic boundary conditions
			if (i+size-1)==j and i%size==0:

				if np.random.random()<fraction:
					
					if inner_strip_matrix[i][j]==0:
						#@TODO NEW RAN
						tL,tH=newTRan(tDist)
						random_num=0#np.random.random()*w-w/2
						inner_strip_matrix[i][j]=-(tH+random_num)
						inner_strip_matrix[j][i]=-(tH+random_num)

				else:
					
					if inner_strip_matrix[i][j]==0:
						#@TODO NEW RAN
						tL,tH=newTRan(tDist)
						small_w=w/10
						random_num=0#np.random.random()*small_w-small_w/2
						inner_strip_matrix[i][j]=-tL+random_num
						inner_strip_matrix[j][i]=-tL+random_num
						
			if i==(j+size) or i==(j-size):
				if np.random.random()<fraction:

					if inner_strip_matrix[i][j]==0:
						#@TODO NEW RAN
						tL,tH=newTRan(tDist)
						random_num=0#np.random.random()*w-w/2
						inner_strip_matrix[i][j]=-(tH+random_num)
						inner_strip_matrix[j][i]=-(tH+random_num)

				else:
					if inner_strip_matrix[i][j]==0:
						#@TODO NEW RAN
						tL,tH=newTRan(tDist)
						small_w=w/10
						random_num=0#np.random.random()*small_w-small_w/2
						inner_strip_matrix[i][j]=-tL + random_num
						inner_strip_matrix[j][i]=-tL + random_num
						
			if i==(j+size**2-size) or i==(j-(size**2-size)):
				if np.random.random()<fraction:
					if inner_strip_matrix[i][j]==0:
						#@TODO NEW RAN
						tL,tH=newTRan(tDist)
						random_num=0#np.random.random()*w-w/2
						inner_strip_matrix[i][j]=-(tH+random_num)
						inner_strip_matrix[j][i]=-(tH+random_num)

				else:
					if inner_strip_matrix[i][j]==0:
						#@TODO NEW RAN
						tL,tH=newTRan(tDist)
						small_w=w/10
						random_num=0#np.random.random()*small_w-small_w/2
						inner_strip_matrix[i][j]=-tL + random_num
						inner_strip_matrix[j][i]=-tL+random_num
	return inner_strip_matrix

						
def Create_Transfer_Matrix_Old(coupling_matrix_down,W,tDist,fraction,size,E,dim):
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

	#fraction=c in Z group
	w=0 #width of off diagonal disorder distributions. In binary model, this is 0
		
	if dim==3:
		#To do the same thing in 3D, we just make a bar as the intra-strip hamiltonian rather than a strip
		coupling_up=[]
		for i in range(size**2):
			#@TODO NEW RAN
			tL,tH=newTRan(tDist)
			if np.random.random()<fraction:
				random_num=0#np.random.normal(0,w)
				coupling_up.append(tH+random_num)
			else:
				small_w=w/10
				random_num=0#np.random.random()*small_w-small_w/2
				coupling_up.append(tL+random_num)
		
		coupling_matrix_up = np.diag(coupling_up)
		coupling_up_inv=la.inv(coupling_matrix_up)

		# for i in range(size**2):
		# 	for j in range(size**2):
		# 		#First the on-site, diagonal pieces
		# 		if i==j:
		# 			inner_strip_matrix[i][j]=E-(np.random.random()*W-W/2)	

		#generate the intra-strip hamiltonian
		inner_strip_matrix = Create_Hamiltonian_Old(W, tDist, fraction, size)

		for i in range(size**2):
			for j in range(size**2):
				#First the on-site, diagonal pieces
				if i==j:
					inner_strip_matrix[i][j]=E-(np.random.random()*W-W/2)

		upper_left=np.matmul(coupling_up_inv,inner_strip_matrix)
		upper_right=-np.matmul(coupling_up_inv,coupling_matrix_down)
		lower_left=np.identity(size**2)
		lower_right=np.zeros((size**2,size**2))

		
		transfer_matrix=np.block([[upper_left,upper_right],[lower_left,lower_right]])
	
	
	return [transfer_matrix,coupling_matrix_up]


def Create_Transfer_Matrix_Chase(coupling_matrix_down, W, t_low, fraction, size, E, dim):
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
	# P(t)= c*delta(t-t_h) + (1-c)*delta(t-t_l)
	c = 1  # c=t_hi=1
	# fraction=c in Z group
	w = 0  # width of off diagonal disorder distributions. In binary model, this is 0
	if dim == 2:

		# coupling_up is used to form the matrix that couples the nth strip or bar to the n+1th strip or bar
		coupling_up = []
		for i in range(size):
			# fraction determines the connectivity (fraction of good links) on the lattice
			if np.random.random() < fraction:
				random_num = np.random.random() * w - w / 2
				coupling_up.append(c + random_num)
			else:
				small_w = w / 10
				random_num = np.random.random() * small_w - small_w / 2
				coupling_up.append(t_low + random_num)

		coupling_matrix_up = np.diag(coupling_up)

		coupling_up_inv = la.inv(coupling_matrix_up)
		inner_strip_matrix = np.zeros((size, size))

		# generate the intra-strip hamiltonian
		for i in range(size):
			for j in range(size):
				# First the on-site, diagonal pieces
				if i == j:
					inner_strip_matrix[i][j] = E - (np.random.random() * W - W / 2)
				# inner_strip_matrix[i][j]=E-(np.random.normal(0,np.sqrt(W/2)))
				# Next, the offdiagonal pieces
				if (i == (j + 1) and (i) % size != 0) or (i == (j - 1) and (i + 1) % size != 0):
					if np.random.random() < fraction:
						if inner_strip_matrix[i][j] == 0 and size != 2:
							random_num = np.random.random() * w - w / 2
							inner_strip_matrix[i][j] = -(c + random_num)
							inner_strip_matrix[j][i] = -(c + random_num)
						if inner_strip_matrix[i][j] == 0 and size == 2:
							random_num = np.random.random() * w - w / 2
							inner_strip_matrix[i][j] = -(2 * (c + random_num))
							inner_strip_matrix[j][i] = -(2 * (c + random_num))
					else:
						if inner_strip_matrix[i][j] == 0:
							small_w = w / 10
							random_num = np.random.random() * small_w - small_w / 2
							inner_strip_matrix[i][j] = -t_low + random_num
							inner_strip_matrix[j][i] = -t_low + random_num
				# This last one ensures periodic boundary conditions
				if i == (j + size - 1) or i == (j - size + 1):
					if np.random.random() < fraction:
						if inner_strip_matrix[i][j] == 0:
							random_num = np.random.random() * w - w / 2
							inner_strip_matrix[i][j] = -(c + random_num)
							inner_strip_matrix[j][i] = -(c + random_num)

					else:
						if inner_strip_matrix[i][j] == 0:
							small_w = w / 10
							random_num = np.random.random() * small_w - small_w / 2
							inner_strip_matrix[i][j] = -t_low + random_num
							inner_strip_matrix[j][i] = -t_low + random_num

		# These four pieces combine into the transfer matrix
		upper_left = np.matmul(coupling_up_inv, inner_strip_matrix)
		upper_right = -np.matmul(coupling_up_inv, coupling_matrix_down)
		lower_left = np.identity(size)
		lower_right = np.zeros((size, size))

		transfer_matrix = np.block([[upper_left, upper_right], [lower_left, lower_right]])

	if dim == 3:
		# To do the same thing in 3D, we just make a bar as the intra-strip hamiltonian rather than a strip
		coupling_up = []
		for i in range(size ** 2):
			if np.random.random() < fraction:
				random_num = np.random.normal(0, w)
				coupling_up.append(c + random_num)
			else:
				small_w = w / 10
				random_num = np.random.random() * small_w - small_w / 2
				coupling_up.append(t_low + random_num)

		coupling_matrix_up = np.diag(coupling_up)
		coupling_up_inv = la.inv(coupling_matrix_up)
		inner_strip_matrix = np.zeros((size ** 2, size ** 2))

		# generate the intra-strip hamiltonian
		for i in range(size ** 2):
			for j in range(size ** 2):
				# First the on-site, diagonal pieces
				if i == j:
					inner_strip_matrix[i][j] = E - (np.random.random() * W - W / 2)
				#				 inner_strip_matrix[i][j]=E-(np.random.normal(0,np.sqrt(W/2)))
				# Next, the offdiagonal pieces
				if (i == (j + 1) and (i) % size != 0) or (i == (j - 1) and (i + 1) % size != 0):
					if np.random.random() < fraction:
						if inner_strip_matrix[i][j] == 0 and size != 2:
							random_num = np.random.random() * w - w / 2
							inner_strip_matrix[i][j] = -(c + random_num)
							inner_strip_matrix[j][i] = -(c + random_num)
						if inner_strip_matrix[i][j] == 0 and size == 2:
							random_num = np.random.random() * w - w / 2
							inner_strip_matrix[i][j] = -(2 * (c + random_num))
							inner_strip_matrix[j][i] = -(2 * (c + random_num))
					else:
						if inner_strip_matrix[i][j] == 0:
							small_w = w / 10
							random_num = np.random.random() * small_w - small_w / 2
							inner_strip_matrix[i][j] = -t_low + random_num
							inner_strip_matrix[j][i] = -t_low + random_num
				#			 This last one ensures periodic boundary conditions
				if (i + size - 1) == j and i % size == 0:
					if np.random.random() < fraction:
						if inner_strip_matrix[i][j] == 0:
							random_num = np.random.random() * w - w / 2
							inner_strip_matrix[i][j] = -(c + random_num)
							inner_strip_matrix[j][i] = -(c + random_num)

					else:
						if inner_strip_matrix[i][j] == 0:
							small_w = w / 10
							random_num = np.random.random() * small_w - small_w / 2
							inner_strip_matrix[i][j] = -t_low + random_num
							inner_strip_matrix[j][i] = -t_low + random_num

				if i == (j + size) or i == (j - size):
					if np.random.random() < fraction:
						if inner_strip_matrix[i][j] == 0:
							random_num = np.random.random() * w - w / 2
							inner_strip_matrix[i][j] = -(c + random_num)
							inner_strip_matrix[j][i] = -(c + random_num)

					else:
						if inner_strip_matrix[i][j] == 0:
							small_w = w / 10
							random_num = np.random.random() * small_w - small_w / 2
							inner_strip_matrix[i][j] = -t_low + random_num
							inner_strip_matrix[j][i] = -t_low + random_num

				if i == (j + size ** 2 - size) or i == (j - (size ** 2 - size)):
					if np.random.random() < fraction:
						if inner_strip_matrix[i][j] == 0:
							random_num = np.random.random() * w - w / 2
							inner_strip_matrix[i][j] = -(c + random_num)
							inner_strip_matrix[j][i] = -(c + random_num)

					else:
						if inner_strip_matrix[i][j] == 0:
							small_w = w / 10
							random_num = np.random.random() * small_w - small_w / 2
							inner_strip_matrix[i][j] = -t_low + random_num
							inner_strip_matrix[j][i] = -t_low + random_num

		upper_left = np.matmul(coupling_up_inv, inner_strip_matrix)
		upper_right = -np.matmul(coupling_up_inv, coupling_matrix_down)
		lower_left = np.identity(size ** 2)
		lower_right = np.zeros((size ** 2, size ** 2))

		transfer_matrix = np.block([[upper_left, upper_right], [lower_left, lower_right]])

	return [transfer_matrix, coupling_matrix_up]


def doCalc(eps,min_Lz,L,W,tDist,c,E,dim):
	#@TODO sigma
	
	#eps: desired error
	#min_Lz: At least this many slices will be computed, no matter what
	#L, V, W: size, potential, diagonal disorder
	
	#P(t)= c*delta(t-t_hi) + (1-c)*delta(t-tL)
	#tH is fixed at 1
	#c: driving parameter. should be 0<c<1
	#tL: sets low off diagonal disorder value. Needs to be low enough to be in conducting regime
	
	
	
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
		#@TODO NEW RAN
		tL,tH=newTRan(tDist)
		if np.random.random()<c:
			random=w*np.random.random()-.5*w
			coupling_down.append(random+tH)
		else:
			small_w=w/10
			random_num=np.random.random()*small_w-small_w/2
			coupling_down.append(tL+random_num)

	coupling_matrix_down = np.diag(coupling_down)
	
	# #Generate Q0
	# Q0=np.random.rand(2*N,N)-0.5
	# Q0 = Q0.astype(np.float64)
	
	# Q0, r = np.linalg.qr(Q0)
	
	# for i in range(Nr):
	# 	T, coupling_matrix_down =Create_Transfer_Matrix(coupling_matrix_down,W,tDist,c,L,E,dim)
		
	# 	Q0=np.matmul(T,Q0)
	# 	if i%n_i==0:
	# 		Q0, r = np.linalg.qr(Q0)
	
	# Q0, r = np.linalg.qr(Q0)

	Q0, coupling_matrix_down = genQ0(coupling_matrix_down,W,tDist,c,L,E,dim,Nr,n_i)
	
	d_a=np.zeros(N,dtype=np.float64)
	e_a=np.zeros(N,dtype=np.float64)
	
	lya=list()
	glst=list()
	Umat=Q0
	
	cnt=0
	while eps_N>eps or Lz<min_Lz:
		#@TODO ADAM this is the meat of the code
		
		Umatbackup = Umat #In case error gets too big
		Lzbackup = Lz
		coupling_matrix_down_backup = coupling_matrix_down
		
		M_ni, coupling_matrix_down = Create_Transfer_Matrix(coupling_matrix_down,W,tDist,c,L,E,dim)
		
		#@TODO this is notes sept 14
		for i in range(n_i):
			T, coupling_matrix_down = Create_Transfer_Matrix(coupling_matrix_down,W,tDist,c,L,E,dim)
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
		glst.append(np.log(np.sum(1/np.cosh(L*xi_a)**2)))#1/28/22 prev line glst.append(np.log(np.sum(2/(np.cosh(xi_a*n_i)**2))))
		#eps_N = eps_a[N-1]

		# if sum_xi>max_sum_deviation and c==1:
		# 	#revert and start the iteration over
		# 	d_a=d_a-np.log(w_a_norm)
		# 	e_a=e_a-np.square(np.log(w_a_norm))
		# 	Umat=Umatbackup
		# 	Lz=Lzbackup
		# 	coupling_matrix_down=coupling_matrix_down_backup
		# 	del lya[-1]
		# 	del glst[-1]
			
		# 	if n_i>n_i_min:
		# 		n_i=n_i-1 #reduce number of steps between orthogonalizations
		# 		print('Reducing n_i to '+str(n_i)+', sum of LEs got too big: '+str(sum_xi))
			
		cnt=cnt+1
		
		# if cnt%1000==0:
			# '''
			# print('xi_N='+str(xi_a[N-1]))
			# print('Lz='+str(Lz))
			# print('LEs sum='+str(sum_xi))
			# print('Lya mean='+str(np.mean(lya)))
			# print('Lya std error='+str(stats.sem(lya)))
			# print('eps_N='+str(eps_a[N-1]))
			# print('g_avg='+str(np.mean(glst)))
			# print('#################')
			# '''
			
		if len(lya)<=1:
			eps_N = 1.0 #avoid problems that occur if n_i is too big on the first go-through
		else:
			eps_N = stats.sem(lya)/np.mean(lya) #standard error
		
	#smallestLya = xi_a[N-1]
	smallestLya = np.mean(lya)
	
	g=np.exp(np.mean(glst))
	
	
	#A=np.matmul(pmat,A)
	A=np.eye(2*N)
	return np.array([float(smallestLya), g],dtype=object)

def test_tmatrix(W, tDist, c, L, E, dim):

	#Performs the actual localization length and conductance calculations
	eps_N=100000000000
	Lz=0 #running length
	n_i=5 #number of steps between orthogonalizations. Only an initial value, will change dynamically
	Nr=2 #number of T matrices to generate Q0 with
	
	L=3
	c=1
	N=L*L

	numTest=1
	seed=123456
	test_t=True
	test_h=True

	max_sum_deviation = 10**(-9) #we check this condition every n_i steps
	
	w=0
	coupling_down=[]
	for i in range(N):
		#@TODO NEW RAN
		tL,tH=newTRan(tDist)
		if np.random.random()<c:
			random=w*np.random.random()-.5*w
			coupling_down.append(random+tH)
		else:
			small_w=w/10
			random_num=np.random.random()*small_w-small_w/2
			coupling_down.append(tL+random_num)

	coupling_matrix_down = np.diag(coupling_down)

	Q0, coupling_matrix_down =genQ0(coupling_matrix_down,W,tDist,c,L,E,dim,Nr, n_i)
	

	coupling_matrix_down=np.identity(N)
	CDownOld = coupling_matrix_down
	CDownNew = coupling_matrix_down


	#transfermatrix test
	if test_t:
		np.random.seed(seed)
		for i in range(numTest):
			#print("i: " + str(i))
			Lt=(i+1)*L
			Told, CDownOld = Create_Transfer_Matrix_Old(CDownOld,W,tDist,c,Lt,E,dim)

		np.random.seed(seed)
		for i in range(numTest):
			#print("i: " + str(i))
			Lt=(i+1)*L
			Tnew, CDownNew = Create_Transfer_Matrix(CDownNew,W,tDist,c,Lt,E,dim)

		print('Xfer Matrix')
		tequal = np.array_equal(Tnew,Told)
		if tequal:
			print('Pass')
		else:
			print('Fail')
			print("Old: ")
			print(Told[0])
			print("New: ")
			print(Tnew[0])



	#hamiltonian test
	if test_h:
		for i in range(numTest):
			#print("i: " + str(i))
			Lt=(i+1)*L
			np.random.seed(seed)
			Hnew = Create_Hamiltonian(W,tDist,c,Lt)
			np.random.seed(seed)
			Hold= Create_Hamiltonian_Old(W,tDist,c,Lt)

			print('Hamiltonian L='+str(Lt))
			hequal = np.array_equal(Hnew,Hold)
			if hequal:
				print('Pass')
			else:
				print('Fail')
				print("Old: ")
				print(Hold)
				print("New: ")
				print(Hnew)


def test_harness():
	eps=1
	min_Lz=500000
	
	t_low_bot=0
	t_low_top=.5
	t_high_bot=1
	t_high_top=10
	W=10
	E=0
	dim=3
	avg=1
	filename="test_harness.txt"
	L=4
	c=.5

	tDist=[[t_low_bot,t_low_top],[t_high_bot,t_high_top]]
	params=(eps,min_Lz,L,W,tDist,c,E,dim)
	test_tmatrix(W, tDist, c, L, E, dim)


if len(sys.argv) == 1:
	test_harness()
else:
	### Calculate localization length
	eps=float(sys.argv[1])
	min_Lz=float(sys.argv[2])
	L=int(sys.argv[3])
	W=float(sys.argv[4])
	t_low_bot=float(sys.argv[5])
	t_low_top=float(sys.argv[6])
	t_high_bot=float(sys.argv[7])
	t_high_top=float(sys.argv[8])
	c=float(sys.argv[9])
	E=float(sys.argv[10])
	dim=int(sys.argv[11])
	avg=int(sys.argv[12])
	name=sys.argv[13]

	tDist=[[t_low_bot,t_low_top],[t_high_bot,t_high_top]]



	params=(eps,min_Lz,L,W,tDist,c,E,dim)
	B = np.array([doCalc(*params) for x in range(avg)],dtype=object) #do the calculation and the averaging
	ret=np.array([np.mean(B[:,0]),np.mean(B[:,1])],dtype=object) #avg lambda, avg g
	save(params,ret,avg,name)
	#save(L,' num ran calls:'+str(newTRan.ranCounter)+' - ',1,"ranTCalls.txt")
