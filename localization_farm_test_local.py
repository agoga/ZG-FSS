import numpy as np
import time
import sys
import datetime
import matplotlib.pyplot as plt
from scipy.linalg import circulant
from scipy.optimize import curve_fit
from scipy import stats
from numpy import linalg as la

import multiprocessing as mp
import itertools
import cProfile

def save(par, return_value, avg, name):
	now = datetime.datetime.now()
	f = open(str(name),"a")
	line = str(now)+" "+str(par)+" "+str(return_value)+" "+str(avg)+"\n"
	f.write(line)

def Create_Transfer_Matrix(coupling_matrix_down, W, t_low, c, L, E, dim):
	# P(t)= c*delta(t-t_h) + (1-c)*delta(t-t_l)
	# fraction=c in Z group
	# Dont use dim==2!
	# W does nothing
	N = L * L
	if dim == 3:
		W_strip = W * (np.random.rand(N) - 0.5)
		# make <\eps_i>=0 exactly
		# W_strip = W_strip - np.mean(W_strip)

		# coupling_up is used to form the matrix that couples the nth strip or bar to the n+1th strip or bar
		coupling_up = []
		for i in range(N):
			# c determines the connectivity (fraction of good links) on the lattice
			if np.random.random() < c:
				coupling_up.append(1.0)  # t_hi
			else:
				coupling_up.append(t_low)

		coupling_matrix_up = np.diag(coupling_up)
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

def doCalc(eps,min_Lz,L,W,t_low,c,E,dim):
	#eps: desired error
	#min_Lz: At least this many slices will be computed, no matter what
	#L, V, W: size, potential, diagonal disorder
	
	#P(t)= c*delta(t-t_hi) + (1-c)*delta(t-t_low)
	#t_hi is fixed at 1
	#c: driving parameter. should be 0<c<1
	#t_low: sets low off diagonal disorder value. Needs to be low enough to be in conducting regime

	#housekeeping

	if dim==2:
		N=L
	else:
		N=L*L
	
	
	#Performs the actual localization length and conductance calculations
	eps_N=100000000000
	Lz=0 #running length
	n_i=5 #number of steps between orthogonalizations.
	n_i_min=5 #Set n_i_min=n_i if you want to force n_i
	Nr=1000 #number of T matrices to generate Q0 with

	#generate first coupling matrix. If c=1, this is the identity matrix.
	coupling_down=[]
	for i in range(N):
		if np.random.random()<c:
			coupling_down.append(1)
		else:
			coupling_down.append(t_low)
	coupling_matrix_down = np.diag(coupling_down)
	#Generate Q0
	Q0=np.random.rand(2*N,N)-0.5
	Q0 = Q0.astype(np.float64)
	Q0, r = np.linalg.qr(Q0)
	
	for i in range(Nr):
		T, coupling_matrix_down =Create_Transfer_Matrix_Chase(coupling_matrix_down,W,t_low,c,L,E,dim)
		Q0=np.matmul(T,Q0)
		if i%n_i==0:
			Q0, r = np.linalg.qr(Q0)
	
	Q0, r = np.linalg.qr(Q0)
	
	d_a=np.zeros(N,dtype=np.float64)
	e_a=np.zeros(N,dtype=np.float64)
	
	lya=list()
	glst=list()
	Umat=Q0
	timeit = 0
	cnt =0
	while eps_N>eps or Lz<min_Lz:
		M_ni, coupling_matrix_down = Create_Transfer_Matrix_Chase(coupling_matrix_down,W,t_low,c,L,E,dim)
		
		for i in range(n_i):
			T, coupling_matrix_down = Create_Transfer_Matrix_Chase(coupling_matrix_down,W,t_low,c,L,E,dim)
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

		old_xi_a = xi_a[N-1]
		sm_pos = np.where(xi_a>0, xi_a, np.inf).min()
		if old_xi_a !=sm_pos:
			cnt += 1
		lya.append(sm_pos)
		glst.append(np.log(np.sum(1/np.cosh(L*xi_a)**2)))
		#eps_N = eps_a[np.where(xi_a==sm_pos)]

			
		if len(lya)<=1:
			eps_N = 1.0 #avoid problems that occur if n_i is too big on the first go-through
		else:
			eps_N = stats.sem(lya)/np.mean(lya) #standard error

		monitor = True
		if Lz % 2000 == 0 and monitor==True:
			print("L: %d, c: %.3f, E: %.1f"%(L,c,E))
			print("Lz: %d" % Lz)
			print("Avg Smallest LE: %.7f" % np.mean(lya))
			print("Std. Error: %.7f%%" % (stats.sem(lya) / np.mean(lya) * 100))
			print("# of discrepencies: %d"%cnt)
			print("======================================")
			timeit = time.time()

	smallestLEAvg = np.mean(lya)
	
	g=np.exp(np.mean(glst))
	print("Calculation complete:")
	print("L: %d, c: %.3f, E: %.1f" % (L, c, E))
	print("Lz: %d" % Lz)
	print("Avg Smallest LE: %.7f" % np.mean(lya))
	print("Std. Error: %.7f%%" % (stats.sem(lya) / np.mean(lya) * 100))
	print("Std. Dev: %.7f"%np.std(lya))
	print("======================================")
	return np.array([float(smallestLEAvg), np.std(lya) , g],dtype=object) #avg of LEs, standard error of mean, g


def main():
	filename = "offdiagtest.txt"
	eps = 1.0
	min_Lz = 10000
	L = 6
	W = 10
	t_low = 0.3
	c = 0.14
	E = 10
	dim = 3
	avg = 1

	crange = np.linspace(0.35,0.45, 11)
	Lrange = np.arange(8, 14)
	params=(eps,min_Lz,L,W,t_low,c,E,dim)

	pool = mp.Pool(processes=6)
	start = time.time()
	results = pool.starmap(doCalc, ((params, ) * avg))
	#results = doCalc(*params)

	end = time.time()
	#print("g: %.7f"%ret[2])
	#save(params, ret, avg, filename)
	results = np.squeeze(results)
	plt.hist(results[:,0], bins='auto')
	plt.xlabel("Lyapunov Exp")
	avgSmPosLE = np.average(results[:,0])
	SESmPosLE = np.sqrt(np.sum(results[:,1]**2))/(avgSmPosLE * np.sqrt(avg))
	#SESmPosLE = stats.sem(results[:,0])/avgSmPosLE
	print("Average LE: %.7f" % avgSmPosLE)
	print("Std Err. LE: %.7f" % SESmPosLE)
	print(str(end-start) + " seconds elapsed.")
	plt.show()

if __name__=="__main__":
	np.set_printoptions(linewidth=400)
	cProfile.run('main()', sort='time')
	#main()