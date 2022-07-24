import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
import matplotlib.ticker as mticker
from scipy import stats
from scipy import optimize
from scipy import special
from scipy.stats import chi2
from collections import defaultdict
import pygmo as pg
import pandas as pd
import warnings
from multiprocessing import Pool
import time
import sys
import os.path
sys.path.append(
    os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir)))
import config as cfg
from datetime import datetime
import itertools


if __name__ == '__main__':
    class objective_function:
            def __init__(self,Lval,Tvarval,Lambda):
                self.L=Lval
                self.Tvar=Tvarval
                self.Lambda=Lambda


            def fitness(self, x):
                def scalingFunc(T, Lval, Tc, nu, y, A, b, c):

                    t = (T - Tc) / Tc  # This is filled with values for different L

                    powers_of_t_chi = np.power(np.column_stack([t] * (m_R)),
                                            np.arange(1, m_R + 1).transpose())  # (powers of t, L data)
                    # [t] * (m_R) = [t, t, t, t] (m_R=4 times)
                    # [t, t^2, t^3, t^4]

                    chi = np.dot(powers_of_t_chi, b)
                    chi_vec = np.power(chi * (self.L ** (1 / nu)), np.column_stack(
                        [np.arange(0, n_R + 1)] * len(t)))  # vector containing 1, chi*L**1/nu, chi**2*L**2/nu, ...

                    if n_I > 0:
                        powers_of_t_psi = np.power(np.column_stack([t] * (m_I + 1)),
                                                np.arange(0, m_I + 1).transpose())  # Always has 1 in the first column
                        psi = np.dot(powers_of_t_psi, c)
                        psi_vec = np.power(psi * (self.L ** y), np.column_stack([np.arange(0, n_I + 1)] * len(t)))

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
                    Lambda_scaled = scalingFunc(self.Tvar, self.L, Tc, nu, y, A, b, c)
                    c2 = np.sum(np.abs(self.Lambda - Lambda_scaled)**2)
                    dof = len(self.Tvar) - numParams

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

    def timing(name='',verbose=True):
        global timestart,timeend
        if timestart == 0:
            timestart = time.time()
        else:
            timeend = time.time()
            tot=timeend-timestart
            if verbose:
                print(name + ' took ' + str(tot) + 's')
            timestart=0

    def plotScalingFunc(T, L, args):
        sz=8#Size of the scaling curve points
        #markerlist=('.', '+', 'o', '*','v', '^', '|', '>', 's', '4', 'p','x','d')
        #colorlist=('tab:blue','tab:orange','tab:green','tab:red','tab:purple','tab:brown','tab:pink','tab:gray','tab:olive','tab:cyan')
        
        clr = itertools.cycle(colorlist) 
        #mkr = itertools.cycle(markerlist) 

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
        #ax1.scatter(chi_exp_L,Lambda)
        
        uniT=np.unique(T)[::-1]
        for ti in uniT:
            # pb=np.percentile(T,25)
            # pp=np.percentile(T,50)
            # po=np.percentile(T,75)
            # pr=np.percentile(T,100)
            cle=chi_exp_L[T==ti]
            lam=Lambda[T==ti]
            # co=''


            # if ti < pb:
            #     co='black'
            # elif ti < pp:
            #     co='purple'
            # elif ti < po:
            #     co='orange'
            # else:
            #     co='red'

            #@TODO remove if you want the scaling curve colorized based on L vals
            #co='black'

            ax1.scatter(cle, lam,marker='o',s=sz,c=next(clr),edgecolor='black',linewidth=.2)#marker=next(mkr)
        
        # cleAbove=chi_exp_L[L>24]
        # cleBelow=chi_exp_L[L<=24]
        # lAbove=Lambda[L>24]
        # lBelow=Lambda[L<=24]
        miny=np.min(Lambda)
        maxy=np.max(Lambda)
        ylen=maxy-miny
        numticks=10
        ticks=np.arange(miny,maxy,ylen/numticks)
        print(ticks)

        ax1.set_xscale('log')
        ax1.set_yscale('linear')

        #ax1.get_yaxis().get_major_formatter().labelOnlyBase = False
        ax1.set_yticks(ticks)
        #ax1.yaxis.set_ticks(np.arange(miny,maxy,ylen/numticks))

        #ax2.scatter(chi_L_exp, Lambda)
        ax1.set_ylabel(r'$\Lambda$', fontsize=fs)
        # ax1.set_ylabel(r'$g$', fontsize=fs)
        ax1.set_xlabel(r'$|\chi|^\nu*L$', fontsize=fs)


        
        #ax2.set_xlabel(r'$\chi L^{1/\nu}$', fontsize=fs)
        #ax2.set_xscale('linear')

        #ax2.set_yscale('log')
        
        
    def plotRawData(Tvar,L,args):
        legendlimiter=0
        legendskip=2#if this is 2 skip every 3rd c value in the legend
            # Plot raw data
        Lrange = Lrange = np.unique(L)
        clr = itertools.cycle(colorlist)
        for T in np.unique(Tvar)[::-1]:
            toPlotX = []
            toPlotY = []
            Yerror = []
            for i, Tv in enumerate(Tvar):
                if T == Tv:
                    toPlotY.append(Lambda[i])
                    toPlotX.append(L[i])
                    Yerror.append(sigma[i])
            npX = np.array(toPlotX)
            npY = np.array(toPlotY)
            npYerror= np.array(Yerror)
            inds = npX.argsort()  # sort the data according to L to make sure it plots right
            npX = npX[inds]
            npY = npY[inds]
            npYerror = npYerror[inds]
            if np.array_equal(Tvar,W):
                lbl='W='
            else:
                lbl='c='
            
            if legendlimiter<legendskip:
                lbl=lbl + str(round(T,4))
                legendlimiter = legendlimiter+1
            else:
                lbl='_nolegend_'
                legendlimiter = 0


            ax3.errorbar(npX, npY,  fmt='o-', yerr=npYerror, label=lbl, c=next(clr), ecolor='k', capthick=2, markersize=0, barsabove=True, capsize=0)
            #ax3.semilogy(npX, npY, 'o-', label=lbl+str(round(T,2)))

        ax3.set_xlabel('L', fontsize=fs)
        #ax3.set_xscale('log')
        ax3.set_yscale('log')
        Lrange = np.arange(min(Lrange),max(Lrange)+1,step=2,dtype=int)
        ax3.set_xticks(Lrange)
        ax3.set_xticklabels(list(map(str, Lrange)))
        #ax3.set_yticks([],[])
        ax3.tick_params('y',labelleft=False)
        #ax3.xaxis.set_minor_formatter(mticker.ScalarFormatter())
        #plt.ticklabel_format(axis='x',style='plain')
        #fig2, ax4 = plt.subplots(nrows=1, ncols=1, figsize=(8, 5), sharey=True)
        ax3.legend(loc='upper right', bbox_to_anchor=(1.4, 1.05))


    def bootVals(Lambda,L,Tvar,sigma,n):
        # bootstrap method  generate n sets of randints.
        output=np.array([])
        Lambdanew=np.array([])
        Lnew=np.array([])
        Tvarnew=np.array([])
        sigmanew=np.array([])

        if n==1:
            #don't do any sampling
            return np.array([1]), Lambda, L, Tvar, sigma

        for i in range(0,n):
            outputtemp=(np.random.choice(len(Lambda), int(len(Lambda)), replace=True))
            outputtemp=np.sort(outputtemp)
            if len(output)==0:
                output=outputtemp
                Lambdanew=Lambda[outputtemp]
                Lnew=L[outputtemp]
                Tvarnew=Tvar[outputtemp]
                sigmanew=sigma[outputtemp]
            else:
                output=np.vstack([output,outputtemp])
                Lambdanew=np.vstack([Lambdanew,Lambda[outputtemp]])
                Lnew=np.vstack([Lnew,L[outputtemp]])
                Tvarnew=np.vstack([Tvarnew,Tvar[outputtemp]])
                sigmanew=np.vstack([sigmanew,sigma[outputtemp]])
        return output,Lambdanew,Lnew,Tvarnew,sigmanew

    def verifyCL(d,minL,maxL):#remove all data which do not have C values for all L.
        d=d[d['L']<=maxL]
        data=d[d['L']>=minL]

        uniL=np.unique(data['L'])
        lenL=maxL-minL+1#len(uniL)

        print('len l: ' + str(lenL))
        uniC=np.unique(data['c'])

        for ci in uniC:
            allC=data[data['c']==ci]

            curUniL=np.unique(allC['L'])
            curLenL=len(curUniL)

            if curLenL != lenL:
                data= data[data['c']!=ci]
                #print('not enough L for c= ' + str(ci))
            #else:
                #print('enough L for c= ' + str(ci))

        return data

    def compressLambda(Lambda_in, L_in, c_in, sigma_in):
        # Eliminates multiple Lambdas per (c,L) pair via average (weighted by std dev) and propagates uncertainty through sigma



        #input: Lambda, L, c, sigma data. May be multiple Lambdas per (c,L)
        #output: for each (c,L), weighted average and std dev
        c_unique = np.unique(c_in)
        L_unique = np.unique(L_in)

        cL_pairs = [[c,L] for c in c_unique for L in L_unique]
        Lambda_out = np.zeros(len(cL_pairs))

        L_out = np.zeros(len(cL_pairs))
        c_out = np.zeros(len(cL_pairs))

        Lambda_out_sigma = np.zeros(len(cL_pairs))

        for i, (c,L) in enumerate(cL_pairs):
            cL_ind = np.where((c_in==c) & (L_in==L))[0] #indices of matching c,L pair in Lambda

            matchingLambdas = [Lambda_in[j] for j in cL_ind]
            matchingSigmas = [sigma_in[j] for j in cL_ind]

            if len(matchingLambdas)>0 and len(matchingSigmas)>0:
                # Final average is weighted by std devs
                Lambda_out[i] = np.average(matchingLambdas, weights=1/np.square(matchingSigmas))
                # Final std dev calculated the usual way
                Lambda_out_sigma[i] = 1/np.sqrt(np.sum(1/np.square(matchingSigmas)))
                # corresponding c,L arrays with same indexes
                c_out[i] = c
                L_out[i] = L
        # Finally, remove zeros in case some c,L pairs don't have an associated Lambda
        L_out = L_out[Lambda_out!=0]
        c_out = c_out[Lambda_out!=0]
        Lambda_out_sigma = Lambda_out_sigma[Lambda_out!=0]
        Lambda_out = Lambda_out[Lambda_out != 0]
        return Lambda_out, L_out, c_out, Lambda_out_sigma

    def showplt(plt, show):
        if show:
            plt.show()

    def runFssAnalysis(rssize,bL,bTvar,bLamda,verbose=False):
        #list containing problem descriptions for each bootstrap resample
        problist = [pg.problem(objective_function(bL[i, :], bTvar[i, :], bLambda[i, :])) for i in range(rssize)]
        probbackup = problist

        Nurange=np.array([])
        Tcrange=np.array([])

        #definition of the algorithm
        algo = pg.algorithm(pg.cmaes(gen=gen_repeats, force_bounds=use_bounds, ftol=1e-8))

        pg.mp_island.init_pool(processes=numCPUs)

        #print("Computing main fit")
        #pop = pg.population(prob=objective_function(L, Tvar, Lambda), size=500)
        #algo.evolve(pop)

        #best_main = pop.champion_x


        #timing()
        islands = [pg.island(algo=algo, prob=problist[i], size=100, udi=pg.mp_island()) for i in range(rssize)]
        #timing("Starting bootstrap")

        #timing()
        _ = [isl.evolve() for isl in islands]
        solutions = np.array([])
        #timing('@a')

        #timing()
        #resample and Cc check loop
        while solutions.shape[0]<rssize:

            for ind, isl in enumerate(islands):
                if isl.status==pg.evolve_status.idle_error or isl.status==pg.evolve_status.busy_error:  # something blew up, try again
                    islands[ind] = pg.island(algo=algo, prob=probbackup[ind], size=100, udi=pg.mp_island())
                    islands[ind].evolve()
                    print('Island %i had an exception, restarting...' % ind)
                if isl.status==pg.evolve_status.idle:  # completed normally, store the result
                    solution = isl.get_population().champion_x
                    score = isl.get_population().champion_f
                    Tcrange = np.append(Tcrange, solution[0])
                    Nurange = np.append(Nurange, solution[1])
                    if len(solutions)==0:
                        solutions = np.expand_dims(solution, axis=0)
                    else:
                        solutions = np.vstack([solutions, solution])

                    percent=round(len(solutions) / rssize * 100)
                    #timing(str(percent)+'%')
                    #timing()
                    
                    print('Bootstrapping {}% Tc: {}, nu: {} from island {}'.format(percent,
                                                                        round(solution[0], 2),
                                                                        round(solution[1], 2), ind))
                    
                    #finally, remove the island from the array so its not counted twice (or three times, ...)
                    del islands[ind]
            time.sleep(1)

        _ = [isl.wait() for isl in islands]
        return solutions,Tcrange,Nurange

    def cullAndClenseData(data,minL,maxL,cc,window_center,window_offset,window_width,cwidth,closewidth,excludeCs=[]):

        pre=len(np.unique(data[:,0]))
        #omit L less than minL to control finite size effects
        data = data[data[:,0]>=minL]
        data = data[data[:,0]<=maxL]
        post=len(np.unique(data[:,0]))

        if post != pre:
            print(f'Cut {pre - post} values from L bounds')

        
        pre=len(np.unique(data[:,2]))
        data = data[np.abs(data[:,2]-window_center)<=window_offset+window_width]
        data = data[np.abs(data[:,2]-window_center)>=window_offset]
        post=len(np.unique(data[:,2]))

        if post != pre:
            print(f'Cut {pre - post} values from window bounds')

        
        pre=len(np.unique(data[:,2]))
        for exc in excludeCs:
            data = data[data[:,2]!=exc]
        post=len(np.unique(data[:,2]))

        if post != pre:
            print(f'Cut {pre - post} values from excluded c\'s')

        if cc!=0:
            minC=cc-cwidth*cc#.2935#
            maxC=cc+cwidth*cc

            minCloseC=cc-closewidth*cc
            maxCloseC=cc+closewidth*cc
            pre=len(np.unique(data[:,2]))
            data = data[data[:,2]>=minC]
            data = data[data[:,2]<=maxC]
            post=len(np.unique(data[:,2]))
            if post != pre:
                print(f'Cut {pre - post} c values for being too far from criticality')
            
            #print('unic: ' + str(np.unique(data[:,2])))

            if closewidth != 0:
                pre=len(np.unique(data[:,2]))
                mask= (data[:,2]>=minCloseC) & (data[:,2]<=maxCloseC)
                data = data[np.logical_not(mask)]

                post=len(np.unique(data[:,2]))
                if post != pre:
                    print(f'Cut {pre - post} c values for being too close to criticality')

            #print('unic: ' + str(np.unique(data[:,2])))
        
        

        #Bad test - trying removing c's that are too precise lol
        #print(len(data[:,2].astype(str)))
        #lenv=np.vectorize(len)
        #strv=np.vectorize(str)
        #print(lenv(strv(data[:,2])))
        #data = data[lenv(strv(data[:,2]))<6]


        Lrange = np.unique(data[:, 0])
        Wrange = np.unique(data[:, 1])
        crange = np.unique(data[:, 2])
        lenL=len(Lrange)

        # for i in crange:
        #     print(i)
        #     print(len(str(i)))

        L = data[:, 0]
        W = data[:, 1]
        c = data[:, 2]
        Lambda = data[:, 3]

        #c=c.round(3)#@TODO this is bad probably but the plots are bad if not
        print('Using ' +str(len(data[:,0]))+ ' total values')
        print('unic: ' + str(np.unique(c)))

        sigma = np.array(df['std'].to_list())
        #sigma = input[:, 4]
        g = np.array(df['g'].to_list())
        #g=input[:, 5]

        # Eliminates multiple Lambdas per (c,L) pair via average (weighted by std dev) and propagates uncertainty through sigma
        Lambda, L, c, sigma = compressLambda(Lambda, L, c, sigma)
        return Lambda, L, c, W, sigma, g


#END OF FUNCTION DEFINITIONS


#initializations

    now = datetime.now()
    startt=time.time()
    np.seterr(all='raise')
    scriptdir=os.getcwd() #os.path.dirname(__file__) 
    timestart=timeend=0

    colorlist=('brown','red','tomato', 'orangered','darkorange','orange','gold','greenyellow','green','springgreen','turquoise','teal','dodgerblue','midnightblue','blue','indigo','purple','magenta','pink')

    
    print(len(sys.argv))

    if len(sys.argv) == 8:
        print(sys.argv)
        datafile=str(sys.argv[1])

        minL = int(sys.argv[2])
        maxL= int(sys.argv[3])

        minC=float(sys.argv[4])
        maxC=float(sys.argv[5])
        resamplesize = int(sys.argv[6])
        numCPUs = int(sys.argv[7])
        datadir= scriptdir+'/'
    else:
        E=2
        W=10
        filestart='E'+str(E)+'W'+str(W)

        datafile=filestart+'Lz100K.csv'

        minL = 10

        maxL= 24

        maxL_lower= maxL
        maxL_upper= maxL


        #E0CC=.29
        #E2CC=.29

        #IF THIS IS 0 THEN WE WILL DO A CRITC CHECK
        critC=00.294#0.3#.2948

        cwidth=.1#use C values within 11% of the critical C
        closewidth=0#.004#drop c values within .4% of critical C

        
        #minC=minCE0=E0CC-cwidth*E0CC
        #maxC=maxCEo=E0CC+cwidth*E0CC
        
        
        
        # number of resamples
        numCritCheck=6
        resamplesize = 20


        numCPUs = 2 # number of processors to use

        datadir= os.path.join(scriptdir, 'data\\')#directory to search for data l

    verbose = True
    fs = 18 #font size 

    gen_repeats=1000

    pdf_name_identifier=''#'-'+str(resamplesize)+'rs'#"high-resample"#will be added to the 
    data_identifier=''

    window_width = 1.0 #width of window
    window_offset = 0.0  #  distance from window center to near edge of window
    window_center = 0.63

    

    cutoffdate = datetime.now()#datetime(2022,4,30)

    crit_bound_lower, crit_bound_upper = 0.01, .99  # critical value bounds

    excludeCs=[]#[.2948]

    #crit_bound_lower, crit_bound_upper = min(c), max(c) # critical value bounds
    nu_bound_lower, nu_bound_upper = 0.1, 3  # nu bounds
    y_bound_lower, y_bound_upper = -100.0, -0.1  # y bounds
    param_bound_lower, param_bound_upper = -500.0, 500.1  # all other bounds
    use_bounds = True
    displayPlots=True
    plotRaw=True

    paperPlot=False#if we want to make a plot worth of paper

    # orders of expansion
    n_R = 3#3
    n_I = 1#1
    m_R = 2#2
    m_I = 1#1


    if not paperPlot:
        fig1, (ax1, ax3) = plt.subplots(nrows=1, ncols=2, figsize=(11, 6), sharey=False)
        box = ax3.get_position()
        box.x0 = box.x0 - 0.05
        box.x1 = box.x1 - 0.05
        ax3.set_position(box)
    else:
        fig1, ax1 = plt.subplots(nrows=1, ncols=1, figsize=(6, 6))

    lrange=np.arange(maxL_lower,maxL_upper)
    if maxL_upper == maxL_lower:
        lrange=np.arange(maxL_lower,maxL_upper+1)

    #main loop
    for maxL in lrange:
        print('Max L:'+str(maxL))

        timing('data validation',verbose)
        #data validation
        if n_I > 0:
            numParams = (n_I + 1) * (n_R + 1) + m_R + m_I - 1
        else:
            numParams = n_R + m_R

        print(datafile)
        filename  = cfg.datafilename(datafile)#wrapper required for plot titles and files saved
        #input = np.array(openfile(tmpdir+datafile))
        df = pd.read_csv(datadir+datafile,engine='python')#,delimiter='\t')#,encoding='utf8',sep=',',

        #print(len(df.index))
        #cut off old data by setting a date here
        df['date']=pd.to_datetime(df['date'])
        df=df[df['date']<cutoffdate]
        #print(len(df.index))


        #df=verifyCL(df,minL,maxL)#@TODO removes all c values that do not have a value for EVERY L

        #print(len(df.index))
        print('Loaded ' +str(len(df.index))+ ' total values')
        #cenrange = np.unique(input[:,6])
        #data = input[:, 0:7]  # L, W, c, LE,std,g,runtime
        #print(np.shape(data))
        #data=np.transpose(np.array([df['L'].to_list(),df['W'].to_list(),df['c'].to_list(),df['lyap'].to_list(),df['std'].to_list(),df['runtime'].to_list()]))
        data=np.transpose(np.array([df['L'],df['W'],df['c'],df['lyap'],df['std'],df['runtime']]))
        data[:, 3] = 1 / (data[:, 0] * data[:, 3])  # L, W, c, normalized localization length
        #print(np.shape(data))



        # sort according to L
        data = data[np.argsort(data[:, 0])]
        timing('data validation')
        ranOnce=False
        # now do a cCheck
        #Two cases, we want to check for the critical c or we dont
        while critC ==0 or ranOnce is False:
            
            if critC == 0:
                numRun=numCritCheck
                curclosewidth=0#don't exclude any vals
            else:
                numRun=resamplesize
                curclosewidth = closewidth
                ranOnce = True

            Lambda, L, c, W, sigma, g = cullAndClenseData(data,minL,maxL,critC,window_center,window_offset,window_width,cwidth,curclosewidth,excludeCs)

            if c.size == 0:
                print('Culled all data, no values to run scaling with')
                quit()

            # set the driving parameter
            Tvar = c

            randints, bLambda, bL, bTvar, bsigma = bootVals(Lambda, L, Tvar, sigma, resamplesize)  # getting new variables with prefix b to designate bootstrap resamples
            print(str(numParams) + "+3 parameters")

            
            if numRun == 1 and bL.ndim == 1:  # fix edge case
                bL = np.expand_dims(L, 0)
                bTvar = np.expand_dims(Tvar, 0)
                bLambda = np.expand_dims(Lambda, 0)


            

            solutions,Tcrange, Nurange = runFssAnalysis(numRun,bL,bTvar,bLambda,verbose)
            critC= np.median(Tcrange)#round(np.median(Tcrange),3)
            print('Found Cc of ' + str(critC))
            

        #solutions,Tcrange, Nurange = runFssAnalysis(resamplesize,bL,bTvar,bLambda)

        print('Finished')


        #results

        TcChoke = np.percentile(Tcrange, [37.5, 62.5]) # tight range
        Tcrangef = Tcrange[Tcrange >= TcChoke[0]]
        Nurangef = Nurange[Tcrange >= TcChoke[0]]
        Nurangef = Nurangef[Tcrangef <= TcChoke[1]]
        Tcrangef = Tcrangef[Tcrangef <= TcChoke[1]]

        nu_1CI = np.percentile(Nurange, [2.5, 97.5])
        TcCI = np.percentile(Tcrange, [2.5, 97.5])
        Tcfinal = np.median(Tcrange)
        Nufinal = np.median(Nurange) # Don't take Nufinal if you are pregnant or nursing

        if len(Tcrangef) >= 2 and len(Nurangef)>=2:
            print('50%ile Tc range: {}, {}'.format(round(min(Tcrangef),3), round(max(Tcrangef),3)))
            print('50%ile nu range: {}, {}'.format(round(min(Nurangef),3), round(max(Nurangef),3)))
        if len(TcCI) >= 2 and len(nu_1CI)>=2:
            print('95%ile Tc range: {}, {}'.format(round(TcCI[0], 3), round(TcCI[1], 3)))
            print('95%ile nu range: {}, {}'.format(round(nu_1CI[0], 3), round(nu_1CI[1], 3)))

        print('Cc: {} [{}, {}]'.format(round(Tcfinal,3), round(TcCI[0], 3), round(TcCI[1], 3)))
        print('nu: {} [{}, {}]'.format(round(Nufinal,3), round(nu_1CI[0], 3), round(nu_1CI[1], 3)))
        #solution = best_main

        solution = np.median(solutions, axis=0)
        print("Solution"+str(solution))

        expansionParamStr='n_R, n_I, m_R, m_I = {}, {}, {}, {}'.format(n_R, n_I, m_R, m_I)
        print(expansionParamStr)


        #plt.figure()
        #plt.hist(Nurange, label=r'$\nu$', color='#1a1af980')
        #plt.hist(Tcrange, label=r'$c_c$', color='r')
        #plt.xlabel(r'$\nu$')
        #plt.ylabel('counts')

        plotScalingFunc(Tvar, L, solution)

        if not paperPlot:
           plotRawData(Tvar,L,solution)
        

    #make title for plot, save figures, and store results data in a excel spreadsheet to analyze 
        endt=time.time()
        exet=str("%.2fs" % (endt-startt))
        print('execution time: ', exet)

        nuval=round(solution[1],2)
        cc=round(solution[0],2)
        print(nuval)
        #plt.title
        
        ostr=str(float(window_offset)).split('.')[1]
        wstr=str(float(window_width)).split('.')[1]
        if nuval > 1:
            nustr=str(round(nuval,2)).replace('.','_')
        else:
            nustr=str(round(nuval,2)).split('.')[1]

        fname='L_%s-%s--nu_%s-r%i' % (str(int(minL)),str(int(maxL)),nustr,resamplesize)
        
        data_files_used=df['fname'].unique()

        if window_offset != 0:
            cc='--'
        title= datafile.removesuffix('.csv')

        dcmplace=3#@TODO change to 2 for final
        #title = title+" %s\n\nCc: %s - nu: %f" % (cc, nuval)
        title = title + '  ' + expansionParamStr + '\nCc: {} [{}, {}]'.format(round(Tcfinal,dcmplace), round(TcCI[0], dcmplace), round(TcCI[1], dcmplace)) + '\nnu: {} [{}, {}]'.format(round(Nufinal,dcmplace), round(nu_1CI[0], dcmplace), round(nu_1CI[1], dcmplace))
    
        plot_name=fname +'-'+ pdf_name_identifier+'.pdf'
        run_file_name=cfg.runfilename(plot_name)

        if window_offset != 0.0 or window_width != 1.0:
            title = title+ " - offset: %.2f - width: %.2f" % ( window_offset,window_width)

        
        if not paperPlot:
            fig1.suptitle(title)
           
            fig1.savefig(run_file_name)



        #datastring='%f, %f, %f, %f, %f' % (solution[0], solution[1], window_center, window_width, window_offset)
        csv_column_names= ['Date','Cc','Cc err upper','Cc err lower','Nu','Nu err upper','Nu err lower','Min L', 'Max L', 'Run time', 'CPUs','Bootstrap Resample size',
                            'Forced bounds','Crit bound lower', 'Crit bound upper','Nu bound lower','Nu bound upper',
                            'Gen Repeats', 'n_R','n_I','m_R','m_I',
                            'Window center', 'Window width','Window offset','Run identifier', 'Plot pdf','Data files used']


        datacsv=[now.strftime("%D %H:%M"), cc, round(TcCI[0], 3),round(TcCI[1], 3),round(Nufinal,3),round(nu_1CI[0], 3),round(nu_1CI[1], 3), minL, maxL, exet.removesuffix('s'),numCPUs, resamplesize,
                    use_bounds,crit_bound_lower,crit_bound_upper,nu_bound_lower,nu_bound_upper,
                    gen_repeats,n_R,n_I,m_R,m_I,
                    window_center, window_width, window_offset,data_identifier,plot_name,data_files_used]

                  
        saved = False
        while(not saved):
            try:
                csv_name=cfg.savecsv(datacsv,csv_column_names)
            except:
                input('Please close the csv then press enter to save again.')
                continue
            saved = True
        
        print('Saved output files: %s & %s' % (fname,csv_name))

        showplt(plt,displayPlots)
   
#E for s in settings