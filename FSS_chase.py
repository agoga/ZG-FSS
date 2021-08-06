# -*- coding: utf-8 -*-
"""
Created on Sun Mar  7 19:20:32 2021

@author: chase
"""

import matplotlib.pyplot as plt
import numpy as np
import os
import helpers as h



def openfile(filename):
    f = open(filename,"r")
    outputlst=[]
    for line in f:
        strlist = line.split()
        L=float(strlist[4][:-1])
        W=float(strlist[5][:-1])
        c=float(strlist[7][:-1])
        lyap=float(strlist[10][1:])
        outputlst.append([L, W, c, lyap])
    return outputlst


def checkfit(xdata,ydata):
    '''
    inputs: 
    only valid for nu>0
    ydata is a list of (scaled) y data sets for varying M values
    xdata is a list of (scaled) x data sets for varying M values
    
    outputs:
    error: A least squares determination of how good the fit is. 
    '''
    used_bases=[]
    error=0
    for i in range(len(xdata)):
        base_curve_x=xdata[len(xdata)-i-1]
        base_curve_y=ydata[len(ydata)-i-1]
        used_bases.append(len(xdata)-i)
        for j in range(len(xdata)):
            if j not in used_bases:
                for k in range(len(base_curve_x)-1):
                    del_x=base_curve_x[k+1]-base_curve_x[k]
                    del_y=base_curve_y[k+1]-base_curve_y[k]
                    slope=del_y/del_x
                    for n in range(len(xdata[j])):
                        if xdata[j][n]<base_curve_x[k+1] and xdata[j][n]>base_curve_x[k]:
                            interpolation_dist=xdata[j][n]-base_curve_x[k]
                            base_y_compare=base_curve_y[k]+slope*interpolation_dist
                            dif=base_y_compare-ydata[j][n]
                            scale=base_y_compare+ydata[j][n]
                            error+=dif**2/scale**2
    return error


raw_data=[]

#f2=open('means_M_W10,c_varied.txt','r')
#lines=f2.readlines()
#i=0
#j=0
#for line in lines:
#    if i%2==1:
#        raw_data.append([])
#        new_line=line.replace('[','')
#        newer_line=new_line.replace(']','')
#        split_line=newer_line.split(',')
#        for num in split_line:
#            raw_data[j].append(float(num))
#        j+=1
#    i+=1
#
#
#
#M_Array=[4,6,8,10,12,14]
#fraction_list=np.linspace(0,1,20)


# outputfile='offdiag_t30_full.txt'
# # outputfile='DD.txt'

# output=openfile(outputfile)



# y_data=[]
# L_list=[]
# c_list=[]
# W_list=[]
# for sim in output:
#     L=sim[0]
#     c=sim[2]
#     W=sim[1]
#     if L not in L_list:
#         L_list.append(L)
#     if c not in c_list:
#         c_list.append(c)
#     if W not in W_list:
#         W_list.append(W)
# L_list.sort()
# c_list.sort()
# W_list.sort()

# #This snippet will show all of the lengths' loc lengths as a function of c
# i=0

# g_data=[]
# for L in L_list:
#     y_data.append([])
#     g_data.append([])
#     for c in c_list:
#         for sim in output:
#             if sim[0]==L and sim[2]==c:
#                 Lambda=1/sim[3]
#                 T=np.exp(-2/(Lambda/sim[0]))
# #                y_data[i].append(1/(sim[0]*sim[3]))
#                 y_data[i].append(Lambda/sim[0])
#                 # g_data[i].append((4/np.cosh((2*sim[0]/Lambda))+1))
#                 g_data[i].append(1/np.pi*T/(1-T))
#     i+=1

#This snippet will show all of the lengths' loc lengths as a function of W
# i=0

# g_data=[]
# for L in L_list:
#     y_data.append([])
#     g_data.append([])
#     for W in W_list:
#         for sim in output:
#             if sim[0]==L and sim[1]==W:
#                 Lambda=1/sim[3]
# #                y_data[i].append(1/(sim[0]*sim[3]))
#                 y_data[i].append(Lambda/sim[0])
#                 g_data[i].append((4/np.cosh((2*sim[0]/Lambda))+1))
#     i+=1



#This snippet will show all of the c's loc lengths as a function of L
# i=0
# g_data=[]
# for c in c_list:
#     y_data.append([])
#     g_data.append([])
#     for L in L_list:
#         for sim in output:
#             if sim[0]==L and sim[2]==c:
#                 Lambda=2*L/sim[3]
#                 y_data[i].append(1/(sim[0]*sim[3]))
#                 g_data[i].append(4/np.cosh((2*sim[0]/Lambda))+1)
#     i+=1
        

# plt.figure(figsize=(10,7))
# for i in range(len(y_data)):
#     plt.plot(c_list,y_data[i], label = 'L= ' +str(L_list[i]),marker='^')
# plt.legend(loc=2)
# plt.ylabel(r'$\Lambda$')
# #plt.ylabel('g')
# plt.xlabel('c')
# plt.yscale('log')
# plt.xscale('log')

# plt.show()


# x_data=W_list

#This plot should identify WHERE the transition is, we should see lines crossing.
#for i in range(len(raw_data)):
#    y_data.append([])
#    for j in range(len(raw_data[i])):
#        y_data[i].append(raw_data[i][j]/L_list[i])
#    plt.plot(x_data,y_data[i],label='M= ' + str(L_list[i]))
#plt.xlabel('c')
#plt.ylabel(r'$\lambda_M$/M')
#plt.legend(loc=2)
#plt.show()



def compute_c_nu():

    
    outputfile = h.datafilename('offdiagE6W10.txt')
    output=openfile(outputfile)
    
    
    
    y_data=[]
    L_list=[]
    c_list=[]
    W_list=[]
    for sim in output:
        L=sim[0]
        c=sim[2]
        W=sim[1]
        if L not in L_list:
            L_list.append(L)
        if c not in c_list:
            c_list.append(c)
        if W not in W_list:
            W_list.append(W)
        
    L_list.sort()
    c_list.sort()
    W_list.sort()
    
    x_data=c_list
    

    
    i=0
    
    g_data=[]
    for L in L_list:
        six_hit=False
        y_data.append([])
        g_data.append([])
        # x_data.append([])
        for c in x_data:
            for sim in output:
                if sim[0]==L and sim[2]==c:
                    Lambda=1/sim[3]
                    T=np.exp(-2/(Lambda/sim[0]))
    #                y_data[i].append(1/(sim[0]*sim[3]))
                    y_data[i].append(Lambda/sim[0])
                    # g_data[i].append((4/np.cosh((2*sim[0]/Lambda))+1))
                    g_data[i].append(1/np.pi*T/(1-T))
                    # x_data[i].append(c)
                    # if sim[2]==.3:
                    #     if six_hit==True:
                    #         y_data[i].pop(len(y_data[i])-1)
                    #     six_hit=True
                        

        i+=1
        
    for i in range(len(y_data)):

        plt.plot(x_data,y_data[i],label='M= ' + str(L_list[i]))
    plt.xlabel('c')
    plt.ylabel(r'$\lambda_M$/M')
    plt.legend(loc=2)
    plt.show()
    
    
    nu_range=np.linspace(1,1.6,20)
    c_range=np.linspace(.7,.8,20)
    
    
    minerror=10000000
    best_nu=1
    for nu in nu_range:
        for c_crit in c_range:
            scaled_xs=[]
            for i in range(len(L_list)):
                scaled_xs.append([])
                for j in range(len(x_data)):
                    scaled_xj=(x_data[j]-c_crit)*L_list[i]**(1/nu)
                    scaled_xs[i].append(scaled_xj)
            error=checkfit(scaled_xs,y_data)
            if error<minerror:
                minerror=error
                best_nu=nu
                best_c=c_crit
    #        
    
    
    
    
    print(best_nu)
    print(best_c)
    
    # best_c=.6329
    # best_nu=1.431
    
    loss_fcn=[]
    for nu in nu_range:
        scaled_xs=[]
        for i in range(len(L_list)):
            scaled_xs.append([])
            for j in range(len(x_data)):
                scaled_xj=(x_data[j]-best_c)*L_list[i]**(1/nu)
                scaled_xs[i].append(scaled_xj)
        error=checkfit(scaled_xs,y_data)
        loss_fcn.append(error)
        
    plt.plot(nu_range,loss_fcn)
    plt.ylabel('loss function')
    plt.xlabel(r'$\nu$')
    plt.show()
    
    #best_c=.306
    # best_nu=1.5
    scaled_plot_x=[]
    scaled_plot_y=[]
    abs_scaled_x=[]
    #best_nu=1.07

    for i in range(len(L_list)):
        for j in range(len(x_data)):
            scaled_plot_x.append(((x_data[j]-best_c)/best_c)*L_list[i]**(1/best_nu))
            abs_scaled_x.append(L_list[i]/np.abs(x_data[j]-best_c)**-best_nu)
            scaled_plot_y.append(y_data[i][j])
    plt.plot(abs_scaled_x,scaled_plot_y,linestyle='None',marker='^')
    #plt.xlabel(r'$tL^{1/\nu}$')
    plt.xlabel(r'$tL^{{1/\nu}}$')
    plt.ylabel(r'$\Lambda$')
    plt.yscale('log')
    plt.xscale('log')
    plt.show()
    #plt.ylabel(r'$\Lambda/L$')
    
    
def compute_W_nu():
    
    outputfile='DD.txt'

    output=openfile(outputfile)
    
    
    
    y_data=[]
    L_list=[]
    c_list=[]
    W_list=[]
    for sim in output:
        L=sim[0]
        c=sim[2]
        W=sim[1]
        if L not in L_list:
            L_list.append(L)
        if c not in c_list:
            c_list.append(c)
        if W not in W_list:
            W_list.append(W)
    L_list.sort()
    c_list.sort()
    W_list.sort()
    
    x_data=W_list
    
    i=0

    g_data=[]
    y_data=[]
    for L in L_list:
        y_data.append([])
        g_data.append([])
        for W in W_list:
            for sim in output:
                if sim[0]==L and sim[1]==W:
                    Lambda=1/sim[3]
    #                y_data[i].append(1/(sim[0]*sim[3]))
                    y_data[i].append(Lambda/sim[0])
                    g_data[i].append((4/np.cosh((2*sim[0]/Lambda))+1))
        i+=1

    
    
    
    nu_range=np.linspace(.8,1.6,200)
    W_range=np.linspace(16,17,200)
    
    
    minerror=10000000
    best_nu=1
    for nu in nu_range:
        for W_crit in W_range:
            scaled_xs=[]
            for i in range(len(L_list)):
                scaled_xs.append([])
                for j in range(len(x_data)):
                    scaled_xj=(x_data[j]-W_crit)*L_list[i]**(1/nu)
                    scaled_xs[i].append(scaled_xj)
            error=checkfit(scaled_xs,y_data)
            if error<minerror:
                minerror=error
                best_nu=nu
                best_W=W_crit
    #        
    
    
    
    
    print(best_nu)
    print(best_W)
    
    #best_c=.305
    #best_nu=1.16
    scaled_plot_x=[]
    scaled_plot_y=[]
    abs_scaled_x=[]
    #best_nu=1.07
    
    
    for i in range(len(L_list)):
        for j in range(len(x_data)):
            scaled_plot_x.append(((x_data[j]-best_W)/best_W)*L_list[i]**(1/best_nu))
            abs_scaled_x.append(L_list[i]/np.abs(x_data[j]-best_W)**-best_nu)
            scaled_plot_y.append(y_data[i][j])
    plt.plot(abs_scaled_x,scaled_plot_y,linestyle='None',marker='^')
    #plt.xlabel(r'$tL^{1/\nu}$')
    plt.xlabel(r'$tL^{\nu}$')
    plt.ylabel('lam')
    plt.yscale('log')
    plt.xscale('log')
    #plt.ylabel(r'$\Lambda/L$')
    
    
    
    
    
def compute_W_s():
    
    outputfile='DD.txt'

    output=openfile(outputfile)
    
    
    
    y_data=[]
    L_list=[]
    c_list=[]
    W_list=[]
    for sim in output:
        L=sim[0]
        c=sim[2]
        W=sim[1]
        if L not in L_list:
            L_list.append(L)
        if c not in c_list:
            c_list.append(c)
        if W not in W_list:
            W_list.append(W)
    L_list.sort()
    c_list.sort()
    W_list.sort()
    
    x_data=W_list
    
    i=0

    g_data=[]
    y_data=[]
    for L in L_list:
        y_data.append([])
        g_data.append([])
        for W in W_list:
            for sim in output:
                if sim[0]==L and sim[1]==W:
                    Lambda=1/sim[3]
                    T=np.exp(-2/(Lambda/sim[0]))
    #                y_data[i].append(1/(sim[0]*sim[3]))
                    y_data[i].append(Lambda/sim[0])
                    #g_data[i].append((4/np.cosh((2*sim[0]/Lambda))+1))
                    g_data[i].append(1/np.pi*T/(1-T))
        i+=1

    
    
    
    nu_range=np.linspace(1,2,100)
    W_range=np.linspace(16.4,16.6,10)
    # kappa_range=np.linspace(-.1,0,10)
    
    minerror=10000000
    best_nu=1
    for kappa in [0]:
        for nu in nu_range:
            for W_crit in W_range:
                scaled_xs=[]
                scaled_ys=[]
                for i in range(len(L_list)):
                    scaled_xs.append([])
                    scaled_yj=[g*L_list[i]**(-kappa/nu) for g in g_data[i]]
                    scaled_ys.append(scaled_yj)
                    for j in range(len(x_data)):
                        scaled_xj=(x_data[j]-W_crit)*L_list[i]**(1/nu)
                        scaled_xs[i].append(scaled_xj)
                error=checkfit(scaled_xs,scaled_ys)
                if error<minerror:
                    minerror=error
                    best_nu=nu
                    best_W=W_crit
                    best_kappa=kappa
    #        
    
    
    
    
    print(best_nu)
    print(best_W)
    print(best_kappa)
    
    
    #best_c=.305
    #best_nu=1.16
    scaled_plot_x=[]
    scaled_plot_y=[]
    abs_scaled_x=[]
    #best_nu=1.07
    
    
    for i in range(len(L_list)):
        for j in range(len(x_data)):
            scaled_plot_x.append(((x_data[j]-best_W)/best_W)*L_list[i]**(1/best_nu))
            abs_scaled_x.append(L_list[i]/np.abs(x_data[j]-best_W)**-best_nu)
            scaled_plot_y.append(g_data[i][j]*L_list[i]**(0/best_nu))
    plt.plot(scaled_plot_x,scaled_plot_y,linestyle='None',marker='^')
    #plt.xlabel(r'$tL^{1/\nu}$')
    plt.xlabel(r'$tL^{\nu}$')
    plt.ylabel('g')
    # plt.yscale('log')
    # plt.xscale('log')
    #plt.ylabel(r'$\Lambda/L$')
    
def compute_c_s():
    
    outputfile='offdiag_t30_full.txt'
    
    output=openfile(outputfile)
    
    
    
    y_data=[]
    L_list=[]
    c_list=[]
    W_list=[]
    for sim in output:
        L=sim[0]
        c=sim[2]
        W=sim[1]
        if L not in L_list:
            L_list.append(L)
        if c not in c_list:
            c_list.append(c)
        if W not in W_list:
            W_list.append(W)
    L_list.sort()
    c_list.sort()
    W_list.sort()
    
    x_data=c_list
    
    i=0
    
    g_data=[]
    for L in L_list:
        y_data.append([])
        g_data.append([])
        for c in c_list:
            for sim in output:
                if sim[0]==L and sim[2]==c:
                    Lambda=1/sim[3]
                    T=np.exp(-2/(Lambda/sim[0]))
    #                y_data[i].append(1/(sim[0]*sim[3]))
                    y_data[i].append(Lambda/sim[0])
                    # g_data[i].append((4/np.cosh((2*sim[0]/Lambda))+1))
                    g_data[i].append(1/np.pi*T/(1-T))
        i+=1
    
    
    x_data=c_list
    
    
    for i in range(len(L_list)):
        plt.plot(x_data,g_data[i],label='M= ' + str(L_list[i]))
    plt.xlabel('c')
    plt.ylabel(r'$\lambda_M$/M')
    plt.legend(loc=2)
    plt.show()
    
    nu_range=np.linspace(1.0,1.6,100)
    kappa_range=np.linspace(-2,2,100)
    c_range=np.linspace(.25,.32,20)
    
    
    minerror=10000000
    loss_fcn=[]
    for kappa in [0]:
        for nu in nu_range:
            for c_crit in c_range:
                scaled_xs=[]
                scaled_ys=[]
                for i in range(len(L_list)):
                    scaled_xs.append([])
                    scaled_yj=[g*L_list[i]**(nu/nu) for g in g_data[i]]
                    scaled_ys.append(scaled_yj)
                    for j in range(len(x_data)):
                        scaled_xj=(x_data[j]-c_crit)*L_list[i]**(1/nu)
                        scaled_xs[i].append(scaled_xj)
                
                error=checkfit(scaled_xs,scaled_ys)
                if error<minerror:
                    minerror=error
                    best_nu=nu
                    best_kappa=kappa
                    best_c=c_crit
    #        
    
    
    # for nu in nu_range:
    #     scaled_xs=[]
    #     for i in range(len(L_list)):
    #         scaled_xs.append([])
    #         for j in range(len(x_data)):
    #             scaled_xj=(x_data[j]-best_c)*L_list[i]**(1/nu)
    #             scaled_xs[i].append(scaled_xj)
    #     error=checkfit(scaled_xs,g_data)
    #     loss_fcn.append(error)
        
    # plt.plot(nu_range,loss_fcn)
    # plt.show()
    
    
    
    
    print(best_nu)
    print(best_c)
    print(best_kappa)
    
    # best_c=.307
    # best_nu=1.50
    scaled_plot_x=[]
    scaled_plot_y=[]
    abs_scaled_x=[]
    #best_nu=1.07
    
    for i in range(len(L_list)):
        for j in range(len(x_data)):
            scaled_plot_x.append(((x_data[j]-best_c)/best_c)*L_list[i]**(1/best_nu))
            abs_scaled_x.append(L_list[i]/np.abs(x_data[j]-best_c)**-best_nu)
            scaled_plot_y.append(g_data[i][j]*L_list[i]**(best_nu/best_nu))
    plt.plot(scaled_plot_x,scaled_plot_y,linestyle='None',marker='^')
    #plt.xlabel(r'$tL^{1/\nu}$')
    plt.xlabel(r'$tL^{{1/s}}$')
    plt.ylabel(r'$g$')
    #plt.yscale('log')
    #plt.xscale('log')
    #plt.ylabel(r'$\Lambda/L$')
    
    
    

compute_c_nu()


nu_list=[1.44,1.43,1.22,1.16,1.17,1.15,1.17,1.16,1.22,1.43,1.44]
E_list=[-6.5,-6,-5,-4,-2,0,2,4,5,6,6.5]
plt.plot(E_list,nu_list,marker='*')
plt.ylabel(r'$\nu$')
plt.xlabel('E')
plt.show()