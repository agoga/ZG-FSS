import os
from datetime import datetime
import random
#import copy
from collections import defaultdict
import numpy as np
import csv
from os import makedirs,path
import pandas as pd
import matplotlib.pyplot as plt

#\x1b[31m\"red\"\x1b[0m
class bcolors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'

now = datetime.now()
#default of no run name
runname=''#now.strftime("%H_%M")

scriptdir=os.getcwd() #os.path.dirname(__file__) 
datadir= os.path.join(scriptdir, 'data\\')#static ..\data\
outputdir=os.path.join(scriptdir, 'output\\')#static ..\data\


def printV(str,v=True):
    if v:
        print(str)

def openfile(filename):
    f = open(filename, "r+")
    output = defaultdict(list)
    # output['L']=()
    # output['W'].ap
    # output['c'].ap
    # output['lyap']
    # output['std'].
    # output['g'].ap
    for line in f:
        if line:
            strlist = line.split()
            output['date'].append(str(strlist[0][:]))
            output['time'].append(str(strlist[1][:]))
            output['eps'].append(float(strlist[2][1:-1]))
            output['Lz'].append(float(strlist[3][:-1]))
            output['L'].append(float(strlist[4][:-1]))
            output['W'].append(float(strlist[5][:-1]))
            output['tLow'].append(float(strlist[6][:-1]))
            output['c'].append(float(strlist[7][:-1]))
            output['E'].append(float(strlist[8][:-1]))
            output['dim'].append(float(strlist[9][:-1]))
            output['lyap'].append(float(strlist[10][1:]))
            output['std'].append(float(strlist[11]))
            output['g'].append(float(strlist[12][:-1]))
            output['reals'].append(float(strlist[13]))
            if len(strlist) >=15:
                output['runtime'].append(float(strlist[14][:-1]))
            else: 
                output['runtime'].append(float(0))
            output['fname'].append(os.path.basename(filename))
    return output




def stats_L(file,warnc,bigc,verbose=True):
    if file.endswith('.txt'):
        data=openfile(file)
    else:
        data= pd.read_csv(file) 
    
    Lz=np.array(data['Lz'])
    L=np.array(data['L'])
    uniL=np.unique(L)
    #uniLz=np.unique(Lz)
    
    for l in uniL:
        curdata=data[L[:]==l]
        runt=np.array(curdata['runtime'])
        cstr=""

        t=runt/60
        c=np.array(curdata['c'])
        uniC=np.unique(c)
        minC=np.min(c)
        maxC=np.max(c)
        avgR=np.average(t)

        outstr=f"{bcolors.OKBLUE}"+"L= %d"%(l)+f"{bcolors.ENDC}"+", Count=%d, Avg Realizations=%.0f," % (curdata.size,curdata.size/uniC.size)
        outstr+="# of C's=%d, C range=(%.3f,%.3f)" % (uniC.size, np.min(c),np.max(c))
        if avgR > 120:
            outstr+=" Avg Runtime= %.1fh," % (avgR/60)
        else:
            outstr+=" Avg Runtime= %.1fm," % (avgR)

        #outstr+=" Min= %.0fm, Max= %.0fm" % (np.min(t),np.max(t))
        cstr+=""
        
        printV(outstr,verbose)
        
        #print(uniC)
        for ci in uniC:
            count=c[c[:]==ci].size


            
            #if ci < .271:
                #continue
            if count < warnc:
                cstr+= f"{bcolors.FAIL}"
            elif count > bigc:
                cstr+= f"{bcolors.WARNING}"

            cstr+= "%.3f=%d"%(ci,count)
             
            if count < warnc or count > bigc:
                cstr+=f"{bcolors.ENDC}"
            cstr+=", "
        printV(cstr,verbose)
        printV("----------------------------------",verbose)


def stats_C(file,warnL,bigL,verbose=True):
    if file.endswith('.txt'):
        data=openfile(file)
    else:
        data= pd.read_csv(file) 
    
    
    C=np.array(data['c'])
    uniC=np.unique(C)

    
    for c in uniC:
        curdata=data[C[:]==c]
        
        L=np.array(curdata['L'])
        uniL=np.unique(L)

        Lstr=''
        outstr=''
        #c=np.array(curdata['c'])
        


        

        outstr=f"{bcolors.OKBLUE}"+"c= %.5f"%(c)+f"{bcolors.ENDC}"
        outstr+=", # of L's=%d, L range=(%d,%d)" % (uniL.size, np.min(L),np.max(L))
        
        printV(outstr,verbose)
        for li in uniL:
            count=L[L[:]==li].size
        
            #if ci < .271:
                #continue
            if count < warnL:
                Lstr+= f"{bcolors.FAIL}"
            elif count > bigL:
                Lstr+= f"{bcolors.WARNING}"

            Lstr+= "%d=%d"%(li,count)
                
            if count < warnL or count > bigL:
                Lstr+=f"{bcolors.ENDC}"
            Lstr+=", "
        printV(Lstr,verbose)
        printV("----------------------------------",verbose)





def combine(folder,E,W,min_realizations,verbose=True):
    ##
    ##Combines all data for specific parameters into a new csv
    ##

    df=pd.DataFrame()
    minreal=min_realizations
    tstdir=datadir+folder#+'\\E2W10-L10-24''\\E2W12'

    for root, dirs, files in os.walk(tstdir):
        for name in files:
            filepath=os.path.join(root, name)
            bad=False
            badnames=['all','combo','bad','offdiag']
            printV(name,verbose)
            for b in badnames:
                if b in name or name.endswith('.csv'):
                    bad=True

            if bad is True:
                printV('ignoring bad file: ' + name,verbose)
                continue
            if path.isfile(filepath):
                try:
                    data=openfile(filepath)
                except:
                    printV("error "+str(name),verbose)
                    continue

                Lz=np.array(data['Lz'])
                uniLz=np.unique(Lz)
                uniE=np.unique(data['E'])
                uniW=np.unique(data['W'])
                c=np.array(data['c'])
                unic=np.unique(data['c'])
                

                if 100000 in uniLz and E in uniE and W in uniW:
                    printV(str(uniLz) + " " +str(name)+ ' success',verbose)

                    df=df.append(pd.DataFrame.from_dict(data))
                else:
                    printV(str(name)+ ' reject',verbose)

    uniE=np.unique(df['E'])
    uniW=np.unique(df['W'])
    uniLz=np.unique(df['Lz'])


    #now save as csv
    for e in uniE:
        for w in uniW:
            for lz in uniLz:
                
                newfname = f"E{int(e)}W{int(w)}Lz{int(lz/1000)}K"
                printV(newfname,verbose)

                df = df.loc[(df['E']==e) & (df['W']==w) & (df['Lz']==lz) & (df['eps']==1.0)]
                df_fin=pd.DataFrame(columns=df.columns)
                
                
                uniL=np.unique(df['L'])
                
                for L in uniL:
                    dfl=df.loc[(df['L']==L)]
                    #print(dfl)
                    uniC=np.unique(dfl['c'])
                    #print(uniC)
                    for c in uniC:
                        count=len(dfl.loc[(dfl['c']==c)])
                        if count >= minreal:
                            rows=dfl.loc[(dfl['c']==c)]
                            df_fin=pd.concat([df_fin,rows])
                            #print('here')
                            #print(dfw['fname'].unique())
                            #pd.DataFrame(df_fin).to_csv(datadir+newfname+'.csv',mode='w')
                pd.DataFrame(df_fin).to_csv(datadir+newfname+'.csv',mode='w')




    # for l in uniL:
    #     curdata=df[L[:]==l]
    #     outstr=''
    #     cd=np.array(curdata['c'])
    #     uniC=np.unique(cd)

    #     for c in uniC:
    #         outstr+= "(%.3f-%d)"%(c,cd[cd[:]==c].size)
    #     #print(outstr)


def create_shell_script(name,jobname,cpus,params,array='',prio='med2',time='10-00:00:00'):
    with open (name, 'w') as rsh:
        sbnl='\n#SBATCH '
        #!/bin/bash
        #
        #$ -cwd
        #$ -j y
        #$ -S /bin/bash
        rsh.write('#!/bin/bash\n#$ -cwd\n#$ -j y\n#$ -S /bin/bash')
        #SBATCH -c 11
        #SBATCH --time=10-00:00:00
        rsh.write('\n#SBATCH -c %d\n#SBATCH --time=10-00:00:00') % (cpus,time)
        #SBATCH --job-name="jobname"
        rsh.write('\n#SBATCH --job-name="%s"') % (jobname)
        #SBATCH -p prio
        rsh.write('\n#SBATCH -p %s') % (prio)
        #SBATCH --array=10-20:1
        if array.len != 0:
                rsh.write('\n#SBATCH --array=%s')

        
        

        



#SBATCH --array=10-20:1'


