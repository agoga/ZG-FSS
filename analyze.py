#NOT CURRENTLY WORKING - my attempt to create a toolbox to analyze data ON farm instead of bringing it local
import os
import argparse
from pathlib import Path
from collections import defaultdict
import numpy as np
import csv
from os import makedirs,path

from errno import EEXIST


def openfile(filename):
    f = open(filename, "r")
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
            output['runtime'].append(float(strlist[14][:-1]))
    return output

def stats(file):
    print(file)
    data=openfile(datadir+file)
    Lz=np.array(data['Lz'])
    L=np.array(data['L'])
    c=np.array(data['c'])
    rt=np.array(data['runtime'])
    uniL=np.unique(L)
    uniLz=np.unique(Lz)

    print(uniLz)
    for l in uniL:
        vals=rt[L[:]==l]
        t=rt[L[:]==l]/60
        print("L= %d, Count=%d, Avg Runtime= %.1fm, Min= %.0fm, Max= %.0fm" % (l , vals.size, np.average(t),np.min(t),np.max(t)))


if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("file_path", type=Path)
    p = parser.parse_args()

    stats(p.file_path)



    import os
from datetime import datetime
import random
#import copy
from collections import defaultdict
import numpy as np
import csv
from os import makedirs,path
import pandas as pd


now = datetime.now()
#default of no run name
runname=''#now.strftime("%H_%M")

scriptdir=os.getcwd() #os.path.dirname(__file__) 
datadir= os.path.join(scriptdir, 'data\\')#static ..\data\

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
            output['runtime'].append(float(strlist[14][:-1]))
    return output

def stats(file):
    data=openfile(file)
    Lz=np.array(data['Lz'])
    L=np.array(data['L'])
    c=np.array(data['c'])
    rt=np.array(data['runtime'])
    uniL=np.unique(L)
    uniLz=np.unique(Lz)

    print(uniLz)
    for l in uniL:
        vals=rt[L[:]==l]
        t=rt[L[:]==l]/60
        print("L= %d, Count=%d, Avg Runtime= %.1fm, Min= %.0fm, Max= %.0fm" % (l , vals.size, np.average(t),np.min(t),np.max(t)))
       print("\n")

def combine(fl,name):
    df=pd.DataFrame()
    E=2
    W=10
    count=0
    tstdir=datadir#+'\\E2W10-L10-24\\'
    verbose=False

    for root, dirs, files in os.walk(tstdir):
            for name in files:
                filepath=os.path.join(root, name)
                bad=False
                badnames=['all','combo','bad','offdiag']

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
                        #print(unic) #       else:
                        #print(str(uniLz) + " " +str(filename)+ ' fail')
                    # L=np.array(data['L'])
                    # c=np.array(data['c'])
                    # rt=np.array(data['runtime'])
                    # uniL=np.unique(L)

    uniE=np.unique(df['E'])
    uniW=np.unique(df['W'])
    uniLz=np.unique(df['Lz'])

    #now save as csv
    for e in uniE:
        for w in uniW:
            for lz in uniLz:
                newfname = f"E{int(e)}W{int(w)}Lz{int(lz/1000)}K"
                printV(newfname,verbose)

                df_write = df.loc[(df['E']==e) & (df['W']==w) & (df['Lz']==lz) & (df['eps']==1.0)]
                print(df_write['fname'].unique())
                pd.DataFrame(df_write).to_csv(tstdir+newfname+'.csv',mode='w')


def oldbadcombine(fl,name):
    dl=[]
    for f in fl:
        df=openfile(f)
        dl.append(df)
    #data=[now.strftime("%D %H:%M")] + data


    file_exists = path.isfile(name)
    print(file_exists)
    with open(name,'a',newline='', encoding='utf-8') as fd:
        w=csv.DictWriter(fd,dl[0].keys())

        if not file_exists:
            w.writeheader()
            
        w.writerows(dl)