#tO use this config file to output files with particular name to the output folder
#1) cfg.datafilename(   - This must be called as it sets the output file name/set name
#2) cfg.savecsv()       - To store any data, will use the set name and will append
#3) cfg.runfilename(    - This gives you a unique filename, you should pass this a descriptive
#                         file name, ex. 'nu_%s--O_%s-W_%s' and it will not overwrite previous




import os
from datetime import datetime
import random
#import copy

import numpy as np
import scipy as sc
import csv
from csv import writer
from os import makedirs,path

from errno import EEXIST
now = datetime.now()
#default of no run name
runname=''#now.strftime("%H_%M")

scriptdir = os.path.dirname(__file__) 
outputdir = os.path.join(scriptdir, 'output\\')#static ..\output\
datadir= os.path.join(scriptdir, 'data\\')#static ..\data\
setdir=''#current set: ..\output\offdiagE10W10\
rundir=''#current run: \offdiagEW\2_10\
dirdir=''
setname=''
hurdir=''


datafile=''
new_name=''




#important values, E, W, nu

#do you want to crash? name something just 'print'...
def dprint(input):
    if debug:
        print(input)
        
def get_immediate_subdirectories(a_dir):
    return [name for name in os.listdir(a_dir)
            if os.path.isdir(os.path.join(a_dir, name))] 
def datafilename(fn):
    '''
        Give W and E
    '''
    global datafile
    global setdir
    global rundir
    global setname
    setname=fn.removesuffix('.txt')
    #newdirectory()
    #https://stackoverflow.com/questions/7165749/open-file-in-a-relative-location-in-python
    #likely unneeded
    #<-- absolute dir the script is in
    #data_dir=os.path.abspath(os.path.join(script_dir, os.pardir))
    #filename="D:\My Documents\Zimanyi Group\Kinetic Disorder\ZG-FSS\data\" + str(fn)

    setdir =os.path.join(scriptdir,outputdir+setname)
    if os.path.exists(setdir) is not True:
        os.mkdir(setdir)

    rundir=os.path.join(setdir,runname)
    
    if os.path.exists(rundir) is not True:
        os.mkdir(rundir)

    filename = os.path.join(scriptdir, datadir+fn)
    return filename

def setrunfolder(folder):
    global rundir
    
    rundir =os.path.join(scriptdir,outputdir+folder)
    if os.path.exists(rundir) is not True:
        os.mkdir(rundir)


def setfilename(identifier=''):
    global setdir

    #TODO adam ugly
    #tmpiter= os.listdir(outputdir)
    #if len(tmpiter) == 0:
    #    new_name=str(1)
    #else:
    #    last_number = max([int(name) for name in tmpiter if name.isnumeric()])
    #    new_name = str(last_number + 1)
    #print(curdir)
    now = datetime.now()
    #print(curdir)

    # dd/mm/YY H:M:S
    #script_dir = os.path.dirname(__file__) #<-- absolute dir the script is in)
    if identifier == '':
        identifier = now.strftime("%H_%M_%S")
    return(os.path.join(setdir, identifier))

# dd/mm/YY H:M:S
    #script_dir = os.path.dirname(__file__) #<-- absolute dir the script is in)
    #if identifier == '':
   #     identifier = now.strftime("%H_%M_%S_")+str(count)
   #     count += 1

def uniquefile(path):
    fn, ext = os.path.splitext(path)
    i = 1
    while os.path.exists(path):
        path = fn + " (" + str(i) + ")" + ext
        i += 1
    return path

def runfilename(identifier=''):
    #outputfilename(identifier)

    #curdir=newdirectory()
    now = datetime.now()
    #print(curdir)

    # dd/mm/YY H:M:S
    #script_dir = os.path.dirname(__file__) #<-- absolute dir the script is in)
    if identifier == '':
        identifier = now.strftime("%H_%M_%S")

    return(uniquefile(os.path.join(rundir, identifier)))

def savecsv(data,folder=None):
    global rundir
    if folder is None:
        folder = setname

    csvfilename=setfilename(rundir+'.csv')
    #data=[now.strftime("%D %H:%M")] + data

    #data=runname+','+data+''
    with open(csvfilename,'a',newline='', encoding='utf-8') as fd:
        csv_writer=writer(fd)
        csv_writer.writerow(data)

    return data 

def createPlotTitle(W=None, E=None, cost_PP=None, execution_time=None, 
        crit_c=None, nu_val=None, window_offset=None, window_width=None, t_low=None, t_high=None):
        title = ''
        title 
        title = "cpp:%.2f   %s\n\nCc: %s - nu: %f - offset: %.2f - width: %.2f" % (costpp,exet,cc, nuval, window_offset,window_width)
        return title
'''
	coupling_matrix_down: The coupling between the nth and n-1th bars/strips.
	W: Diagonal disorder
	size: the width of the strip or bar. Currently only square bars are supported
	fraction: the fraction of links that are "good", AKA necking fraction
	E: The fermi level, 0 represents the center of the band
'''