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
    rundir=''
    rundir = folder

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

def uniquify(path):
    filename, extension = os.path.splitext(path)
    counter = 1

    while os.path.exists(path):
        path = filename + " (" + str(counter) + ")" + extension
        counter += 1
    print(path)
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

    return(uniquify(os.path.join(rundir, identifier)))

def savecsv(data):
    global runname
    csvfilename=setfilename(setname+'.csv')
    data=[runname] + data

    #data=runname+','+data+''
    with open(csvfilename,'a',newline='', encoding='utf-8') as fd:
        csv_writer=writer(fd)
        csv_writer.writerow(data)

    return data

'''
	coupling_matrix_down: The coupling between the nth and n-1th bars/strips.
	W: Diagonal disorder
	size: the width of the strip or bar. Currently only square bars are supported
	fraction: the fraction of links that are "good", AKA necking fraction
	E: The fermi level, 0 represents the center of the band
'''