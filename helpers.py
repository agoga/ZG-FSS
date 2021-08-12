import os
from datetime import datetime
import random

script_dir = os.path.dirname(__file__) 
outputdir = os.path.join(script_dir, 'output\\')
datadir="data\\"
newdir=""
 
count=1

#TODO adam ugly
tmpiter= os.listdir(outputdir)
if len(tmpiter) == 0:
    new_name=str(1)
else:
    last_number = max([int(name) for name in tmpiter if name.isnumeric()])
    new_name = str(last_number + 1)
newdir = os.mkdir(os.path.join(script_dir,outputdir+new_name))

def datafilename(fn):
    #newdirectory()
    #https://stackoverflow.com/questions/7165749/open-file-in-a-relative-location-in-python
    #likely unneeded
    #<-- absolute dir the script is in
    #data_dir=os.path.abspath(os.path.join(script_dir, os.pardir))
    #filename="D:\My Documents\Zimanyi Group\Kinetic Disorder\ZG-FSS\data\" + str(fn)
    filename = os.path.join(script_dir, datadir+fn)
    return filename

def outputfilename():
    global count
    global newdir
    #curdir=newdirectory()
    now = datetime.now()
    #print(curdir)

    # dd/mm/YY H:M:S
    #script_dir = os.path.dirname(__file__) #<-- absolute dir the script is in)
    dt_string = now.strftime("%H_%M_%S_")+str(count)
    count += 1
    return( os.path.join(script_dir,outputdir+new_name+"\\"+ dt_string))




#
#import os
#import numpy as np
##path = 'define your directory'
#f='Flag_variable.npy'
#try:
#    Flag_variable = np.load('Flag_variable.npy')
#    Flag_variable = int(Flag_variable)
#    Flag_variable =Flag_variable+ 1
#    np.save(f,Flag_variable)
#except FileNotFoundError:
    
#    np.save(f, 1)
#    Flag_variable=1
    
 
#os.chdir(path)
#Newfolder= 'ID'+ str(Flag_variable)
#os.makedirs(path+Newfolder)
#print('Total Folder created', Flag_variable)


#for i in range (1,11):
#    pass
#     os.chdir(path)
#     Newfolder= 'ID'+ str(i)
#     os.makedirs(path+Newfolder)