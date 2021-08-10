import os
from datetime import datetime
import random

outputdir='output\\'
datadir="data\\"
newdir=""
count=1

script_dir = os.path.dirname(__file__) 
outputdir = os.path.join(script_dir, 'output\\')

folder_names = filter(os.path.isdir, os.listdir(outputdir))

print(len(os.listdir(outputdir)))

for d in os.listdir(outputdir):
    print(d)
print('a'b)

for name in folder_names:
    print(name)
[print(name) for name in folder_names if name.isnumeric()]

last_number = max([int(name) for name in folder_names if name.isnumeric()])
new_name = str(last_number + 1).zpad(4)
newdir =   sos.mkdir(new_name)

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
    print(newdir)
    #print(curdir)

    # dd/mm/YY H:M:S
    #script_dir = os.path.dirname(__file__) #<-- absolute dir the script is in)

    
    dt_string = now.strftime("%H_%M_%S_")+str(count)
    count += 1
    return( os.path.join(script_dir, outputdir+dt_string))
