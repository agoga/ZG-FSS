import os
from datetime import datetime
import random
#import copy

import numpy as np
import scipy as sc

from os import makedirs,path

from errno import EEXIST

script_dir = os.path.dirname(__file__) 
outputdir = os.path.join(script_dir, 'output\\')
datadir="data\\"
dirpath="tmp\\"
new_name=""
count=1

#do you want to crash? name something just 'print'...
def dprint(input):
    if debug:
        print(input)
def get_immediate_subdirectories(a_dir):
    return [name for name in os.listdir(a_dir)
            if os.path.isdir(os.path.join(a_dir, name))] 

def outputfilename(identifier=''):
    global count
    global dirpath
    
    #TODO adam ugly
    #tmpiter= os.listdir(outputdir)
    #if len(tmpiter) == 0:
    #    new_name=str(1)
    #else:
    #    last_number = max([int(name) for name in tmpiter if name.isnumeric()])
    #    new_name = str(last_number + 1)

    

    #curdir=newdirectory()
    now = datetime.now()
    #print(curdir)

    # dd/mm/YY H:M:S
    #script_dir = os.path.dirname(__file__) #<-- absolute dir the script is in)
    if identifier == '':
        identifier = now.strftime("%H_%M_%S_")+str(count)
        count += 1
    return(os.path.join(dirpath, identifier))

#remix
raw=[]

'''
	coupling_matrix_down: The coupling between the nth and n-1th bars/strips.
	W: Diagonal disorder
	size: the width of the strip or bar. Currently only square bars are supported
	fraction: the fraction of links that are "good", AKA necking fraction
	E: The fermi level, 0 represents the center of the band
'''

# class analyzedData(raw=[],win,outp):


#     window:[]

#     outputStr=''
#     day:int
#     hour:int

#     #decimalYr:Time
#     label:str


#     def __init__(self, raw=None,window=None,outputPath=None):
#         if raw is not None:
#             super().__init__(copy=raw)


#         if hasattr(self,'mjd') and self.mjd is not None:
#             mjd = self.mjd
#             self.day = int(np.floor(mjd))
#             self.hour = int(str(mjd).split('.')[1])
#             self.decimalYr = Time(mjd,format='mjd')
#             self.decimalYr.format = 'decimalyear'
#         else:
#             self.mjd = None

#         self.outputPath=outputPath
#         self.window = window

#     def raw(self):#for 
#         return self

    
#     #if this is an average then we put it in the day's folder as day_combined_data and save it slightly diff
#     #location that the data should go to
#     def data_path(self):
#         if self.average is False:
#             return self.outputDir()+str(self.hour)+"_data"
#         else:
#             return self.outputDir()+str(self.day)+"_combined_data"

#     #location pdf reports go to
#     def report_path(self):
#         if self.average is False:
#             return  self.outputDir()+str(self.hour)+"_"+"_report.pdf"
#         else:
#             return  self.outputDir()+str(self.day)+"_"+"_combined_report.pdf"

#     #def label(self):
#         #return 

#     def outputDir(self):
#         if self.outputPath is None:
#             return "output/"+self.lin+"/"+str(self.day)+"/"
#         else:
#             return self.outputPath+self.lin+"/"+str(self.day)+"/"

#     def pdfTitle(self):
#         t = ''
# #        if self.bad:
# #            t = 'bad '
#         #t += 'N
#         return t


# #https://stackoverflow.com/questions/11373610/save-matplotlib-file-to-a-directory
# def mkdir_p(mypath):
#     '''Creates a directory. equivalent to using mkdir -p on the command line'''
#     try:
#         makedirs(mypath)
#     except OSError as exc: # Python >2.5
#         if exc.errno == EEXIST and path.isdir(mypath):
#             pass
#         else: raise

        
# def datafilename(fn):
#     #newdirectory()
#     #https://stackoverflow.com/questions/7165749/open-file-in-a-relative-location-in-python
#     #likely unneeded
#     #<-- absolute dir the script is in
#     #data_dir=os.path.abspath(os.path.join(script_dir, os.pardir))
#     #filename="D:\My Documents\Zimanyi Group\Kinetic Disorder\ZG-FSS\data\" + str(fn)

#     global dirpath
#     dirpath =os.path.join(script_dir,outputdir+fn.removesuffix('.txt'))

#     if os.path.exists(dirpath) is not True:
#         os.mkdir(dirpath)

#     filename = os.path.join(script_dir, datadir+fn)
#     return filename


#broken bits

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




#Mikes variables
'''
fs = 18 #font size


#filename="offdiagE6W10.txt"#dataset from localization script, up of box

#A 
#small L and finite size effect
minL = 8#slice off lowest L, tip of the data

#crit_bound_lower, crit_bound_upper = 16.0, 17.0  # critical value bounds
#A looking for transition from lower to upper
crit_bound_lower, crit_bound_upper = 0.62, 0.68 # critical value bounds
nu_bound_lower, nu_bound_upper = 1.05, 1.8  # nu bounds
y_bound_lower, y_bound_upper = -10.0, -0.1  # y bounds
param_bound_lower, param_bound_upper = -10.0, 10.1  # all other bounds

# orders of expansion
n_R = 3
n_I = 1
m_R = 2
m_I = 1



#end shar

window_width = 1.0 #width of window
window_offset = 0.0  #  distance from window center to near edge of window
window_center = 0.74

input = np.array(openfile(filename))

Lrange = np.unique(input[:, 0])
Wrange = np.unique(input[:, 1])
crange = np.unique(input[:, 2])


data = input[:, 0:5]  # L, W, c, LE
data[:, 3] = 1 / (data[:, 0] * data[:, 3])  # L, W, c, normalized localization length

# sort according to L
data = data[np.argsort(data[:, 0])]
#omit L less than minL to control finite size effects
data = data[data[:,0]>=minL]
data = data[np.abs(data[:,2]-window_center)<=window_offset+window_width]
data = data[np.abs(data[:,2]-window_center)>=window_offset]

Lambda = data[:, 3]
L = data[:, 0]
W = data[:, 1]
c = data[:, 2]
sigma = data[:, 4] #uncomment for MacKinnon
# set the driving parameter
Tvar = c

#when L is

numBoot = len(Lambda)//4
numBoot = 1


fig1, (ax1, ax3) = plt.subplots(nrows=1, ncols=2, figsize=(11, 6), sharey=True)


if n_I > 0:
    numParams = (n_I + 1) * (n_R + 1) + m_R + m_I - 1
else:
    numParams = n_R + m_R
print(str(numParams) + "+3 parameters")


if numBoot==1:
    bootSize=1
else:
    bootSize = 0.9 #fraction of data to keep
bootResults=[]
Lambda_restart = Lambda
L_restart = L
Tvar_restart = Tvar
sigma_restart = sigma'''