import imageio
import os

filepath='output/offdiagE6W15/temp2/'
outputfilename='E6W15-test2'
outputfile=filepath+outputfilename+'.gif'

files=os.listdir(filepath)

numrepeats=7 #my method of changing framerate
# Build GIF
with imageio.get_writer(outputfile, mode='I') as writer:
    for filename in files:
        image = imageio.imread(filepath+filename)
        for i in range(0,numrepeats):
            writer.append_data(image)
            #print(filename) uncomment if frames seem out of order