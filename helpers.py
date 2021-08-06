import os

def datafilename(fn):
    
    #https://stackoverflow.com/questions/7165749/open-file-in-a-relative-location-in-python
    #likely unneeded
    script_dir = os.path.dirname(__file__) #<-- absolute dir the script is in
    print(script_dir)
    #data_dir=os.path.abspath(os.path.join(script_dir, os.pardir))
    #filename="D:\My Documents\Zimanyi Group\Kinetic Disorder\ZG-FSS\data\" + str(fn)
    filename = os.path.join(script_dir, "data\\"+fn)
    return filename