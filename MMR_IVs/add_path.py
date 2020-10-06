import os,sys
filepath = os.path.realpath(__file__)
directory,filename = os.path.split(filepath)
root_path = os.path.dirname(directory)
sys.path.append(root_path) # append root path
