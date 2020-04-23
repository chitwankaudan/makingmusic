import os
import glob
import re
import h5py
import numpy as np
import random

division_len = 16 # interval between possible start locations
num_files = 500

def getSegment(filename, num_time_steps):
    # Open state matrix file
    with h5py.File(filename, 'r') as hf:
        matrix = hf[hf.keys()[0]][:]

    # Pick a random starting point, and collect num_time_steps length sample
    start = random.randrange(0,len(matrix)-num_time_steps,division_len)
    seg = matrix[start:start+num_time_steps]

    return seg

def getBatch(start_index, batch_size, num_time_steps, datadir="../clean-data"):
    if start_index + batch_size > num_files:
        end_index = num_files
    else:
        end_index = start_index+batch_size
    
    batch_files = [glob.glob(datadir+"/%s*"%(str(i).zfill(3)))[0] for i in range(start_index,end_index)]
    #print("Using the following files:", batch_files)
    return [getSegment(file,num_time_steps) for file in batch_files]