# adpated from: https://github.com/nikhil-kotecha/Generating_Music/blob/master/multi_training.py
# with some modifications
import os
import glob
from midi_state_matrix import *
import re
import h5py
import numpy
import random
import argparse

division_len = 16 # interval between possible start locations
num_files = 500

def saveStateMatrices(dirpath, min_time_steps, outpath):
    p = re.compile(r"\.\d{1,2}\.")
    i = 0

    for artist in glob.glob(dirpath+"/*"):
        for song in glob.glob(artist+"/*"):
            # Ignore duplicate songs
            if p.search(song) is not None:
                continue
            
            try:
                statematrix = midiToNoteStateMatrix(song)
            except:
                print("skipping bad file...")
                continue
                
            # Ignore songs that are too short
            if len(statematrix) < min_time_steps:
            	print("skipping short file...")
            	continue

            # Save state matrix as hd5file
            name = os.path.basename(song)
            with h5py.File(outpath + "/" + str(i).zfill(3) + "_" + name[:-4] + ".h5", 'w') as hf:
                hf.create_dataset(name,  data=statematrix)
            i += 1

            # Use only i songs
            if i >= num_files:
                print("Loaded " + str(i) + " midi files")
                return None


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

def parse_arguments():
	parser = argparse.ArgumentParser(description='Creates Note State Matrices \
		and stores them in hd5files')
	parser.add_argument('midi_path', type=str, 
		help='Path to midi files')
	parser.add_argument('-min_time', type=int, default=256, 
		help='Min number of time steps for a song')
	parser.add_argument('-clean_path', type=str, default="../clean-data",
		help='Specify where to store matrix files')
	
	args = parser.parse_args()
	return args


if __name__ == '__main__':
	args = parse_arguments()
	saveStateMatrices(args.midi_path,args.min_time,args.clean_path)

	
