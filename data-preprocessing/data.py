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

	
