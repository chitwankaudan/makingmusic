from music21 import converter, instrument, note, chord, stream
import glob
import re
import os
import pickle
import argparse
import numpy as np
import h5py

class Sequence:
	"""
	Takes output of Extract and creates input and output sequences for training.
	"""
	def __init__(self, clean_path, sequence_length):
		# Get notes (output of extract)
		self.clean_path = clean_path
		self.sequence_length = sequence_length

		with open(self.clean_path+'/notes.pkl', 'rb') as filepath:
			notes = pickle.load(filepath)

		self.notes = notes
		pitchnames = set([elem for song in notes for elem in song])
		self.num_vocab = len(pitchnames)

		# Create note to int maps
		self.note_to_int = dict((note, number) for number, note in enumerate(pitchnames))
		self.int_to_note = dict((value, key) for key, value in self.note_to_int.items())


	def sequence(self):
		X = []
		y = []

		# Create input sequences and the corresponding outputs
		for song in self.notes:
			for i in range(0, len(song) - self.sequence_length, 1):
				sequence_in = song[i:i + self.sequence_length]
				sequence_out = song[i + self.sequence_length]
				X.append([self.note_to_int[elem] for elem in sequence_in])
				y.append(self.note_to_int[sequence_out])

		# Create numpy arrays
		N = len(y)
		X = np.array(X).reshape(N,self.sequence_length,1)
		y = np.array(y)

		# Normalize input
		X = X / float(self.num_vocab)

		# Hot one-encode y
		y = np.eye(self.num_vocab, dtype='uint8')[y]

		# Save dataset
		with h5py.File(self.clean_path + "/data.h5", 'w') as hf:
			hf.create_dataset("X",  data=X)
			hf.create_dataset("y", data=y)

def parse_arguments():
	# dir_path, clean_path, num_songs, track, min_length
	parser = argparse.ArgumentParser(description='Creates X and y \
		 datasets from notes.pkl')
	parser.add_argument('-clean_path', type=str, default="../clean-data-1",
		help='Specify where to store elements list and cleaned data')
	parser.add_argument('-sequence_length', type=int, default=50, 
		help='Length of input sequence')
	args = parser.parse_args()
	return args


if __name__ == '__main__':
	args = parse_arguments()
	sequencer = Sequence(args.clean_path,args.sequence_length)
	sequencer.sequence()
		
