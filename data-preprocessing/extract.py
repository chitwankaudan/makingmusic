from music21 import *
import glob
import re
import os
import pickle
import argparse

class Extract:
	"""
	Extracts node/chord/duration/rest elements from midi 
	files for the vocabulary approach. 
	"""
	def __init__(self, midi_path, clean_path, num_songs, track, min_length, artists):
		self.complete = set()
		self.midi_path = midi_path
		self.clean_path = clean_path
		self.num_songs = num_songs
		self.track = track
		self.min_length = min_length
		self.notes = []
		self.artists = None

	def extract(self):
		is_looping = True
		# Iterate through every song for every artist
		for artist in glob.glob(self.midi_path+"/*"):

			# If specific artists have been specified, skip artists not in list of artists
			if self.artists is not None: 
				if os.path.basename(artist) not in artists:
					continue 

			for song in glob.glob(artist+"/*"):

				# Check to make sure song hasn't already been parsed (dataset includes duplicates)
				curr_song = os.path.basename(song).split(".")[0]
				if curr_song in self.complete:
					print("Skipping duplicate file...")
					continue

				# Parse midi and split by intsrument
				try:
					print("Trying to parse", curr_song)
					midi = converter.parse(song)
					parts = instrument.partitionByInstrument(midi) 
				except:
					# Skip corrupt files or files with no parts...
					print("Skipping bad file...")
					continue

				# Skip songs with no parts to parse
				if parts is None:
					print("Skipping file with no parts...")
					continue

				# Extract specific track stream
				elements_to_parse = self.getPartStream(parts)
				if elements_to_parse == None:
					# Skip song with no elements to parse
					print("Skipping file with no %s parts..."%(self.track))
					continue

				# Extract notes, chords, rests and their durations
				curr_notes = self.getElements(elements_to_parse)
				self.notes.append(curr_notes)
				self.complete.add(curr_song)

				# Stop if required # of songs have been extracted
				if len(self.complete) >= self.num_songs:
					is_looping = False
					break # break out of inner loop

			if not is_looping:
				break # break out of outer loop

		# Save notes
		with open(self.clean_path +"/"+ self.track + "_notes.pkl", 'wb') as filepath:
			pickle.dump(self.notes, filepath)


	def getPartStream(self,parts):
		"""
		Extract specific track if it meets min_length requirement.
		"""
		elements_to_parse = None
		for i in range(len(parts)):
			name = parts[i].partName
			# Extract part duration (1.0 = one quarter note)
			duration = str(parts[i].duration)
			duration = float(re.findall(" \d*\.?\d+",duration)[0]) 

			if (name==self.track) & (duration>=self.min_length):
				elements_to_parse =  self.transposeKey(parts[i])
				if elements_to_parse != None: 
					elements_to_parse = parts[i].recurse() 
				break # Stop after extracting 1 track from each song
		return elements_to_parse


	def transposeKey(self,part):
		"""
		Transposes all majors keys to Cmajor and minor keys to Aminor
		"""
		# Music21 will try to detect key for the part
		try:
			key = part.analyze('key')
		except: # if music21 can't detect key, skip song
			return None

		#source: https://stackoverflow.com/questions/37494229/music21-transpose-streams-to-a-given-key
		if key.mode == 'major': #transpose all major keys to C
			steps_to_C = interval.Interval(key.tonic, pitch.Pitch('C'))
			sNew = part.transpose(steps_to_C)
			return sNew
		elif key.mode == 'minor': #transpose all minor keys to A
			steps_to_A = interval.Interval(key.tonic, pitch.Pitch('A'))
			sNew = part.transpose(steps_to_A)
			return sNew
		else:
			return None


	def getElements(self,elements_to_parse):
		"""
		Parse notes, chords, rests, and respective their durations
		source: https://github.com/JakeNims1305/DataScienceMusic/blob/master/3LSTMAttLayer-fulldataset-resultsFAILasWAS2LSTMAtt/DataScienceMusic.ipynb
		"""
		curr_notes = []
		for element in elements_to_parse:
			if isinstance(element, note.Note):
				curr_notes.append(str(element.pitch) + " " +  str(element.quarterLength))
			elif isinstance(element, chord.Chord):
				curr_notes.append('.'.join(str(n) for n in element.normalOrder) + " " + str(element.quarterLength))
			elif isinstance(element, note.Rest):
				curr_notes.append(str(element.name)  + " " + str(element.quarterLength))
		return curr_notes


def parse_arguments():
	parser = argparse.ArgumentParser(description='Extracts notes,chords,rests \
		 and durations from midi files and encodes them to input to LSTM.')
	parser.add_argument('midi_path', type=str, 
		help='Path to midi files')
	parser.add_argument('-clean_path', type=str, default="../clean-data-1",
		help='Specify where to store elements list and cleaned data')
	parser.add_argument('-num_songs', type=int, default=100, 
		help='Number of songs to extract')
	parser.add_argument('-track', type=str, default='Piano',
		help='Specify track/instrument to extract from each song \
		(i.e. Voice, Piano, Guitar, etc.)')
	parser.add_argument('-min_length', type=int, default=100, 
		help='Specify min duration of each track (1 = quarternote)')
	parser.add_argument('-artists', type=list, default=None, 
		help='Subset of artists to pull songs from (["ABBA", "TLC"])')
	
	args = parser.parse_args()
	return args


if __name__ == '__main__':
	args = parse_arguments()
	extractor = Extract(args.midi_path,args.clean_path,args.num_songs,
		args.track,args.min_length,args.artists)
	extractor.extract()
		

	




					
			


			