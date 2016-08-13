


# General Imports
import sys
from sys import stdout
import os
from os import system,path
import fileinput
import shutil
from brian.hears import loadsound
import ctypes

LIB_VTL = ctypes.cdll.LoadLibrary('src/VTL_API/VocalTractLabApi.so')

class VTL_API_class():
	"""
	VTL_API_class
	Basic: Synthesizes wav files from .speaker and .gesture files (and parameters / gestures fed in from higher up).

	-> The module aims to be broadly applicable. (in ambient_speech as well as in the babblin process "learn").
	Either directly execute the right function, or simply execute main, which will call the right function.
	(Main is at the end of the script.

	Audiofiles can be produced in Vocal Tract Lab through a speaker file and a gesture file.
	
		Speaker files: 	Store Anatomical parameters, but also various shapes (look at a '.speaker' file for an example)
						'Input' shape means, that we're using parameters which we don't yet classify as a phoneme.
						Then, there are various syllable shapes (e.g. 'a','e', ..).
						So which one of the shapes does VTL take and produce the sound with? How long should this take?
						Do we want more than one shape after each other? All these things are fed into the gesture file.
					
		Gesture Files: 	Tell VTL which shape (in the speaker file) is to be used for the simulation of the air flow through
						the anatomy of the speaker (anatomy taken from the speaker file too, e.g. length of vocal folds etc..).
						Gesture files are used quite simply by Murakami and Zurbuchen, while learning vowels.
	
	VTL needs both a speaker file and a gesture file.

	What does this module really do? The aim was to have a script that would call VocalTractLabApi.so in various ways. The user
	can either use an adult/infant standard speaker, or a speaker from a speaker-group 'groupname+space+number_in_that_group', e.g.
	"srange 0", which is the first speaker (of age 0, male) in my speaker group (Philip Zurbuchen).

	The user can also chose whether he wants to synthesize an already present (e.g. vowel) shape in the speaker file, of if he wants to
	feed in shape-parameters (that are then saved in the 'input' shape in the speaker file + the gesture file knows that it has to
	execute 'input' and not 'a' for example).

	'main' (which is generally called) sees if the user is entering parameters or something like 'a',
	if he entered 'infant' as speaker or a speaker from a speaker group ('srangefm 0' i.e.).
	Then, the right function is called, which in turn call the VocalTractLabApi.so to synthesize a sound.

	Author:
	Philip Zurbuchen, 2016
	"""
	
	def __init__(self):
		"""
		Define working header (needed in all functions)
		"""
		#self.WAV_HEADER =  ('RIFF'+chr(0x8C)+chr(0x87)+chr(0x00)+chr(0x00)+'WAVEfmt'+chr(0x20)+
		#					chr(0x10)+chr(0x00)+chr(0x00)+chr(0x00)+chr(0x01)+chr(0x00)+chr(0x01)+
		#					chr(0x00)+'"V'+chr(0x00)+chr(0x00)+'D'+chr(0xAC)+chr(0x00)+chr(0x00)+
		#					chr(0x02)+chr(0x00)+chr(0x10)+chr(0x00)+'data'                         )
		self.WAV_HEADER = 'RIFF'+chr(0x8C)+chr(0x87)+chr(0x00)+chr(0x00)+'WAVEfmt'+chr(0x20)+chr(0x10)+chr(0x00)+chr(0x00)+chr(0x00)+chr(0x01)+chr(0x00)+chr(0x01)+chr(0x00)+'"V'+chr(0x00)+chr(0x00)+'D'+chr(0xAC)+chr(0x00)+chr(0x00)+chr(0x02)+chr(0x00)+chr(0x10)+chr(0x00)+'data'
		
		
		

	def main(self,input_dict,paths):
		""" 
		Main function
		"""
		
		# Interpret input (see below)
		self.extract_input(input_dict,paths)
		
		# Speaker
		# -------------------------------------------------------------------------------------------------------
		#	Get speaker paths
		if self.group_speaker:
			# Split (for example "srangefm 0" > "srange","0")
			self.group,self.speaker = self.group_speaker.split()
			
			# Get the speaker path used in GesToWav (VTL)
			self.speaker_path = 'data/speakers_gestures/%s/%s.speaker'%(self.group,self.speaker)
			
		elif self.standard_speaker:
			#either adult or infant speaker paths
			self.speaker = self.standard_speaker
			self.speaker_path = 'data/speakers_gestures/standard/%s.speaker'%self.speaker
			
			
		else:
			raise RuntimeError('Invalid use of api_class. Enter an input path or chose a standard or group speaker. \n Then make sure you have your speaker in the right folder..')
		
		# Get output name (for wav and area file name)
		output_name = self.speaker + '_' + self.simulation_name + '_' + str(self.rank)
		

		#	If parameter-based production, change speaker ('<input>'-shape)
		if type(self.params) == list:
			# Now, this is a bit tricky. If the user fed in parameters, that means the speaker
			# file needs to be changed. If no gesture name was entered by the user, gesture is
			# auto-set to 'input' in 'extract_input'. If, however, the user entered parameters
			# AND a gesture, those parameters are put into that specific shape place in the speaker
			# file.
			self.par_to_speaker()
			
			
			
		
		
		# Gesture
		# -------------------------------------------------------------------------------------------------------
		# 	Get gesture paths
		if self.group_speaker:
			self.gesture_path = 'data/speakers_gestures/%s/%s_%s.ges'%(self.group,self.speaker,self.gesture)
		elif self.standard_speaker:
			self.gesture_path = 'data/speakers_gestures/standard/%s.ges'%(self.standard_speaker)
		else:
			raise RuntimeError('No speaker/gesture (input) folder!)')
		
		
		
		#	Create that gesture if not already existing. self.gesture is either something like 'a', or 'o', or 'input' (see __init__())
		self.create_gesture(overwrite=False)
		
		
		# Output
		# -------------------------------------------------------------------------------------------------------
		if not self.wav_path: # If user hasn't already entered a wav path.
			self.wav_path = self.wav_folder+'/'+output_name+'.wav'
		self.area_path = self.wav_path[:-3]+'txt'
		# Create Output directory if not yet there.
		if not path.isdir(self.wav_folder):
				system('mkdir --parents '+self.wav_folder)
		
		
		if self.verbose:
			print "Calling vtlGesToWav to synthesize.."
		
		# SYNTHESIZE
		# Run through Vocal Tract Lab, feeding in all the paths to the right speaker and gesture files.
		#	(Call LIB_VTL) in src/VTL_API/VocalTractLabApi.so
		# -------------------------------------------------------------------------------------------------------
		
		LIB_VTL.vtlGesToWav(self.speaker_path,self.gesture_path,self.wav_path,self.area_path)#self.gesture_path,self.wav_path,self.area_path
		
		
		# Another option: Call in a transfer script called 'link.py', in the same folder as api_class.py
		# The script would have to look like this:
		"""
		#!/usr/bin/python
		import sys # For argument passing
		import ctypes
		LIB_VTL = ctypes.cdll.LoadLibrary('src/VTL_API/VocalTractLabApi.so')
		LIB_VTL.vtlGesToWav(sys.argv[1],sys.argv[2],sys.argv[3],sys.argv[4])#self.gesture_path,self.wav_path,self.area_path
		"""
		# 	.. and be called like this (in this script).
		"""
		system('python src/VTL_API/link.py \"%s\" \"%s\" \"%s\" \"%s\"'%(self.speaker_path,self.gesture_path,self.wav_path,self.area_path))
		"""
		# This is an option, if the user notices, that VTL seems to take longer and longer to synthesize.. 
		
		
		if self.verbose:
			print "Sound produced, processing.."
	
		# Repair header of wav File
		# -------------------------------------------------------------------------------------------------------
		with open(self.wav_path, 'rb') as file_:
			content = file_.read()
			
		with open(self.wav_path, 'wb') as newfile:
			newcontent = self.WAV_HEADER + content[68:]
			newfile.write(newcontent)

		
		# 1.Correct initial sound
		# 2.Check if not dead sound (which happens when parameters are such, that airflow is cut off in the vocal tract)
		# -------------------------------------------------------------------------------------------------------
		self.sound_is_valid = self.correct_and_check(self.wav_path)
		
		
		if self.verbose:
			print "Sound processing completed!"
			
		# Play sound?
		if self.verbose:
			system('aplay '+self.wav_path)
		
	
	
	
	
	
	
	# Functions called from "main"
	# ----------------------------------------------------------------------------------------------------------------------------------
	# ----------------------------------------------------------------------------------------------------------------------------------
	
	def extract_input(self,input_dict,paths):
		# Check input dictionary.
		# ----------------------------------------------------------------------------------------
			# Synth from parameters?
		self.params = input_dict['params'] if 'params' in input_dict else False
		
			# Synth a specific gesture?
		self.gesture = input_dict['gesture'] if 'gesture' in input_dict else 'input'
		
			# Use a standard speaker (adult/infant)
		self.standard_speaker = input_dict['standard_speaker'] if 'standard_speaker' in input_dict else False
		
			# Use a speaker from a group (format: 'groupname'+' '+'number in that group')
		self.group_speaker = input_dict['group_speaker'] if 'group_speaker' in input_dict else False
		
			# Use a learning speaker (format: 'groupname'+' '+'number in that group')
		self.learning_speaker = input_dict['learning_speaker'] if 'learning_speaker' in input_dict else False
		
			# Special simulation name?
		self.simulation_name = input_dict['simulation_name'] if 'simulation_name' in input_dict else ''
		
			# How much above standard (52) pitch should the sound be?
		self.pitch_var = input_dict['pitch_var'] if 'pitch_var' in input_dict else 0
		
			# Length of sound?
		self.len_var = input_dict['len_var'] if 'len_var' in input_dict else 1
		
			# Rank of worker?
		self.rank = input_dict['rank'] if 'rank' in input_dict else 0
		
			# Verbose output? (Print statements?)
		self.verbose = input_dict['verbose'] if 'verbose' in input_dict else False
		# ----------------------------------------------------------------------------------------
		
		# Check input path dictionary
		# ----------------------------------------------------------------------------------------
		# Path of speakers and gestures.
		if 'input_path' in paths: # input path? (without '.speaker' at the end)
			self.input_path = paths['input_path']
		
		# Which folder to save the wav file into?
		if 'wav_folder' in paths:
			self.wav_folder = paths['wav_folder']
		else:
			raise RuntimeError('No wav_folder! (for area file!)')
		
		# Save directly to this path?
		self.wav_path = paths['wav_path'] if 'wav_path' in paths else False
		# ----------------------------------------------------------------------------------------
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	def get_dic(self,p,shape_name):
		"""
		Convert input parameters to dictionary
		"""
		# Make strings out of params
		ps = []
		for i in range(len(p)):
			ps.append("%.4f" % p[i])
		
		
		table = {	'HX':ps[0], 'HY':ps[1], 'JA':ps[2], 'LP':ps[3], 'LD':ps[4], 'VS':ps[5], 'TCX':ps[6], 
					'TCY':ps[7], 'TTX':ps[8], 'TTY':ps[9], 'TBX':ps[10], 'TBY':ps[11], 'TS1':ps[12], 'TS2':ps[13], 
					'TS3':ps[14], 'TS4':ps[15], 'SHAPE':shape_name}
		return table









	
	def correct_and_check(self,path):
		"""
		The VTL-Produced sounds often come with short bursts right at the onset of the sound. This little function removes such noises as they interfere with learning.
		
		Sometimes we don't have any sound at all. Give that info as output to the main function! passed: "did it pass the test?"
		"""
		
		sound = loadsound(path) 			   #loadsound from brianhears
		low = 249                              #duration of initial silence
		sound[0:low] = 0
		sound.save(path)
		
		
		# Look at the sound at different places: Will they add up to a reasonable number?
		mean = 0
		
		for here in range(4000,12001,200):
			mean += abs(sound[here][0])
			
		valid = mean>0.2
		
		#from matplotlib import pyplot as plt
		#plt.plot(sound)
		#plt.show()
		
		return valid
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	def par_to_speaker(self):
		"""
		
		The problem with user-made speakers (created in VTL) of various anatomies (/ages): We need a way to modify these speaker files without rewriting the whole file. 
		That is why this script was created. We'll go into an already existing .speaker file and change the vocal parameters, in order to optain (for example) a sample 'o' 
		when we execute synthesize_wav.py for a speaker that is neither of the predefined 'adult' or 'infant' speakers.
		"""
	
		# Compose insertion
		# --------------------------------------------------------------------------------------------
		
		dic = self.get_dic(self.params,self.gesture) # Used for the following insertion into the speaker file

		insertion = """      <shape name="{SHAPE}">
        <param name="HX" value="{HX}" domi="0.0"/>
        <param name="HY" value="{HY}" domi="0.0"/>
        <param name="JX" value="-0.2000" domi="0.0"/>
        <param name="JA" value="{JA}" domi="0.0"/>
        <param name="LP" value="{LP}" domi="0.0"/>
        <param name="LD" value="{LD}" domi="0.0"/>
        <param name="VS" value="{VS}" domi="0.0"/>
        <param name="VO" value="-0.1000" domi="0.0"/>
        <param name="WC" value="0.0000" domi="0.0"/>
        <param name="TCX" value="{TCX}" domi="0.0"/>
        <param name="TCY" value="{TCY}" domi="0.0"/>
        <param name="TTX" value="{TTX}" domi="0.0"/>
        <param name="TTY" value="{TTY}" domi="0.0"/>
        <param name="TBX" value="{TBX}" domi="0.0"/>
        <param name="TBY" value="{TBY}" domi="0.0"/>
        <param name="TRX" value="-1.8530" domi="0.0"/>
        <param name="TRY" value="-1.7267" domi="0.0"/>
        <param name="TS1" value="{TS1}" domi="0.0"/>
        <param name="TS2" value="{TS2}" domi="0.0"/>
        <param name="TS3" value="{TS3}" domi="0.0"/>
        <param name="TS4" value="{TS4}" domi="0.0"/>
        <param name="MA1" value="0.0000" domi="0.0"/>
        <param name="MA2" value="0.0000" domi="0.0"/>
        <param name="MA3" value="0.0000" domi="0.0"/>\n      </shape>\n""".format(**dic)
		
		
		# Put the insertion in the right place
		# --------------------------------------------------------------------------------------------
		
		
		
		with open(self.speaker_path,'r+') as f:
			lines = f.readlines()
			
			searchstring = '<shape name="%s">'%self.gesture
		
			try:
				startline = [line for line in lines if searchstring in line][0]
				found_shape = True
			except IndexError:
				found_shape = False
		
			if found_shape:
				start = lines.index(startline)
				end = start + 25
				#if the shape is already contained in the file, we'll have to delete it first
				del lines[start:end+1]
			else:
				line_bfore = [line for line in lines if '</anatomy>' in line][0]
				start = lines.index(line_bfore) + 2
			
			lines.insert(start,insertion) # Insert before start
			"""
			# Get rid of empty lines at the end of the speaker file..
			# --------------------------------------------------------------
			while True:
				line = lines[-1].strip()
				if line == '':
					lines.pop()
				else:
					break
			"""
			f.seek(0)
			f.write("".join(lines)) # Overwrite with the (now joined) stringlist
			f.truncate()
			
			if int(len(lines)/600) > 1:
				raise RuntimeError("Speaker files getting too long! Read/write problem, which I thought I had solved..")
	
	
	
	
	
	def create_gesture(self,overwrite=False):
		"""
		As for now, we only look at syllables as our gestures. This function simply produces a gesture file for gesture = syllable.
		main function takes general speaker/gesture path, syllable to be gestured (gesture), used speaker, pitch perturbation,
		and duration factor; outputs resulting file name.
		"""
	
		# Check if we want to produce a file ( / overwrite the last file?)
		# ------------------------------------------------------------------------
		if path.exists(self.gesture_path) and not overwrite:
			if self.verbose:
				print "Using already-existing gesture file."
			return None
		else:
			if path.exists(self.gesture_path):			
				system('rm '+self.gesture_path)             # delete previous version of gesture score (if needed)
				system('touch '+self.gesture_path)          # create empty gesture score file
			if self.verbose:
				print 'Creating gesture file '+self.gesture

		

		#print 'opening gesture file'
		file = open(self.gesture_path, 'w')			# access gesture score file for writing

		#print 'writing gestural score'
		file.write('<gestural_score>\n')		# from here on: write gesture score -> vowel gestures



		########################
		# Vowel gestures
		########################
		duration_s = str(0.661085 * self.len_var)
		time_constant_s = str(0.015 * self.len_var)

		file.write('  <gesture_sequence type="vowel-gestures" unit="">\n')
							# this accesses the modified versions of the vowel to be learned from within the modified speaker file
		file.write('    <gesture value="'+self.gesture+'" slope="0.000000" duration_s="'+duration_s+'" time_constant_s="'+time_constant_s+'" neutral="0" />\n')
		file.write('  </gesture_sequence>\n')

		########################
		# Lip gestures
		########################
		file.write('  <gesture_sequence type="lip-gestures" unit="">\n')

		#!! "consonnant" not defined in this script. As a remedy; inserted the following lines
		try:
			consonant
		except NameError:
			consonant = None

		if consonant == 'm':
			file.write('    <gesture value="ll-labial-nas" slope="0.000000" duration_s="0.208372" time_constant_s="0.015000" neutral="0" />\n')
			file.write('    <gesture value="" slope="0.000000" duration_s="0.090543" time_constant_s="0.015000" neutral="1" />\n')
			file.write('    <gesture value="ll-labial-nas" slope="0.000000" duration_s="0.107907" time_constant_s="0.015000" neutral="0" />\n')
			file.write('    <gesture value="" slope="0.000000" duration_s="0.347287" time_constant_s="0.015000" neutral="1" />\n')
		file.write('  </gesture_sequence>\n')

		########################
		# Tongue tip gestures
		########################
		file.write('  <gesture_sequence type="tongue-tip-gestures" unit="">\n')

		file.write('  </gesture_sequence>\n')

		########################
		# Tongue body gestures
		########################
		file.write('  <gesture_sequence type="tongue-body-gestures" unit="">\n')

		file.write('  </gesture_sequence>\n')

		########################
		# Velic gestures
		########################
		file.write('  <gesture_sequence type="velic-gestures" unit="">\n')

		file.write('  </gesture_sequence>\n')

		########################
		# Glottal shape gestures
		########################

		time_constant_glottal = str(0.02 * self.len_var)

		file.write('  <gesture_sequence type="glottal-shape-gestures" unit="">\n')
		file.write('    gesture value="modal" slope="0.000000" duration_s="'+duration_s+'" time_constant_s="'+time_constant_glottal+'" neutral="0" />\n')
		file.write('  </gesture_sequence>\n')

		########################
		# F0 gestures
		########################


		durations = [str(0.084341*self.len_var), str(0.179845*self.len_var), str(0.235659*self.len_var)]
		time_constants = [str(0.01515*self.len_var), str(0.0099*self.len_var), str(0.0078*self.len_var)]
		slopes = [str(0.0/self.len_var), str(9.28/self.len_var), str(-27.68/self.len_var)]

		file.write('  <gesture_sequence type="f0-gestures" unit="st">\n')
	
		if self.speaker == 'adult':

			pitch = [str(32.0), str(34.0), str(28.0)]

			file.write('    <gesture value="'+pitch[0]+'00000" slope="'+slopes[0]+'" duration_s="'+durations[0]+'" time_constant_s="'+time_constants[0]+'" neutral="0" />\n')
			file.write('    <gesture value="'+pitch[1]+'00000" slope="'+slopes[1]+'" duration_s="'+durations[1]+'" time_constant_s="'+time_constants[1]+'" neutral="0" />\n')
			file.write('    <gesture value="'+pitch[2]+'00000" slope="'+slopes[2]+'" duration_s="'+durations[2]+'" time_constant_s="'+time_constants[2]+'" neutral="0" />\n')
		
		elif self.speaker == 'infant':
		
			pitch = [str(52.0+self.pitch_var), str(54.0+self.pitch_var), str(48.0+self.pitch_var)]

			file.write('    <gesture value="'+pitch[0]+'00000" slope="'+slopes[0]+'" duration_s="'+durations[0]+'" time_constant_s="'+time_constants[0]+'" neutral="0" />\n')
			file.write('    <gesture value="'+pitch[1]+'00000" slope="'+slopes[1]+'" duration_s="'+durations[1]+'" time_constant_s="'+time_constants[1]+'" neutral="0" />\n')
			file.write('    <gesture value="'+pitch[2]+'00000" slope="'+slopes[2]+'" duration_s="'+durations[2]+'" time_constant_s="'+time_constants[2]+'" neutral="0" />\n')

		else:                      # use predefined infant speaker pitch + self.pitch_var for other speakers.
	
			pitch = [str(52.0+self.pitch_var), str(54.0+self.pitch_var), str(48.0+self.pitch_var)]

			file.write('    <gesture value="'+pitch[0]+'00000" slope="'+slopes[0]+'" duration_s="'+durations[0]+'" time_constant_s="'+time_constants[0]+'" neutral="0" />\n')
			file.write('    <gesture value="'+pitch[1]+'00000" slope="'+slopes[1]+'" duration_s="'+durations[1]+'" time_constant_s="'+time_constants[1]+'" neutral="0" />\n')
			file.write('    <gesture value="'+pitch[2]+'00000" slope="'+slopes[2]+'" duration_s="'+durations[2]+'" time_constant_s="'+time_constants[2]+'" neutral="0" />\n')


		file.write('  </gesture_sequence>\n')

		#--------------------------------------------------------------------------------------------------------------------------------------------------------------------
		#--------------------------------------------------------------------------------------------------------------------------------------------------------------------
		#--------------------------------------------------------------------------------------------------------------------------------------------------------------------



		########################
		# Lung pressure gestures
		########################

		durations_lung = [str(0.01*self.len_var), str(0.528295*self.len_var), str(0.1*self.len_var)]
		time_constants_lung = [str(0.005*self.len_var), str(0.005*self.len_var), str(0.005*self.len_var)]

		file.write('  <gesture_sequence type="lung-pressure-gestures" unit="Pa">\n')
		file.write('    <gesture value="0.000000" slope="0.000000" duration_s="0.050000" time_constant_s="0.050000" neutral="0" />\n')
		file.write('    <gesture value="0.000000" slope="0.000000" duration_s="'+durations_lung[0]+'" time_constant_s="'+time_constants_lung[0]+'" neutral="0" />\n')
		file.write('    <gesture value="1000.000000" slope="0.000000" duration_s="'+durations_lung[1]+'" time_constant_s="'+time_constants_lung[1]+'" neutral="0" />\n')
		file.write('    <gesture value="0.000000" slope="0.000000" duration_s="'+durations_lung[2]+'" time_constant_s="'+time_constants_lung[2]+'" neutral="0" />\n')
		file.write('  </gesture_sequence>\n')

		file.write('</gestural_score>')



		#print 'closing gesture file'
		file.close()







	
