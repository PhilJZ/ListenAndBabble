#  FUNCTIONS CALLED BY "setup_ambient_speech"
#  ------------------------------------------------------------------------------------------------


#	
#   8. Jan 2016 - Philip Zurbuchen
#  ------------------------------------------------------------------------------------------------




# Class imports
# -------------------------------------------------------
	# Import the class containing the parameters and arguments for this function.
from parameters.get_params import parameters as params
	# Import the class containing Vocal Tract Lab API (for synthesizing speaker-gesture sounds)
from src.VTL_API.api_class import VTL_API_class
synthesize = VTL_API_class() #"synthesize is-a VTL_API_class"



# General Imports
# ---------------------------------------------------------
from os import system,path,listdir,getcwd    		# for filehandling and exec-documentation
import sys                                          #
import pickle										# for saving the class in a file later
import random
import numpy
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import fileinput
import runpy
	#Importing modules used to process sound
import gzip
from brian import kHz, Hz
from brian.hears import Sound, erbspace, loadsound, DRNL
from scipy.signal import resample
from subprocess import call

# Python debugger
from pdb import set_trace as debug

class functions(params):
	"""
	Provides functions, called in "src/ambient_speech/ambient_speech.py"
	Sets up VTL-speakers (in a speaker group) to be ready to be used for hearing and babbling.
	"""
	
	# Since setting up ambient speech is the first step in our project,
	# no other class instances are imported in the beginning.. (compare
	# with 'hear' and 'learn', where we import all methods and variables
	# which are defined / computed here in ambient_speech.
	
	def __init__(self):
		"""
		Initialize from params:
		Import all relevant parameters as self. (from 'params.variable' to 'self.variable')
		"""
		params.__init__(self)
		self.get_ambient_speech_params()
		
		
		#Initialize group specific parameters
		self.n_speakers = int(self.size) if self.speakers=="all" else len(self.speakers)#size of the group (how many tot. speakers)
		
		self.age_m = []								#age males
		self.age_f = []								#age females
		
		self.gender = []											#gender of each file ( e.g. ['male','female',...])
		
		#The speakers in self.speakers that are male / female (computed in read_age)
		self.male_speakers = []								#textfile Number male
		self.female_speakers = [] 								#textfile Number female
		
		self.f0vals_m = numpy.real(numpy.array(self.f0s))    	#parameters of the fit for the f0 sequence
		self.f0vals_f = numpy.imag(numpy.array(self.f0s))
		
		self.par_names = {0:'HX', 1:'HY', 2:'JA', 3:'LP', 4:'LD', 5:'VS', 6:'TCX', 7:'TCY', 8:'TTX', 9:'TTY', 10:'TBX', 11:'TBY', 12:'TS1', 13:'TS2', 14:'TS3', 15:'TS4'}
		
		if self.vowels == "all":
			self.vowels = ["a","e","i","o","u"]
		if self.speakers == "all":
			self.speakers = numpy.array(range(22))
			
		self.plot_colors = {"a":"m", "e":"c", "i":"y", "o":"g", "u":"b", "@":"k", "null":"k"} #magenta,cyan,yellow,green,blue,black
			
		self.vowels_with_schwa = self.vowels[:]
		self.vowels_with_schwa.append('@')
		self.vowels_with_schwa_phonetic = []
		for i in range(len(self.vowels_with_schwa)):
			self.vowels_with_schwa_phonetic.append('/'+self.vowels_with_schwa[i]+'/')
			
		# Initialize as dicts. Equipped with empty numpy arrays a bit later.
		self.male_formants = dict()
		self.female_formants = dict()
		self.samp_formants = dict()
		
		#Initialize paths
		self.current_path = path.dirname(path.abspath(__file__)) #Get path of this script
		self.base_path = getcwd()
		self.speaker_path = self.base_path+'/data/speakers_gestures/'+self.sp_group_name
		#self.gesture_path = self.base_path+'/data/speakers_gestures/ambient_speech/'+self.sp_group_name
			# output_path stands for the path where the vowel-prototypes are saved.
		self.output_path = self.base_path+'/data/output/ambient_speech/'+self.sp_group_name+'/prototypes'
		self.output_paths = dict()
		self.output_path_samples = self.base_path+'/data/output/ambient_speech/'+self.sp_group_name+'/samples'
		self.temp_samp_path = self.output_path_samples+'/temp'
		self.result_path = self.base_path+'/results/ambient_speech/'+self.sp_group_name
		
		# Initialize with empty numpy arrays
		for vowel in self.vowels_with_schwa:
			self.male_formants[vowel] = numpy.array([])
			self.female_formants[vowel] = numpy.array([])
			self.samp_formants[vowel] = numpy.array([])
			# Save right output paths for prototype vowel sounds. These are used in 'hear' to test the ESN classification. 
			# (These prototypes should give very clear class distinctions, when classified by the ESN.) Prototypes made in 
			# create_prototype_wav.
			self.output_paths[vowel] = [('','') for i in range(22)] # first string: .wav, second: .dat.gz
			for speaker in self.speakers:
				self.output_paths[vowel][speaker] = (	self.output_path+'/'+vowel+'/'+str(speaker)+'__'+str(self.rank)+'.wav',
														self.output_path+'/'+vowel+'/'+str(speaker)+'__'+str(self.rank)+'.dat.gz')

			
#SETUP
###########################################################################################################################
	
	
	def setup_directories(self):
		"""
		Sets up the appropriate folders for the group "self".
		(.. and cleans them up.)
		"""
		
		#Create Folders (a,e,i,o,u) in output

		if self.do_make_proto:
			
			if path.isdir(self.output_path):
				system('rm -rf '+self.output_path+'/*')
			else:
				for vowel in self.vowels:
					system('mkdir --parents '+self.output_path+'/'+vowel)
		
		#Create folder for results (plots etc.)
		if self.do_setup_analysis or self.do_sample_analysis:
			if not path.isdir(self.result_path):
				system('mkdir --parents '+self.result_path)
			elif not(listdir(self.result_path) == []):
				system('rm -r '+self.result_path+'/*')
				
		#Create Folders (a,e,i,o,u) in output for the generated samples (a and null_a for example)	
		if self.do_make_samples:
			if path.isdir(self.temp_samp_path):
				system('rm -rf '+self.temp_samp_path+'/*')
			for vowel in self.vowels_with_schwa:
				if not path.isdir(self.output_path_samples+'/'+vowel):
					system('mkdir --parents '+self.output_path_samples+'/'+vowel)
				if not path.isdir(self.output_path_samples+'/null'):
					system('mkdir '+self.output_path_samples+'/'+'null')
				
				
				
		
	def read_age(self):
		"""Read out age of speakers (in float(yrs))"""
		
		#(This file "_file_age.txt" maps which speaker file has which age. See the corresponding speaker group documentation for more details)
		with open(self.speaker_path+'/_file_age.txt','r') as f:
			for line in f:
				# Extract numbers from current line (f:file y:years m:months g:gender)
				f,y,m,g = ([float(s) for s in line.split() if s.isdigit()]) 
				
				# Append right values to gender, age and filename of male/female speakers, depending on the last digit in the
				# line (which specifies gender, g==0: male, g==1: female)
				if g == 0: 		#male
					self.gender.append('male')
					if f in self.speakers:
						self.age_m.append(y+m/12)
						self.male_speakers.append(int(f))
				elif g ==1:		#female
					self.gender.append('female')
					if f in self.speakers:
						self.age_f.append(y+m/12)
						self.female_speakers.append(int(f))
				else:
					raise RuntimeError("No gender specified! Check file: '_file_age.txt' of group %s"%self.sp_group_name)


		
	
	def calc_f0_pitch(self,plot=True):
		"""
		Initializing f0 and pitch variables (later fed into par_to_wav)
		Option: add noise to the heights (f0) of the speakers.
		Option: Plot f0.
		"""
		self.f0_m = numpy.array([self.f0vals_m[self.male_speakers.index(speaker)] for speaker in self.male_speakers])
		self.f0_f = numpy.array([self.f0vals_f[self.female_speakers.index(speaker)] for speaker in self.female_speakers])
		
		# Which speakers are present?
		self.any_male = self.f0_m.size > 0
		self.any_female = self.f0_f.size > 0
		self.only_male = self.any_male and not self.any_female
		self.only_female = self.any_female and not self.any_male
		self.both_sexes = self.any_female and self.any_male
		
		#If wanted, add noise
		if self.f0_sigma != 0:
			self.f0_m = numpy.random.normal(self.f0_m,self.f0_sigma)
			self.f0_f = numpy.random.normal(self.f0_f,self.f0_sigma)
		
		#frequency to pitch mapping
		self.pitch_abs_m = self.pitch_corr + 17.312340490667562*numpy.log(self.f0_m/16.4) if self.any_male else [] #used for plotting
		self.pitch_abs_f = self.pitch_corr + 17.312340490667562*numpy.log(self.f0_f/16.4) if self.any_female else [] #used for plotting
		
		self.pitch_rel_m = self.pitch_abs_m - 52 if self.any_male else [] #used to synthesize in VTL
		self.pitch_rel_f = self.pitch_abs_f - 52 if self.any_female else [] #used to synthesize in VTL
		
		#Change counting.. (combine the pitch arrays to an array for all speakers)
		self.speaker_pitch_rel = numpy.zeros(self.size)
		m = self.pitch_rel_m.tolist() if self.any_male else []
		f = self.pitch_rel_f.tolist() if self.any_female else []
		
		
		for speaker in self.speakers:
			if speaker in self.male_speakers:
				self.speaker_pitch_rel[speaker] = m.pop(0)
			elif speaker in self.female_speakers:
				self.speaker_pitch_rel[speaker] = f.pop(0)
			else:
				raise RuntimeError("Neither male nor female speaker???")

		
		
		if plot:
		
			system('mkdir '+self.result_path+'/pitch')
			f = plt.figure()
			plt.title('Fundamental frequencies for speaker ages between 0 and 20 yrs')
			sf = f.add_subplot(111)#(121)
			if self.only_male:
				male, = sf.plot(self.age_m,self.f0_m,'b-',label='Male')
				plt.legend([male],['male'])
			elif self.only_female:
				female, = sf.plot(self.age_f,self.f0_f,'r-',label='Female')
				plt.legend([female],['female'])
			elif self.both_sexes:
				male, = sf.plot(self.age_m,self.f0_m,'b-',label='Male')
				female, = sf.plot(self.age_f,self.f0_f,'r-',label='Female')
				plt.legend([male,female],['male','female'])
			plt.xlabel('Age in yrs')
			plt.ylabel('F0 in Hz')
			f.savefig(self.result_path+'/pitch/f0_over_age.png')
			

		
		
	def sp_backup(self):
		"""
		Backup the .speaker files we could be changing to the directory 'backup'
		in the speaker group directory.
		This remains optional
		"""
		
		if path.isdir(self.current_path+'/speakers/groups/'+self.sp_group_name+'/backup'):
			print "Backup of original speaker files already exists!"
			s = raw_input("Skip backup process? (Y/y). Will otherwise terminate process.\n\t > ")
			if not (s=='y' or s=='Y'):
				print "Exiting program. - No changes were made to the speaker files."
				sys.exit(0)
				
		else: #else make a backup directory and copy all our files in the group directory into the backup directory
			system('mkdir '+self.speaker_path+'/backup')
			system('rsync --exclude=backup '+self.speaker_path+'/* '+self.speaker_path+'/backup/')
			
	

	
	def adjust_vocal_chords(self):
		"""
		This function adjusts some vocal chord parameters in the speaker file, in order to
		get more realistic results. We'll use a linear interpolation between 0 and 20 yrs, 
		e.g. in vocal chord length. (starting at 2 mm)
		"""
		#men
		if self.any_male:
			temp = 0.002+numpy.array(self.age_m)*0.0007
			self.glottis_m = {'Chord rest length':temp,'Chink length':temp/8,'Chord rest thickness':temp/3.5}		
			self.glottis_m_min = {'Chord rest length':temp/3,'Chink length':(temp/8)/2,'Chord rest thickness':(temp/3.5)/1.5}
			self.glottis_m_max = {'Chord rest length':temp*1.5,'Chink length':(temp/8)*2.5,'Chord rest thickness':(temp/3.5)*2}
		
		#women
		if self.any_female:
			factor = [0.3,0.3]
			temp = 0.002+numpy.array(self.age_f)*0.0004
			self.glottis_f = {'Chord rest length':temp*factor[0],'Chink length':temp/8*factor[1],'Chord rest thickness':temp/3.5}		
			self.glottis_f_min = {'Chord rest length':temp/3*factor[0],'Chink length':(temp/8)/2*factor[1],'Chord rest thickness':(temp/3.5)/1.5}
			self.glottis_f_max = {'Chord rest length':temp*1.5*factor[0],'Chink length':(temp/8)*2.5*factor[1],'Chord rest thickness':(temp/3.5)*2}
		
		def replaceAll(theFile,searchExp,replaceExp):
			for line in fileinput.input(theFile, inplace=1):
				if searchExp in line:
					line = line.replace(searchExp,replaceExp)
				sys.stdout.write(line)
				
		
		#for men...
		for j in self.male_speakers:
			i = self.male_speakers.index(j)
			#print self.glottis_m['Chord rest length']
			#print "That was the chord rest length for %d" %i
			the_file = self.speaker_path+'/'+str(j)+'.speaker'
			#replacing chord rest thickness
			search_exp = "<param index=\"0\" name=\"Cord rest thickness\" abbr=\"rest_thickness\" unit=\"m\" min=\"0.003000\" max=\"0.010000\" default=\"0.004500\" value=\"0.004500\"/>"
			replace_exp = "<param index=\"0\" name=\"Cord rest thickness\" abbr=\"rest_thickness\" unit=\"m\" min=\""+str(self.glottis_m_min['Chord rest thickness'][i])+"\" max=\""+str(self.glottis_m_max['Chord rest thickness'][i])+"\" default=\""+str(self.glottis_m['Chord rest thickness'][i])+"\" value=\""+str(self.glottis_m['Chord rest thickness'][i])+"\"/>"
			replaceAll(the_file,search_exp,replace_exp)
			
			#replacing chord rest length
			search_exp = "<param index=\"1\" name=\"Cord rest length\" abbr=\"rest_length\" unit=\"m\" min=\"0.005000\" max=\"0.020000\" default=\"0.016000\" value=\"0.016000\"/>"
			replace_exp = "<param index=\"1\" name=\"Cord rest length\" abbr=\"rest_length\" unit=\"m\" min=\""+str(self.glottis_m_min['Chord rest length'][i])+"\" max=\""+str(self.glottis_m_max['Chord rest length'][i])+"\" default=\""+str(self.glottis_m['Chord rest length'][i])+"\" value=\""+str(self.glottis_m['Chord rest length'][i])+"\"/>"
			replaceAll(the_file,search_exp,replace_exp)
			
			#replacing Chink length
			search_exp = "<param index=\"2\" name=\"Chink length\" abbr=\"chink_length\" unit=\"m\" min=\"0.001000\" max=\"0.005000\" default=\"0.002000\" value=\"0.002000\"/>"
			replace_exp = "<param index=\"2\" name=\"Chink length\" abbr=\"chink_length\" unit=\"m\" min=\""+str(self.glottis_m_min['Chink length'][i])+"\" max=\""+str(self.glottis_m_max['Chink length'][i])+"\" default=\""+str(self.glottis_m['Chink length'][i])+"\" value=\""+str(self.glottis_m['Chink length'][i])+"\"/>"
			replaceAll(the_file,search_exp,replace_exp)
			
			
		#for women...
		for j in self.female_speakers:
			i = self.female_speakers.index(j)
			#print self.glottis_f['Chord rest length']
			#print "That was the chord rest length for %d" %i
			the_file = self.speaker_path+'/'+str(j)+'.speaker'
			#replacing chord rest thickness
			search_exp = "<param index=\"0\" name=\"Cord rest thickness\" abbr=\"rest_thickness\" unit=\"m\" min=\"0.003000\" max=\"0.010000\" default=\"0.004500\" value=\"0.004500\"/>"
			replace_exp = "<param index=\"0\" name=\"Cord rest thickness\" abbr=\"rest_thickness\" unit=\"m\" min=\""+str(self.glottis_f_min['Chord rest thickness'][i])+"\" max=\""+str(self.glottis_f_max['Chord rest thickness'][i])+"\" default=\""+str(self.glottis_f['Chord rest thickness'][i])+"\" value=\""+str(self.glottis_f['Chord rest thickness'][i])+"\"/>"
			replaceAll(the_file,search_exp,replace_exp)
			
			#replacing chord rest length
			search_exp = "<param index=\"1\" name=\"Cord rest length\" abbr=\"rest_length\" unit=\"m\" min=\"0.005000\" max=\"0.020000\" default=\"0.016000\" value=\"0.016000\"/>"
			replace_exp = "<param index=\"1\" name=\"Cord rest length\" abbr=\"rest_length\" unit=\"m\" min=\""+str(self.glottis_f_min['Chord rest length'][i])+"\" max=\""+str(self.glottis_f_max['Chord rest length'][i])+"\" default=\""+str(self.glottis_f['Chord rest length'][i])+"\" value=\""+str(self.glottis_f['Chord rest length'][i])+"\"/>"
			replaceAll(the_file,search_exp,replace_exp)
			
			#replacing Chink length
			search_exp = "<param index=\"2\" name=\"Chink length\" abbr=\"chink_length\" unit=\"m\" min=\"0.001000\" max=\"0.005000\" default=\"0.002000\" value=\"0.002000\"/>"
			replace_exp = "<param index=\"2\" name=\"Chink length\" abbr=\"chink_length\" unit=\"m\" min=\""+str(self.glottis_f_min['Chink length'][i])+"\" max=\""+str(self.glottis_f_max['Chink length'][i])+"\" default=\""+str(self.glottis_f['Chink length'][i])+"\" value=\""+str(self.glottis_f['Chink length'][i])+"\"/>"
			replaceAll(the_file,search_exp,replace_exp)
		
		

			
	def create_prototype_wav(self): #Call function in VTL_API (api_src in data)
		"""
		Create wav files for the group "self", using the right pitch, for each (vowel) gesture in ges (array of strings)
		"""		
		
		
		for vowel in self.vowels_with_schwa:
			
			system('rm -rf '+self.output_path+'/'+vowel)
			system('mkdir '+self.output_path+'/'+vowel)
			
			
			for speaker in self.speakers:
				gender = 'male' if speaker in self.male_speakers else 'female'
				pitch = self.speaker_pitch_rel[speaker]
				print "\n\nSynthesizing "+gender+" speaker "+str(int(speaker))+"s  \""+vowel+"\"-shape with the following pitch: "+str(pitch)
				
				group_speaker = self.sp_group_name+" "+str(speaker)
				
				
				# -----------> Here, we call our VTL_API source code. (in src/VTL_API) <----------- #
				
				
				
				input_dict = {	'gesture':vowel,
								'group_speaker':group_speaker,
								'pitch_var':self.speaker_pitch_rel[speaker],
								'verbose':False }
				paths = {		'wav_folder':self.output_path+'/'+vowel }
				
				synthesize.main(input_dict,paths)
				
				
				# Process sound for the ESN network
				wav_path = self.output_paths[vowel][speaker][0]
				# simply use the sampling parameters used later. (compressed output, n_channels etc)
				self.process_sound_for_hearing(wav_path,self.sampling,dump=True)
				
				
	
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
	
	
	

#ANALYZE THE USER-SET GESTURES OF THE SPEAKER GROUP
#########################################################################################################################

	
	def transf_param_coordinates(self,direction='',parameters_input=dict(),upper_limits=[],lower_limits=[]):
		"""
		Function accepts a dictionary or an array (1 or 2 dim) of parameters and a string that is either'absolute_to_relative' or 'relative_to_absolute'
			---	Speaker shape parameters have a minimum and a maximum (for example: the toungue will only go up to a certain point). 
				Thus, we can write parameters relative to their minimum and maximum extent. The next 2 lines transform absolute parameters
				(e.g. 4.5 with min 0 and max 5) and returns relative coordinates (0.9 = 4.5 / 5).
		"""
		# Check if the input is a dictionary
		# -----------------------------------------------------------------------------------------------------------------------
		dict_input = (type(parameters_input)==dict)
		
		
		# Check how this function is being used (called from within the class, or from an other class (e.g. from 'learn')
		# -----------------------------------------------------------------------------------------------------------------------
		if dict_input:
			internal_use = not parameters_input
		else:
			internal_use = False
		
		# Correct initialisation of parameters, (saving them as dicts, or calling them from self etc.)
		# -----------------------------------------------------------------------------------------------------------------------
		if internal_use:
			parameters_rel = self.pars_rel # These internally already are dictionaries, so ready for the transformation (see next step)
			parameters_abs = self.pars
			upper_limits = self.pars_top
			lower_limits = self.pars_low
		else:
			if direction == 'absolute_to_relatve':
				parameters_rel = dict()
				if not dict_input:
					parameters_abs = dict()
					parameters_abs['vowel'] = parameters_input # Convert to a dictionary, so we can handle such input 'normally'
				else:
					parameters_abs = parameters_input
			elif direction == 'relative_to_absolute':
				parameters_abs = dict()
				if not dict_input:
					parameters_rel = dict()
					parameters_rel['vowel'] = parameters_input # Convert to a dictionary, so we can handle such input 'normally'
				else:
					parameters_rel = parameters_input
		
		
		# ACTUAL CALCULATION (parameter transformation)
		# -----------------------------------------------------------------------------------------------------------------------
		if direction == 'absolute_to_relatve':
			for vowel in parameters_abs:
				parameters_rel[vowel] = (parameters_abs[vowel] - lower_limits) / (upper_limits - lower_limits)
		
		elif direction == 'relative_to_absolute':
			for vowel in parameters_rel:
				parameters_abs[vowel] = lower_limits + parameters_rel[vowel]* (upper_limits - lower_limits)
		else:
			raise RuntimeError("Invalid use! Either 'absolute_to_relatve' or 'relative_to_absolute' as first argument")
		
		# Correct output
		# -----------------------------------------------------------------------------------------------------------------------
		if internal_use:
			self.pars_rel = parameters_rel
			self.pars = parameters_abs
		else:
			if direction == 'absolute_to_relatve':
				if not dict_input: # If the input was a list, the output should also match that type.
					return list(parameters_rel['vowel'])
				else:
					return parameters_rel
			else:
				if not dict_input:
					return list(parameters_abs['vowel'])
				else:
					return parameters_abs
			
			
			
	
	
	
	
	
	
	
	def get_shape_params_from_speakers(self,plot=True):
		"""
		Read out all shapes (e.g. "a", "e" etc) that are saved in the speaker file itself.
		The user can also simply enter one vowel, or a list of speakers (as in 
		The function will fill the following dictionaries initialized in __init__(..): 
			-self.pars[vowel][speaker,parameter]
			-self.pars_min[speaker,parameter]
			-self.pars_max[speaker,parameter]
		Plots some aspects of various parameters that e.g. were fitted by hand to produce certain vowels in certain speakers...
		
		"""
		
		
		#Initialising
		self.pars = {'@':numpy.zeros([self.size,16]),'a':numpy.zeros([self.size,16]), 'e':numpy.zeros([self.size,16]), 'i':numpy.zeros([self.size,16]), 'o':numpy.zeros([self.size,16]), 'u':numpy.zeros([self.size,16])}
		self.pars_rel = {'@':numpy.zeros([self.size,16]),'a':numpy.zeros([self.size,16]), 'e':numpy.zeros([self.size,16]), 'i':numpy.zeros([self.size,16]), 'o':numpy.zeros([self.size,16]), 'u':numpy.zeros([self.size,16])}
		self.pars_low = numpy.zeros([self.size,16])
		self.pars_top = numpy.zeros([self.size,16])
		
		
		#Do for all vowels asked for..
		for vowel in self.vowels:
			#Do for all speakers in the speaker group..
			for speaker in self.speakers:
				with open(self.speaker_path+'/'+str(speaker)+'.speaker','r') as f:
					# Find min, max and schwa (neutral) with the first vowel of the current speaker
					# ----------------------------------------------------------------------------------------
					flag = "<nasal_cavity length=\""
					start = 0
					i = 0
					text = f.readlines()
					found = False
					for line in text:
						start = start+1 #always one line ahead
						if flag in line:
								found = True
								break
								#We now have found the beginning of the relevant text.
					if not found:
						raise RuntimeError("Parameter limits not found in speaker file!")
					#Now for only the part of the speaker file with the parameters for the specific vowel:
					cut = text[start:start+23]
					mins =  numpy.array([])
					maxs =  numpy.array([])
					neutral = numpy.array([])
					# the subset of shape parameters used in learning:
					parchoice = [0,1,3,4,5,6,9,10,11,12,13,14,17,18,19,20] 
					par = 0
					for line in cut:
						li = ([s for s in line.split('\"')]) # Extract numbers from current line
						if par in parchoice:
							mins = numpy.append(mins,float(li[5])) # ..append to a temporary array
							maxs = numpy.append(maxs,float(li[7]))
							neutral = numpy.append(neutral,float(li[9]))
						par=par+1
					
					self.pars['@'][speaker,:] = neutral # and store that array in the right place..
					self.pars_low[speaker,:] = mins
					self.pars_top[speaker,:] = maxs
					# ----------------------------------------------------------------------------------------
					
					
					# Find shape parameters
					# ----------------------------------------------------------------------------------------
					flag = "<shape name=\""+vowel+"\">"
					flag2 = "<param name=\"HX\""
					start = 0
					i = 0
					#already done: text = f.readlines()
					found = False
					for line in text:
						start = start+1#always one line ahead
						if flag in line and flag2 in text[start]:
								found = True
								break
								#We now have found the beginning of the relevant text.
					if not found:
						raise RuntimeError("The vowel \""+vowel+"\" was not found in the speaker file")
					#Now for only the part of the speaker file with the parameters for the specific vowel:
					cut = text[start:start+23]
					prs =  numpy.array([])
					parchoice = [0,1,3,4,5,6,9,10,11,12,13,14,17,18,19,20]
					i = 0
					for line in cut:
						l = ([s for s in line.split('\"')]) # Extract numbers from current line
						if i in parchoice:
							prs = numpy.append(prs,l[3]) #..append to a temporary array
						i = i+1
					
					self.pars[vowel][speaker,:] = prs # and store that array in the right place..
					# ----------------------------------------------------------------------------------------
					
		

		
		# Compute relative parameters
		# ---------------------------
		self.transf_param_coordinates('absolute_to_relatve')
		
		# See if all rel pars are ok.
		for speaker in self.speakers:
			for vowel in self.vowels:
				if (self.pars_rel[vowel][speaker,:]<0).any()  or (self.pars_rel[vowel][speaker,:]>1).any():
					print "Relative Parameters bejond boundaries [0,1]!"
					raise RuntimeError("Please check gesture %s of speaker %s again in VTL"%(vowel,speaker))
		

		
		
		
		
		# Have a closer look at the parameters:
		# - Compute a motor parameter gradient (mean over all parameters for a certain age) for each vowel.
		# -----------------------------------------------------------------------------------------------------------------
		self.shape_gradient_male = dict()
		self.shape_gradient_female = dict()
		
		for vowel in self.vowels_with_schwa:
			
			# Male speakers
			# -------------------------
			
			# Initialize the gradient
			self.shape_gradient_male[vowel] = numpy.zeros((len(self.male_speakers),1))
			
			
			for speaker in self.male_speakers:
			
				
				# Current speaker index:
				index = self.male_speakers.index(speaker)
				
				if index > 0:
					
					# Compute the shape parameters of the last speaker in the series.
					# Calling index-1 will give an IndexError the first time. Simply pass then.
					previous_params = self.pars_rel[vowel][self.male_speakers[index-1]]
					
					# Compute the age difference between this speaker and the last one.
					age_difference = numpy.fabs(self.age_m[index] - self.age_m[index-1])
					
					self.shape_gradient_male[vowel][index] = numpy.mean( numpy.fabs(previous_params - self.pars_rel[vowel][speaker]) )# / age_difference
			
			
			# Female speakers
			# -------------------------
			
			# Initialize the gradient
			self.shape_gradient_female[vowel] = numpy.zeros((len(self.male_speakers),1))
				
			for speaker in self.female_speakers:
			
				
				# Current speaker index:
				index = self.female_speakers.index(speaker)
				
				if index > 0:
				
					# Compute the shape parameters of the last speaker in the series.
					# Calling index-1 will give an IndexError the first time. Simply pass then.
					previous_params = self.pars_rel[vowel][self.female_speakers[index-1]]
					
					# Compute the age difference between this speaker and the last one.
					age_difference = numpy.fabs(self.age_f[index] - self.age_f[index-1])
					
					self.shape_gradient_female[vowel][index] = numpy.mean( numpy.fabs(previous_params - self.pars_rel[vowel][speaker]) )# / age_difference
		
		# -----------------------------------------------------------------------------------------------------------------	
			
		
		if plot:
			system('mkdir '+self.result_path+'/param_plots')
			
			print ('Plots are saved in '+self.result_path+'/param_plots')
			print "Plotting and saving the mean motorshape gradient over age for each vowel....."

			
			
			
			# Male shape gradient bar plot
			import plotly.plotly as py
			import plotly.graph_objs as go
			
			
			if self.male_speakers:
				
				i=0
				data=[]
				ages = [str(int(age))+' yrs' for age in self.age_m]
			
				for vowel in self.vowels_with_schwa:
				
					xvals = ages
					yvals = self.shape_gradient_male[vowel].T.tolist()[0]
				
					datapoint = go.Bar(	x=xvals,
										y=yvals,
										name=vowel,
										marker=dict(color=self.plot_colors[vowel]))
					data.append(datapoint)
					i+=1
			
				layout = go.Layout(
										barmode='stack',
										title='Shape differences of each speaker to previous'
				)
			
				fig = go.Figure(data=data, layout=layout)
			
				py.image.save_as(fig, self.result_path+'/param_plots/shape_gradient_male.png')
				
			if self.female_speakers:
			
				# Female shape gradient bar plot
				import plotly.plotly as py
				import plotly.graph_objs as go
			
				i=0
				data=[]
				ages = [str(int(age))+' yrs' for age in self.age_m]
			
				for vowel in self.vowels_with_schwa:
				
					xvals = ages
					yvals = self.shape_gradient_female[vowel].T.tolist()[0]
				
					datapoint = go.Bar(	x=xvals,
										y=yvals,
										name=vowel,
										marker=dict(color=self.plot_colors[vowel]))
					data.append(datapoint)
					i+=1
			
				layout = go.Layout(
										barmode='stack',
										title='Shape differences of each speaker to previous'
				)
			
				fig = go.Figure(data=data, layout=layout)
			
				py.image.save_as(fig, self.result_path+'/param_plots/shape_gradient_female.png')
			


	def formants(self,plot=True):
		"""
		Getting and analyzing formant development over age / sex...
			
			- Get ages again.
			- Get formants using praat_formants_python of
				. male speakers
				. female speakers
			- Plot formants of
				. male speakers
				. female speakers
				. both together
			
			
			- Compute a 'formant gradient'
				. for male speakers
				. for female speakers
			- Plot the gradient..
				. for male speakers
				. for female speakers
		
		"""
		#raw_input("Continue and analize created ambient speech sounds?")
		
		import praat_formants_python as pfp #This requires Praat to be installed.
		
		
		# We'll need the speaker ages (draw bigger circles for older speakers in plots)..
		# ---------------
		if not self.do_setup:
			# if done already, we'd append the whole list again, behind the already present list!
			self.read_age()
		
		
		
		# Setup plot path
		# ---------------
		if plot:
			system('mkdir '+self.result_path+'/proto_formant_plots')
		
		
		
		
		
		# Getting male formants, and plotting
		# ------------------------------------------------------------------------------------------------------------
		if self.male_speakers != []:
			for vowel in self.vowels_with_schwa:
				#initialize
				self.male_formants[vowel] = numpy.array([0,0,0,0])
				for speaker in self.male_speakers:
					# Extract a list of measured formants (f0,f1,f2,f3) in the time-window (0.01 gaps)
					#formants = pfp.formants_at_interval(self.output_paths[vowel][speaker][0], 0.08,0.12)
					formants = pfp.formants_at_interval(self.output_paths[vowel][speaker][0], 0.08,1)
					
					# Compute mean across time window
					formants_mean = numpy.mean(formants,axis=0)
					# stack..
					self.male_formants[vowel] = numpy.vstack((self.male_formants[vowel],formants_mean))
				
				self.male_formants[vowel] = self.male_formants[vowel][1:,:]

			if plot:
				plt.close("all")
				#Set up the plot
				f = plt.figure()
				plt.title("Vowel formants of male speaker series")
				fsub = f.add_subplot(111)
				scatters = []
				for vowel in self.vowels_with_schwa:
					#The things we want to plot..
					x = self.male_formants[vowel][:,1] 	#F1
					y = self.male_formants[vowel][:,2]	#F2
					
#					# F1-F2 ?
#					for i in range(len(y)):
#						y[i] = y[i]-x[i]
						
					msize = (numpy.array(self.age_m)+7)**2
					new_scatter = fsub.scatter(x,y,marker='o',c=self.plot_colors[vowel],s=msize,label=vowel,alpha = 0.7)
					scatters.append(new_scatter)
				plt.legend(scatters,self.vowels_with_schwa_phonetic)	
				plt.xlim(200,1200)
				plt.ylim(500,3500)
				plt.xlabel("F1 [Hz]")
				plt.ylabel("F2 [Hz]")
				
				f.savefig(self.result_path+'/proto_formant_plots/all_male.png')
		# ------------------------------------------------------------------------------------------------------------
		
		
		
		
		
		
		# Getting female formants, and plotting
		# ------------------------------------------------------------------------------------------------------------
		if self.female_speakers != []:
			for vowel in self.vowels_with_schwa:
				#initialize
				self.female_formants[vowel] = numpy.array([0,0,0,0])
				for speaker in self.female_speakers:
					# Extract a list of measured formants (f0,f1,f2,f3) in the time-window (0.01 gaps)
					formants = pfp.formants_at_interval(self.output_paths[vowel][speaker][0], 0.08,1)
					# Compute mean across time window
					formants_mean = numpy.mean(formants,axis=0)
					# stack..
					self.female_formants[vowel] = numpy.vstack((self.female_formants[vowel],formants_mean))
				
				self.female_formants[vowel] = self.female_formants[vowel][1:,:]
				
			if plot:
				plt.close("all")
				#Set up the plot
				f = plt.figure()
				plt.title("Vowel formants of female speaker series")
				fsub = f.add_subplot(111)
				scatters = []
				for vowel in self.vowels_with_schwa:
					#The things we want to plot..
					x = self.female_formants[vowel][:,1]#F1
					y = self.female_formants[vowel][:,2]#F2
					
					
#					# F1-F2 ?
#					for i in range(len(y)):
#						y[i] = y[i]-x[i]
					
					msize = (numpy.array(self.age_m)+7)**2
					new_scatter = fsub.scatter(x,y,marker='o',c=self.plot_colors[vowel],s=msize,label=vowel,alpha = 0.7)
					scatters.append(new_scatter)
				plt.legend(scatters,self.vowels_with_schwa_phonetic)
				plt.xlim(200,1200)
				plt.ylim(500,3500)
				plt.xlabel("F1 [Hz]")
				plt.ylabel("F2 [Hz]")
				
				f.savefig(self.result_path+'/proto_formant_plots/all_female.png')
		# ------------------------------------------------------------------------------------------------------------
				
		
		
		
		
		
		
		
		# The combined male and female plot.
		# ------------------------------------------------------------------------------------------------------------
		if self.female_speakers != [] and self.male_speakers != []:
			
			if plot:
				plt.close("all")
				#Set up the plot
				f = plt.figure()
				plt.title("Vowel formants entire speaker series (male and female)")
				fsub = f.add_subplot(111)
				scatters = []
				for vowel in self.vowels_with_schwa:
					#The things we want to plot..
					x = [self.male_formants[vowel][:,1],self.female_formants[vowel][:,1]]#F1
					y = [self.male_formants[vowel][:,2],self.female_formants[vowel][:,2]]#F2
					
					
#					# F1-F2 ?
#					for i in range(len(y)):
#						y[i] = y[i]-x[i]
					
					msize = (numpy.append(numpy.array(self.age_m),numpy.array(self.age_f))+7)**2
					new_scatter = fsub.scatter(x,y,marker='o',c=self.plot_colors[vowel],s=msize,label=vowel,alpha = 0.7)
					scatters.append(new_scatter)
				plt.legend(scatters,self.vowels)
				plt.xlim(200,1200)
				plt.ylim(500,3500)
				
				plt.xlabel("F1 [Hz]")
				plt.ylabel("F2 [Hz]")
				
				f.savefig(self.result_path+'/proto_formant_plots/all_speakers.png')
		# ------------------------------------------------------------------------------------------------------------
		
		
		
		
		
		
		
		
		
		
		
		
		
		
		
			
		"""
		# Have a closer look at the formants (analogous to shape-gradient in previous function)
		# - Compute a formant-gradient (mean difference over all formants) for each vowel. These differences add up in the bar plot
		# to a bar-height which can be used as an indicator on how drastically vowel sounds changed (compared to the preceding speaker).
		# A large mean formant gradient for a certain speaker might predict trouble in speaker generalisation between that speaker and
		# it's predecessor.
		# ------------------------------------------------------------------------------------------------------------
		"""
		# ------------------------------------------------------------------------------------------------------------
		self.mean_formant_gradient_male = dict()
		self.mean_formant_gradient_female = dict()
		
		for vowel in self.vowels_with_schwa:
			
			# Male speakers
			# -------------------------
			
			# Initialize the gradient
			self.mean_formant_gradient_male[vowel] = numpy.zeros((len(self.male_speakers),1))
			
			
			for speaker in self.male_speakers:
			
				
				# Current speaker index:
				index = self.male_speakers.index(speaker)
				
				
				if index > 0:
					
					# Compute the formants f0,f1,f2&f3 of the last speaker in the series.
					# Calling index-1 will give an IndexError the first time. Simply pass then.
					previous_formants = self.male_formants[vowel][index-1,:]
					
					# Compute the age difference between this speaker and the last one.
					age_difference = numpy.fabs(self.age_m[index] - self.age_m[index-1])
					
					# Compute mean formant gradient
					self.mean_formant_gradient_male[vowel][index] = numpy.mean( numpy.fabs(previous_formants - self.male_formants[vowel][index,:]) )# / age_difference
			
			
			# Female speakers
			# -------------------------
			
			# Initialize the gradient
			self.mean_formant_gradient_female[vowel] = numpy.zeros((len(self.male_speakers),1))
				
			for speaker in self.female_speakers:
			
				
				# Current speaker index:
				index = self.female_speakers.index(speaker)
				
				if index > 0:
				
					# Compute the formants f0,f1,f2&f3 of the last speaker in the series.
					# Calling index-1 will give an IndexError the first time. Simply pass then.
					previous_formants = self.female_formants[vowel][index-1,:]
					
					# Compute the age difference between this speaker and the last one.
					age_difference = numpy.fabs(self.age_f[index] - self.age_f[index-1])
					
					# Compute mean formant gradient
					self.mean_formant_gradient_female[vowel][index] = numpy.mean( numpy.fabs(previous_formants - self.female_formants[vowel][index,:]) )# / age_difference
					
					
					
					
					
					
		
			
		# Plot gradients (male and female separately)
		# ------------------------------------------------------------------------------------------------------------
		if plot:
			system('mkdir '+self.result_path+'/proto_formant_plots')
			
			print "Plotting and saving the mean formant gradient over age for each vowel....."
			print ('Plots are saved in '+self.result_path+'/proto_formant_plots')
			
			
			
			# Male mean_formant gradient bar plot
			import plotly.plotly as py
			import plotly.graph_objs as go
			
			
			if self.male_speakers:
				
				i=0
				data=[]
				ages = [str(int(age))+' yrs' for age in self.age_m]
			
				for vowel in self.vowels_with_schwa:
				
					xvals = ages
					yvals = self.mean_formant_gradient_male[vowel].T.tolist()[0]
				
					datapoint = go.Bar(	x=xvals,
										y=yvals,
										name=vowel,
										marker=dict(color=self.plot_colors[vowel]))
					data.append(datapoint)
					i+=1
			
				layout = go.Layout(
										barmode='stack',
										title='Formant-difference-to-previous of all male speakers'
				)
			
				fig = go.Figure(data=data, layout=layout)
			
				py.image.save_as(fig, self.result_path+'/proto_formant_plots/mean_formant_gradient_male.png')
				
			if self.female_speakers:
			
				# Female mean_formant gradient bar plot
				import plotly.plotly as py
				import plotly.graph_objs as go
			
				i=0
				data=[]
				ages = [str(int(age))+' yrs' for age in self.age_m]
			
				for vowel in self.vowels_with_schwa:
				
					xvals = ages
					yvals = self.mean_formant_gradient_female[vowel].T.tolist()[0]
				
					datapoint = go.Bar(	x=xvals,
										y=yvals,
										name=vowel,
										marker=dict(color=self.plot_colors[vowel]))
					data.append(datapoint)
					i+=1
			
				layout = go.Layout(
										barmode='stack',
										title='Formant-difference-to-previous of all female speakers'
				)
			
				fig = go.Figure(data=data, layout=layout)
			
				py.image.save_as(fig, self.result_path+'/proto_formant_plots/mean_formant_gradient_female.png')
			


#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#




#GENERATE SPEECH SAMPLES
###########################################################################################################################


		
	def generate_samples(self,null):
		"""
		Generates speech samples that are either representative of the vowels in the speakerfiles or non-representative (most of those are later classified as null samples)
		"""
		
		
		
		# Initialize
		# -----------------------------------------------------------------------------------------
		# -----------------------------------------------------------------------------------------
		if null == "null":
			sampling = self.sampling_null
			prefix = 'null_'
			isnull = True
		elif null == '':
			sampling = self.sampling
			prefix = ''
			isnull = False
		else:
			raise RuntimeError("Unknown sample argument! Function accepts '' or 'null' only")
		
			
		#Get shape parameters if needed
		try:
			self.pars_rel
		except AttributeError:
			self.get_shape_params_from_speakers(plot=False)
		# -----------------------------------------------------------------------------------------
		
		
		
		
		
		
		
		# Assisting function for getting the noisy samples (slight variations of the prototypes)
		# -----------------------------------------------------------------------------------------
		def get_special_sigma(vowel,sigma):
			"""
			For vowel /u/: Sampling goes better if a slightly lower sigma is chosen (since /u/ shapes are a bit more delicate than others).
			"""
			if vowel == "u":
				sigma*=0.7
			return sigma
			
		
		def get_noisy_samples(self,vowel,sigma,isnull):
			"""
			Introduce gaussian noise in the parameters with the right sigma. 
			This will, when being synthesized, produce sounds that are either classifiable as actual gestures or null samples.
			Do this in relative coordinates and then compute absolute coordinates after adding noise (needed abso for synthesize).
			
			Note: If we're going into huge speaker groups, resampling will take much longer, since ALL samples have to be right.
			"""
			print 'sampling..'
			
			
			invalid = numpy.ones(self.pars_rel[vowel].shape, dtype=bool)	#invalid is a boolean matrix of parameters dimension.
			sample = self.pars_rel[vowel]								#initialize the sample parameters as the precise parameters set by user.
			while invalid.any():
				numpy.random.seed()
				#only those that are invalid will be (re-) 'noised'
				noise = numpy.random.normal(0,sigma,self.pars_rel[vowel].shape)
				noise[numpy.logical_not(invalid)] = 0
				sample = sample + noise
				# Which ones are invalid? (invalid is a matrix!)
				# The samples have to be in relative coordinates between 0 and 1!
				invalid = numpy.logical_or((sample < 0.0) , (sample > 1.0))
				if invalid.any():
					sample[invalid] = self.pars_rel[vowel][invalid] #take away the noise where invalid
				else:
					print 'all samples accepted!'
			return sample
			# ----
		# -----------------------------------------------------------------------------------------
		
		
		# Actual sample production
		# -----------------------------------------------------------------------------------------
		# -----------------------------------------------------------------------------------------
		
		# Define a temporary folder, where all samples will be saved to. Later on, these are sorted by the user into specific
		# Vowel folders for the ESN training.
		
		# We can allow for a few silents to pass our test, in order to ensure that we have some null samples in the ESN training.
		# It is important that silents are classified as null class later on.
		n_silents = 0
		
		for vowel in self.vowels:
			
			i_samp = 0
			for i in range(sampling['n_samples']):
				
				# Get an individual sigma for certain vowels (slightly less or more than standard sigma)
				sigma = get_special_sigma(vowel,sampling['sigma'])
				
				# Get samples from the user-set vowel parameters - with some noise. - for all the speakers!
				sample_pars_rel = get_noisy_samples(self,vowel,sigma,isnull)
				
				#Transform to absolute coordinates
				sample_pars = self.transf_param_coordinates(direction='relative_to_absolute',parameters_input=sample_pars_rel,upper_limits=self.pars_top,lower_limits=self.pars_low)
				
				
				
				for speaker in self.speakers:
					
					# Some samples will be voiceless (and useless for learning). Reason: No airflow.
					# Reject these and resample, if they are voiceless, otherwise continue to next speaker.
					rejected = True
					
					
					while rejected:
						
						#Setup paths - 2nd step
						sample_path = self.temp_samp_path+'/'+prefix+vowel+'_'+str(i_samp)
						sample_path_wav = sample_path + '.wav'
						
						i_samp = i_samp + 1
						
					
						# Synthesize sound in VTL
						# ----------------------------------------------------------------------------------------------------
						group_speaker = self.sp_group_name+" "+str(speaker)
					
						input_dict = {	'params':sample_pars[speaker].tolist(),
										'group_speaker':group_speaker,
										'pitch_var':self.speaker_pitch_rel[speaker] }
						paths = {		'wav_folder':self.temp_samp_path,
										'wav_path':sample_path_wav}
					
						print "Synthesizing speech samples of speaker "+group_speaker+" to the following path:\n\t"+path.relpath(sample_path_wav)+'\n.................'
						
						synthesize.main(input_dict,paths)
						
						rejected = not synthesize.sound_is_valid
						
						
						# If we are producing null samples, and n_silents is still below 4, we want to pass the resampling
						if rejected and isnull and n_silents<4:
							
							n_silents+=1
							
							print "\nNo airflow, Nr. %d silent null samples of 3 in total"%n_silents
							
							# Simply accept the silent sample.
							rejected = False
							
						elif rejected:
							
							print "\nWav file rejected. No airflow!\n"
							
							sample_pars_rel = get_noisy_samples(self,vowel,sigma,isnull)
						
							#Sampling only one speaker would speed up the re-sampling. For now, just re-
							#sample all the speakers again.
						
							# Parameter transformation again, since we sampled in relative parameters
							sample_pars = self.transf_param_coordinates(direction='relative_to_absolute',
																		parameters_input=sample_pars_rel,
																		upper_limits=self.pars_top,
																		lower_limits=self.pars_low)
						
							i_samp -= 1 #-which means, the false sample will be overwritten.
					
					
					
					print 'Wav file ' + sample_path_wav + ' produced'
					
					# Process sound for the learning
					if sampling['process sound']:
						self.process_sound_for_hearing(sample_path_wav,sampling,dump=True)
				
					# Output
					progress = str(int(100*i_samp/(len(self.speakers)*sampling['n_samples'])))
					print "\n Current progress of sampling /"+vowel+"/ : "+progress+" percent\n"+50*'-'+"\n"
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	

	def user_sort_samples(self):
		"""
		Adapted version of sort_samples Program by Markus Ernst
		(See additional_scripts/sort/sort_samples_v2_1.py)
		"""
		
		# Remove all txt files (because they go unused anyway) If you need them, you have to also move those around!
		#system('sudo find %s -type f -name "*.txt" -exec rm -rf {} \;'%(self.temp_samp_path))
		
		
		# Some helpful functions
		# ---------------------------------------------------------------------------------------------
		
		def playback(filename):
			"""
			function for playing back sound file
			 - argument filename: str, path to sound file
			"""
			call(["aplay", "-q", filename])
			
		
		instructions =str("#####################\n"+
						"### sort_samples ###\n"+
						"#####################\n\n"+                                          
						"This program allows for simple classification of vowels into\n"+
						"the corresponding class.\n\n"+
						"During each trial you will hear a vowel.\n"+
						"If the vowel sounds like 'a', 'i', or 'u', hit the\n"+
						"corresponding key (i.e. 'a' for a) and press enter.\n\n"+
						"To repeat the sound you just heard, just hit enter.\n"+
						"If you hear nothing, just classify it as '0'. \n\n"+
						"White noise is played between each trial.\n\n"+
						"To exit, press 'Q'. The next time you start the program,\n"+
						"it will resume where you left.\n"+
			            "#####################\n")
		
		def create_list_of_paths(directory):
			"""create_list_of_paths returns a list of strings with all paths to files found in the temp
			directory """
			list_of_paths = []
			for f in listdir(directory):
				if f.endswith('.wav') and path.isfile(directory+'/'+f):
					list_of_paths.append(directory+'/'+f)
			return list_of_paths
		
		
		def move_files(soundpath,soundclass):
			"""
			Define a small function that simply moves the .wav, .dat.gz, and .txt files to the right directory 
			(from temp to e.g. a, or null)
			"""
			
			def move(old,new):
				system('mv '+old+' '+new)
			
			
			old = soundpath[:-4]	# get rid of '.wav'
			
			oldwav = soundpath
			olddatgz = old+'.dat.gz'
			oldtxt = old+'.txt'
			
			folder,filename = path.split(old)
			
			new = self.output_path_samples+'/'+soundclass+'/'+filename
			newwav = new+'.wav'
			newdatgz = new+'.dat.gz'
			newtxt = new+'.txt'
			
			# Do the moving.
			move(oldwav,newwav)
			move(olddatgz,newdatgz)
			if path.isfile(oldtxt):
				move(oldtxt,newtxt)
		
		# ---------------------------------------------------------------------------------------------
		
		
		
		
		# Main part
		# ---------------------------------------------------------------------------------------------
		
		
		
		while True:   # while loop enables quick termination
			
			list_of_paths = create_list_of_paths(self.output_path_samples+'/temp')
			
			
			# define path to whitenoise file
			path_to_whitenoise = self.base_path+'/data/other/whitenoise.wav'
			if not path.exists(path_to_whitenoise):         # check whether noise is actually there
				print 'Error:', path_to_whitenoise, 'not found!'
				break
			
			
			
			# For more convenient user-choice of classes, use numbers (numpad of keyboard)
			# (When classifying, stick little lables on the keys?)
			classifier_dict = dict()
			for vowel in self.vowels_with_schwa:
				classifier_dict[str(self.vowels_with_schwa.index(vowel)+1)] = vowel
			classifier_dict[str(0)] = 'null'
			
			print classifier_dict
				
			
			
			# iterate through sounds for classification
			exit_signal_received = False     # is True once user wants to quit
			while (len(list_of_paths) > 0) and not exit_signal_received:
				system('clear')      # rewrite everything during before each playback
				print instructions
				
				N_remaining = len(list_of_paths)

				print 'Playing whitenoise...'
				playback(path_to_whitenoise)
				# choose random path of list_of_paths
				random_soundpath = random.choice(list_of_paths)
				
				
				# Classifier_key must be a string of an integer
				classifier_key = "-1"
				allowed_keys = []
				for key in classifier_dict:
					allowed_keys.append(key)
				
				# use a loop here, in case user enteres wrong stuff.
				while classifier_key not in allowed_keys:
					system('clear')      # rewrite everything before each playback
					print instructions

					# play the corresponding sound
					playback(random_soundpath)
					
					print "See which number to press for which vowel:"
					print classifier_dict
					
					# ask for user input
					classifier_key = raw_input('({} sounds to go) What vowel is played?: '.format(N_remaining))
							# improved string formatting

							
				# Get category category
				sound_category = classifier_dict[classifier_key]

				# remove path from list_of_paths
				list_of_paths.remove(random_soundpath)
				
				
				
				print sound_category
				move_files(random_soundpath,sound_category)
			
			
			break       # important to avoid infinite loop!

	
	
		
		
		
		
		
	def compress(self):
		"""
		Compress the speech data
		"""
		
		
		
		pth = self.base_path+'/data/output/ambient_speech/'+self.sp_group_name
		
		if not path.exists(pth+"/sample_backup.tar.gz"):
			
			import tarfile
			tar = tarfile.open(pth+"/sample_backup.tar.gz", "w:gz")
			tar.add(pth+"/samples", arcname="sample_backup")
			tar.close()
		
		
		
		
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#



#ANALYZE SPEECH SAMPLES
###########################################################################################################################

	

	def sample_formants(self,plot=True):
		"""
		Formants of vowel samples generated in 'generate_samples'.
		If only male or only female speaker data wanted, simply call ambient speech with only the right speakers as part of the group..
		"""
		
		import praat_formants_python as pfp #This requires Praat to be installed.
		
			
		vowels_and_null = self.vowels[:]
		vowels_and_null.append('null')
			
			
		for vowel in vowels_and_null:
			# Initialize
			self.samp_formants[vowel] = numpy.array([0,0,0,0])
			
			
			# Get list of paths to all samples in the corresp. sample folder
			list_of_paths = []
			directory = self.output_path_samples+'/'+vowel
			for f in listdir(directory):
				if f.endswith('.wav') and path.isfile(directory+'/'+f):
					list_of_paths.append(directory+'/'+f)
			
			for sample_path in list_of_paths:
				
				# Extract a list of measured formants (f0,f1,f2,f3) in the time-window (0.01 gaps)
				formants = pfp.formants_at_interval(sample_path, 0.08,1)
				# Compute mean across time window
				formants_mean = numpy.mean(formants,axis=0)
				# stack..
				self.samp_formants[vowel] = numpy.vstack((self.samp_formants[vowel],formants_mean))
				
					
			#Get rid of f0 and zero-rows in the beginning
			self.samp_formants[vowel] = self.samp_formants[vowel][1:,:]
		
		
			
		if plot:
			system('mkdir '+self.result_path+'/sample_formant_plots')
			
			plt.close("all")
			#Set up the 2D-plot
			f = plt.figure()
			plt.title("Formants of vowels produced by speaker series")
			fsub = f.add_subplot(111)
			plt.hold(True)
			plt.xlabel("F1 [Hz]")
			plt.ylabel("F2 [Hz]")
			scatters = []
			scatter_labels = []
			for vowel in vowels_and_null:
				#The things we want to plot..
				x = self.samp_formants[vowel][:,1]
				y = self.samp_formants[vowel][:,2]
				
#				# F1-F2 ?
#				for i in range(len(y)):
#					y[i] = y[i]-x[i]
				
				
				# Normal vowels samples:
				if not (vowel == 'null'):
					new_scatter = fsub.scatter(x,y,s=50,c=self.plot_colors[vowel],alpha=0.5,label=vowel)
				else:
					new_scatter = fsub.scatter(x,y,marker='x',s=50,c=self.plot_colors[vowel],alpha=0.3,label='null') # vowel will be "null"
				
				scatters.append(new_scatter)
				scatter_labels.append('/'+vowel+'/')
			
			plt.legend(scatters,scatter_labels)
			plt.xlim(200,1200)
			plt.ylim(500,3500)
				

			f.savefig(self.result_path+'/sample_formant_plots/all_vowels.png')
			
			"""
			#Set up the 3D-plot (doesn't look like much)
			f = plt.figure()
			plt.title("Formants of vowels produced by speaker series")
			fsub = f.add_subplot(111,projection='3d')
			plt.hold(True)
			fsub.set_xlabel("F1 [Hz]")
			fsub.set_ylabel("F2 [Hz]")
			fsub.set_zlabel("F3 [Hz]")
			scatters = []
			scatter_labels = []
			for vowel in vowels_and_null:
				#The things we want to plot..
				x = self.samp_formants[vowel][:,1]
				y = self.samp_formants[vowel][:,2]
				z = self.samp_formants[vowel][:,3]
				
#				# F1-F2 ?
#				for i in range(len(y)):
#					y[i] = y[i]-x[i]
				
				
				# Normal vowels samples:
				if not (vowel == 'null'):
					new_scatter = fsub.scatter(x,y,s=50,c=self.plot_colors[vowel],alpha=0.5,label=vowel)
				else:
					new_scatter = fsub.scatter(x,y,marker='x',s=50,c=self.plot_colors[vowel],alpha=0.3,label='null') # vowel will be "null"
				
				scatters.append(new_scatter)
				scatter_labels.append(vowel)
			
			plt.legend(scatters,scatter_labels)
#			plt.xlim(200,1200)
#			plt.ylim(500,3500)
			
			f.savefig(self.result_path+'/sample_formant_plots/all_vowels_3D_F123.png')
			"""
				

#FUNCTION FOR SOUND PROCESSING (Cochlea, etc)
###########################################################################################################################
	
	
	def process_sound_for_hearing(self,path,sampling,dump=True):
		"""
		Some small functions used in their mother-function "generate_samples". This is the only place we need brian hears while setting up ambient speech.
		"""
		
		
		
		def correct_initial(sound):
			""" function for removing initial burst from sounds"""

			low = 249                             # duration of initial silence
			for i in xrange(low):                 # loop over time steps during initial period
				sound[i] = 0.0                    # silent time step

			return sound



		def get_resampled(sound):
			""" function for adapting sampling frequency to AN model
				VTL samples with 22kHz, AN model requires 50kHz"""

			target_nsamples = int(50*kHz * sound.duration)
												# calculate number of samples for resampling
												# (AN model requires 50kHz sampling rate)
			resampled = resample(sound, target_nsamples)
												# resample sound to 50kHz
			sound_resampled = Sound(resampled, samplerate = 50*kHz)
												# declare new sound object with new sampling rate
			return sound_resampled

		  

		def get_extended(sound):
			""" function for adding silent period to shortened vowel
				 ESN requires all samples to have the same dimensions"""

			target_nsamples = 36358               # duration of longest sample
			resized = sound.resized(target_nsamples)
											    # resize samples
			return resized



		def drnl(self,sound):
			""" use predefined cochlear model, see Lopez-Poveda et al 2001"""
			cf = erbspace(100*Hz, 8000*Hz, sampling['n_channels'])    # centre frequencies of ERB scale
									            #  (equivalent rectangular bandwidth)
									            #  between 100 and 8000 Hz
			drnl_filter = DRNL(sound, cf, type='human')
									            # use DNRL model, see documentation
			print 'processing sound'
			out = drnl_filter.process()           # get array of channel activations
			if sampling['compressed']:
				out = out.clip(0.0)                    # -> fast oscillations can't be downsampled otherwise
				out = resample(out, int(round(sound.nsamples/1000.0)))
					            # downsample sound for memory reasons
			return out



		# Process sound for the learning

		# load sound file for brian.hears processing
		sound = loadsound(path)
		sound = correct_initial(sound)      # call correct_initial to remove initial burst

		sound_resampled = get_resampled(sound)
											# call get_resampled to adapt generated sound to AN model
		sound_extended = get_extended(sound_resampled)
											# call get_extended to equalize duration of all sounds
		sound_extended.save(path)       # save current sound as sound file


		# Audio processing:

		# 	-call drnl to get cochlear activation
		cochlear_activation = drnl(self,sound_extended)
	
		
		# 	-create and open new output file in gzip write mode
		with gzip.open(path[:-4]+'.dat.gz', 'wb') as outputfile: # (took off the '.wav' at the end of the path)
			cochlear_activation.dump(outputfile) # dump numpy array into output file

		
		return cochlear_activation
	
	
	
	
	
		
		
###########################################################################################################################
###########################################################################################################################
