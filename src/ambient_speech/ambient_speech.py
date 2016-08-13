"""
This script calls all relevant functions (from setup_ambient_speech_functions in the same directory) in order to:

	1. ... set up a speaker group which was already produced in VTL and make more realistic. The speaker group is expected to contain a file with all the corresponding ages and sex of each speaker (see data/speakers&gestures/.. > Group Documentation). This speaker group should contain saved vocal shapes (such as /a/, /o/ etc). This part contains the possibility to change glottis parameters (esp. for male/female differences), change gesture-pitch to that of the f0 parameters, synthesize sounds using the VTL_API contained in src/ambient_speech.

	2. ... analyze the speaker group and the sounds produced. (looking at vocal shape parameter development over age, formant development of vowels over age) 

	3.	generate ambient speech then used in "learn"-the next step of the babbling learner. Ambient speech serves as training data for the (supervised) learning of the ESN in "learn". It is produced by taking vowel shapes and adding noise in order to produce speech samples which more or less resemble real /a/s and /i/s (by adding little or much noise).

Author:
Philip Zurbuchen

Original Version:
"Listen and Babble" by Max Murakami:
https://github.com/Masaaki88/ListenAndBabble
"""




	





# General Imports
# ---------------------------------------------------------
from os import system,path,listdir,getcwd    		# for filehandling and exec-documentation
import sys                                          #
import numpy
import matplotlib.pyplot as plt
import fileinput
import runpy
import pickle

# Class imports
# -------------------------------------------------------
	# Import the class(es) containing all the called functions and inherit all subfunctions
from src.ambient_speech.ambient_speech_functions import functions as funcs




class ambient_speech(funcs):
	



	def __init__(self):
		"""
		Import all relevant parameters/functions as self. Since the class 'funcs' uses 
		'params' as a base class, we're also importing all variables in get_params as self.
		"""
		
		print 80*"-"
		print "Initialize sub function variables.."
		funcs.__init__(self)


	def main(self):
			
		

		print "\n"
		print 80*"-"
		print "Setting up the appropriate directories and deleting all preexisting output or gesture data."
		print " -- >  calling setup_output_folder"
		self.setup_directories()
		print 80*"-"


		if self.do_setup:
			#  ------------------------------------------------------------------------------------------------
			print 80*"-"
			print "Setting correct f0 parameters for the following speaker group: \n\t\t"+self.sp_group_name #see get_params.py
			print "\n"


			print 80*"-"
			print "These are the F0 parameters for each male speaker:"
			print numpy.real(self.f0s)
			print "These are the F0 parameters for each female speaker:"
			print numpy.imag(self.f0s)
			print 80*"-"



			print "\n"
			print 80*"-"
			print "Reading age (and gender) for each speaker from speakerfile-age-file..."
			print " -- >  calling read_age"
			
			self.read_age()
			print 80*"-"



			print "\n"
			print 80*"-"
			print "Calculating f0 (fundamental frequency) for each speaker..."
			print " -- >  calling calc_f0_pitch"
			self.calc_f0_pitch(plot=self.do_setup_analysis) # plot only if do_setup_analysis is True
			print 80*"-"


			print "\n"
			print 80*"-"
			print "Creating a backup of the original speaker files..."
			print " -- >  calling sp_backup"
			self.sp_backup()
			print 80*"-"


			print "\n"
			print 80*"-"
			print "Adjusting vocal chord parameters in 'Titze model' in the speaker files..."
			print " -- >  calling adjust_vocal_chords"
			self.adjust_vocal_chords()
			print 80*"-"



		if self.do_make_proto:
	
			print "\n"
			print 80*"-"
			print "Synthesising the current speaker of our group..."
			print " -- >  calling create_wav()"
			self.create_prototype_wav()
			print 80*"-"


		if self.do_setup:
			
			print "\n"
			print 80*"-"
			print "Reading parameters of shapes (e.g. vowels) for each speaker from speaker file... + Plot"
			print " -- >  calling get_shape_params_from_speakers"
			self.get_shape_params_from_speakers(plot=self.do_setup_analysis) #Plot only if setup analysis is True
			print 80*"-"
			
			
		if self.do_setup_analysis:


			print "\n"
			print 80*"-"
			print "Getting the formants of the produced vowels..."
			print " -- >  calling formants()"
			self.formants()
			print 80*"-"
	
	
			




		if self.do_make_samples:
	
	
			print "\n"
			print 80*"-"
			print "Generating REPRESENTATIVE speech sounds that are to be used in the ESN - training"
			print " -- >  calling generate_samples with argument 'True'"
			self.generate_samples('')
			print 80*"-"
	
			print "\n"
			print 80*"-"
			print "Generating MIS-REPRESENTATIVE speech sounds that are to be used in the ESN - training (used as 'null' classification)"
			print " -- >  calling generate_samples with argument 'False'"
			self.generate_samples('null')
			print 80*"-"

		if self.do_user_check:
			
			print "\n"
			print 80*"-"
			print "The User is asked to listen to the sounds (in data/output/ambient_speech/#####_samples/...) and check the classification."
			print " -- >  calling user_sort_samples"
			self.user_sort_samples()
			print 80*"-"
			
		
	
		if self.do_sample_analysis:
	
	
			print "\n"
			print 80*"-"
			print "Getting the formants of the produced vowel samples that were generated for the ESN training..."
			print " -- >  calling sample_formants()"
			self.sample_formants(plot=True)
			print 80*"-"
			
		
		print "\n"
		print 80*"-"
		print "Compressing the samples"
		print " -- >  calling compress()"
		self.compress()
		print 80*"-"
			
		
		print "\n"
		print 80*"-"
		print "Pickle this class into data/classes/ambient_speech_instance.pickle"
		f = open('data/classes/ambient_speech_instance.pickle', 'w+')
		f.truncate()
		pickle.dump(self,f)
		f.close()
		print 80*"-"
		
		
		
		
		print "Thanks for using 'ambient speech, & functions'"
		
		
		
		
		
		
