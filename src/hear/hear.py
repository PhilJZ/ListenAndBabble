

# Class imports
# -------------------------------------------------------
	# Import the class(es) containing all the called functions and inherit all subfunctions
from src.hear.hear_functions import functions as funcs

	


# General imports
# -------------------------------------------------------
from datetime import date
import os
from os import system
import pickle
import gzip
import random
import mdp
import scipy as sp
import numpy

#global Oger
import Oger

import matplotlib
matplotlib.use('Agg')

#global pylab
import pylab

from pdb import set_trace as debug
	
class hear(funcs):
	
	
	
	def __init__(self):
		# Import all relevant parameters/functions as self. Since the class 'funcs' uses 
		# 'params' as a base class, we're also importing all variables in get_params as self.
		funcs.__init__(self)
	
		
	def main(self):	
		"""
		hear.main() does the auditory learning with the BRIAN hears and the echo state
		network (ESN).
		Initialize..

		Murakami:
		# n_training used to be 183
		# n_samples used to be 204
		"""
		
		# Setting up correct paths and folders
		self.setup_folders()
	
		# Master plots info
		if self.rank == 0:
			print 80*"-"
			print('learning', self.n_vowels, 'vowels')
			print('averaging over', self.n_trains, 'trials')
			print('network size:', self.reservoir_sizes)
			print('using compressed DRNL output:', self.compressed_output)
			print('comparing leaky network to non-leaky network:',self.do_compare_leaky)
			print('verbose mode:', self.verbose)
			print('plot mode:', self.do_plot_hearing)
			print 80*"-"
	
	
		# Make reservoir states inspectable for plotting..
		self.make_Oger_inspectable()
		

		# 2 options: (if / else)
		
		# A Test speaker generalisation (how well the ESN can generalize from other speakers, if one/some speaker(s)
		# 	are omitted from training)
		# -----------------------------------------------------------------------------------------------------------------------
		if self.do_sweep_omitted_speakers:
		# -----------------------------------------------------------------------------------------------------------------------
			
			self.do_plot_hearing = False # Turn off plotting the prototypes every single time.
			
			
			for speaker_group in self.omitted_groups:
				self.omitted_test_speakers = list()
				self.omitted_test_speakers = speaker_group
				
				print "\n\nSampling and learning from all speakers exept %s"%str(self.omitted_test_speakers)
				print "-----------------------------------------------------------------------------\n"
				
				self.get_samples()
								
				self.simulate_ESN()
				
				if self.rank == 0:
					self.master_collect_and_postprocess(choose=False) # SEE BELOW.
				
				self.error_matrix['leaky'] = numpy.vstack((self.error_matrix['leaky'],self.final_errors['leaky']))
				self.error_matrix['non-leaky'] = numpy.vstack((self.error_matrix['leaky'],self.final_errors['non-leaky']))
				
			self.error_matrix['leaky'] = self.error_matrix['leaky'][1:,:]
			self.error_matrix['non-leaky'] = self.error_matrix['non-leaky'][1:,:]
			
			
			# Plotting the error matrix:
			
			self.plot_error_matrix()
		
		
		# B The normal case: Simulate ESN and pick the best one. Plot results if wanted.
		# -----------------------------------------------------------------------------------------------------------------------	
		#else:
		self.do_plot_hearing = True
		self.omitted_test_speakers = []
	
		
		# Get all the samples produced in ambient speech..	
		self.get_samples()


		# Simulate ESN (basically the main function, which then in itself calls the interesting function 'learn')..
		self.simulate_ESN()	


		if self.rank == 0:
			self.master_collect_and_postprocess(choose=False) # SEE BELOW.
		# -----------------------------------------------------------------------------------------------------------------------
		
		"""
		# Partial ESN analysis as in what samples are chosen. See get_params. This is quite complicated and not yet fully written.
		if self.do_partial_ESN_analysis:			
			# Basically run through most of what we've done already, but for different sample data.
			# --------------------------------------------------------------------------------------------------------------
			
			# Get the first speaker to include in data samples.
			# -------------------------------------------------------------------------
			try:
				firstspeaker_male = self.age_m.index(self.generalisation_age)
			except ValueError:
				firstspeaker_male = 1000
			try:
				firstspeaker_female = self.age_f.index(self.generalisation_age)
			except ValueError:
				firstspeaker_female = 1000
			firstspeaker = min(firstspeaker_female,firstspeaker_male)
			# -------------------------------------------------------------------------
			
			
			
			# Execute this 'setup' only by master.
			if self.rank == 0: 
				list_of_excluded = [speaker for speaker in self.speakers if speaker < firstspeaker]
				
				self.exclude_speakers_from_data(list_of_excluded)
		
			
			self.get_samples()
			
			self.simulate_ESN()
			
			if self.rank == 0:
				self.master_collect_and_postprocess(choose=False) # since choose is False, we call self.theshold_analysis separately..
			
			self.threshold_analysis()
		
		"""
		
			
			
		
		
		# Save this class instance
		# ---------------------------------------------------
		f = open('data/classes/hear_instance.pickle', 'w+')
		f.truncate()
		pickle.dump(self,f)
		f.close()
		
		
		
		
		
		
	
	def master_collect_and_postprocess(self,choose=True):
		"""
		# Define a neat function of all that the master does in the end.
		# This we can simply call like that, or after every time we change the samples (take away some speakers in the samples)
		# ----------------------------------------------------------------------------------------------------------------------
		Master collects all errors and confusion matrices of leaky (and non-leaky) simulations from workers
		"""
		if self.n_workers > 1:
			self.final_errors['leaky'] = comm.gather(self.errors['leaky'], root=0)
			self.final_errors['non-leaky'] = comm.gather(self.errors['non-leaky'], roo=0) if self.do_compare_leaky else []
			self.final_cmatrices['leaky'] = comm.gather(self.c_matrices['leaky'], root=0)
			self.final_cmatrices['non-leaky'] = comm.gather(self.c_matrices['non-leaky'], root=0) if self.do_compare_leaky else []
		else:
			self.final_errors['leaky'] = self.errors['leaky']
			self.final_errors['non-leaky'] = self.errors['non-leaky'] if self.do_compare_leaky else []
			self.final_cmatrices['leaky'] = self.c_matrices['leaky']
			self.final_cmatrices['non-leaky'] = self.c_matrices['non-leaky'] if self.do_compare_leaky else []
	
		# Post-processing only by master (write results to a file)
	
		self.write_and_plot_results()
	
	
		# Chose an output ESN for the learning.
		if choose:
			self.choose_final_ESN()

			# Analyze reward threshold (after choosing).
			#if self.do_analyze_output_ESN:
			#	self.threshold_analysis()
		

	
