


# Class imports
	# Import the class containing the parameters and arguments for this function.
from control.get_params import parameters as params

	# Import VTL_API needed.
from src.VTL_API.api_class import VTL_API_class
	#	.. make a class instance, or: "synthesize is-a VTL_API_class" before calling main.
	#	Call synthesize like this: synthesize.main(input_dict,paths)
synthesize = VTL_API_class()


# General imports
import os
from os import path,system,listdir
import matplotlib.pyplot as plt

import numpy
from numpy.linalg import norm
import argparse

import Oger

from brian import kHz, Hz, exp, isinf
from brian.hears import Sound, erbspace, loadsound, DRNL
from scipy.signal import resample

#from mpi4py import MPI #now use simpler parallel computing from joblib
from joblib import Parallel, delayed  
import multiprocessing

import datetime
from datetime import date
from time import *

import cPickle
import pickle
import random
from copy import deepcopy

# Python debugger
from pdb import set_trace as debug



class functions(params):
	"""
	Provides functions, called in "src/learn/learn.py"
	Includes main learning function for the Echo State Network.
	
	As with all functions that are called through a main function ('learn.py' in this case) from our shell function, this program is
	controlled using control parameters in 'get_params.py' - in the learning section. The idea is to have all parameters neatly in one
	place, and to save the parameter file along with the results.
	
	In order to understand what happens, we have to understand the structure of the code.
	
	Some segments are commented but could be build in in order to learn with multiple workers. The basic idea would be to
	leave the generation sampling to many workers, then do the main cmaes loop with the master worker.
	
	
	First group of functions 
		..are basically part of the setup. (e.g. __init__(), setup_folders(), etc.) They are all called before the actual reinforcement learning starts.
			
			Here, we initialize the parameters of the learner (from parameters gathered in ambient_speech), set up paths,
			get the ideal size of the population of samples generated in the second group of functions..
	
	
	
	
	
	Second group of functions:
		- CMA-ES Master function, which calls:  (cmaes() is called from 'learn.py' after the first group of functions)
		
			- "evaluation(...)" (creates a generation of samples for a certain x_mean. evaluates the reward), - this function in turn 
																													calls the next one:
																													
				- "environment(...)" (gets called for each sample in 'evaluation'), produces sound , calling the VocalTractLab API over a 
																						small function, api_class (in src/VTL_API/api_class.py)
																						
					- "get_confidence(...)" Gets the confidence levels for a given sample. Confidence between 0 and 1 for each class.
																							These confidences are then passed to environment, then
																							to evaluation, then to the main cmaes() function, where
																							we compute a reward for each sample (called fitness),
																							using the confidences.
					
	
	
	Third group of functions: 
		- Smaller help function to update learner parameters, get next targets etc., called from the cmaes algorithm.
		- Smaller plot and output functions.
		
		
	Author:
	Max Murakami, Philip Zurbuchen
	
	
	"""

	
	#############################################################################################################################################
	# Functions called directly from 'learn' (setup) ############################################################################################
	#############################################################################################################################################

	def __init__(self):
		"""
		Initializing relevant parameters, structures and paths.
		"""	
		
		#Inherit all parameters from params in 'parameters/get_params.py' (which is a class)
		params.__init__(self)
		self.get_learn_params()
		self.get_hear_params()
		self.get_ambient_speech_params()
		
		
		# Reload required class instances from previous steps
		# ---------------------------------------------------
		self.amb_speech = pickle.load(file('data/classes/ambient_speech_instance.pickle','r'))
		self.hear = pickle.load(file('data/classes/hear_instance.pickle','r'))
		
		
		# Taking over parts of dictionaries from amb_speech.
		# ---------------------------------------------------------------------------------------------------------
			# List of vowels for which we have parameters.
		self.vowels = self.amb_speech.vowels[:]
		self.n_vowels = len(self.vowels)
		
		# Get rid of /schwa/
		if '@' in self.vowels:
			self.vowels.pop(self.vowels.index('@'))
		
		self.ESN_std_sequence = self.vowels[:]
		self.ESN_std_sequence.append('null')
		
			# Targets to be learnt (vowels)
		if self.targets == "all":
			self.targets = self.vowels[:]
			self.targets.pop(self.targets.index('null'))
		
		self.target_index = self.targets.index(self.target)
		
		# Targets will be moved to this array when learnt:
		self.targets_learnt = []
		self.n_targets = len(self.targets)
		
		
		# Parameters from ambient speech
			# Get the relative pitch of our learner (will be added to abs pitch 52)
		if type(self.learner)==int:
			# the number of the learner is also the right index in speaker_pitch_rel - list.
			index = self.learner 
			self.learner_pitch_rel = self.amb_speech.speaker_pitch_rel[index]
		else:
			raise RuntimeError('Please chose an integer for the learner. (a speaker from the speaker group)')
			
		
		
			# 1. Take over target parameters (dict format) of all vowels from the ambient speech setup.
			# 2. Get schwa parameters as neutral position (used for energy cost etc.)
		self.target_pars = dict()
		self.target_pars_rel = dict()
		for vowel in self.targets:
			self.target_pars[vowel] = self.amb_speech.pars[vowel][self.learner,:]
			self.target_pars_rel[vowel] = self.amb_speech.pars_rel[vowel][self.learner,:]
		
		# The current record fitness of each target.
		self.peak = dict()
		for vowel in self.targets:
			self.peak[vowel] = (0,numpy.zeros([len(self.target_pars[vowel])]))
		
		
			# Get schwa parameters as neutral position (used for energy cost etc.)
		self.neutral_pars = self.amb_speech.pars['@'][self.learner,:]
		self.neutral_pars_rel = self.amb_speech.pars_rel['@'][self.learner,:]
		
			# look up dictionary for parameter indices
		self.par_names = self.amb_speech.par_names
		
		
		# Take over the parameter-coordinate-system-transformation-function from ambient speech setup.
			# This includes taking over upper and lower boundaries for the vowel parameters.
		self.pars_top = self.amb_speech.pars_top[self.learner][:]
		self.pars_low = self.amb_speech.pars_low[self.learner][:]
		
		
		# Save certain parameters for every step.
		self.reward_history = dict()
		self.sigma_history = dict()
		self.learner_pars_history = dict()
		self.learner_pars_rel_history = dict()
		for target in self.targets:
			self.reward_history[target] = []
			self.sigma_history[target] = []
			self.learner_pars_history[target] = []
			self.learner_pars_rel_history[target] = []
			
			
		
		
		
		# Verbosity
		self.verbose = self.be_verbose_in['learn']
		
		# The stages (e.g. [8,16,32] of nr of iteration (each sample counts as an iteration!)
		# Initialize: Zero list, for each target one entry.
		self.iteration_stages = []
		for i in range(len(self.targets)):
			self.iteration_stages.append([])
		
		
		
		# Initialize paths
		# ---------------------------------------------------------------------------------------------------------
			# We have the same main folder, so..
		self.base_path = self.amb_speech.base_path
			#Get the path to the .flow file of the current auditory system
		self.ESN_path = self.base_path+self.ESN_path
		
			#Open flow file once! (Classifier
		flow_file = open(self.ESN_path, 'r')   
		self.flow = cPickle.load(flow_file)      	
		flow_file.close()
		
			#Store general output, trained speaker etc. in an output_folder
		self.output_folder = self.base_path+'/data/output/learn'
			#Store results of the reinforement learning
		self.result_folder = self.base_path+'/results/learn'
		if self.subfolder['learn']:
			self.output_folder = self.output_folder+'/'+self.subfolder['learn']
			self.result_folder = self.output_folder+'/'+self.subfolder['learn']
			
		# Where to find the learner (speaker file +'.speaker')
		self.learner_path = self.base_path+'/data/speakers_gestures/'+self.amb_speech.sp_group_name+'/'+str(self.learner)

		
		
	def setup_folders(self):
		"""
		Sets up the appropriate folders for the group "self".
		(.. and cleans them up.)
		"""
		
		if self.rank==0:
			
			
			#Create folder Output
			#-----------------------------------------------------
			if path.isdir(self.output_folder):
				system('rm -r '+self.output_folder+'/*')
			else:
				system('mkdir --parents '+self.output_folder)
				
			#Create environment folder (learner sounds etc..)
			#-----------------------------------------------------
			if path.isdir(self.output_folder):
				system('rm -r '+self.output_folder+'/*')
			else:
				system('mkdir --parents '+self.output_folder)
			if self.save_peak:
				system('mkdir --parents '+self.output_folder+'/current_peak')
			
			#Create folder for results (plots etc.)
			#-----------------------------------------------------
			if not path.isdir(self.result_folder):
				system('mkdir --parents '+self.result_folder)
			elif not(listdir(self.result_folder) == []):
				system('rm -r '+self.result_folder+'/*')
			system('mkdir '+self.result_folder+'/snapshot')

	
	def init_par_indices_and_dimension(self):
		"""
		Get the right indices of the parameters included in the learning.
		Other parameters kept static.
		self.N_dim (dimension of the learning problem)
		"""
		self.i_pars_to_learn = []			 # prepare list for parameter indices
		if self.rank==0:
			print 'Parameters to learn:', self.pars_to_learn

		if self.pars_to_learn == ['all']:		   # case: full-dimensional problem
			self.i_pars_to_learn = range(16)  # all 16 parameters are being learnt
			self.N_dim = 16						  # change dimension from 1 to 16 
		elif self.pars_to_learn == ['flat']:
			self.i_pars_to_learn = range(12)
			self.N_dim = 12
		else:							   # case: only some parameters are learnt, rest is fixed
			# go through all parameter indices, and check whether
			# it's corresponding parameter is a parameter to learn.
			# If yes, simply append. We now have two lists: param-
			# eters to learn, and their indices in a list. (i_pars_to_learn)
			for i_par in range(len(self.par_names)):
				if self.par_names[i_par] in self.pars_to_learn:
					self.i_pars_to_learn.append(i_par)
	
		self.N_dim = len(self.i_pars_to_learn)
	
	
	def init_shape_parameters(self):
		"""
		- Copy target parameters
		- Change values, where we want to learn, to /@/ values (neutral)
		"""
		self.learner_pars = deepcopy(self.target_pars)
		self.learner_pars_rel = deepcopy(self.target_pars_rel)
		
		for target in self.targets:
			self.learner_pars[target][self.i_pars_to_learn] = self.neutral_pars[self.i_pars_to_learn]
			self.learner_pars_rel[target][self.i_pars_to_learn] = self.neutral_pars_rel[self.i_pars_to_learn]

		
	def get_population_size(self):
		"""
		Getting population size (lambda_). (see Hansen) 
		"""
		
		
		self.population_size = self.n_workers - 1

		# numpy random seed w.r.t. global runtime
		numpy.random.seed()
		# numpy random seed w.r.t. worker
		numpy.random.seed(numpy.random.randint(256) * self.rank+1)

		if self.n_workers == 1:	  # serial mode -> disable parallel features
			lambda_list = [4,6,7,8,8,9,9,10,10,10,11,11,11,11,12,12,12,12]
				# list of recommended lambda values for given number of
				#  dimensions (see Hansen)					
			self.population_size = lambda_list[self.N_dim-1]
			
		

	
	#############################################################################################################################################
	#############################################################################################################################################
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
																	#		MAIN FUNCTION STARTS HERE:
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
	#############################################################################################################################################
	# MASTER FUNCTION ###########################################################################################################################
	#############################################################################################################################################
	
	
	
	
	
	
	
	def cmaes(self):						# actual CMA-ES part
		"""
		CMA-ES:
		See http://image.diku.dk/igel/paper/NfRLUES.pdf
		Christian Igel. "Neuroevolution for Reinforcement Learning Using Evolution Strategies".
		
		http://image.diku.dk/igel/paper/NSfERL-orig.pdf
		
		Implemented in Code:
		"""


		

		
		#######################################################
		# Loop-Parameter-Initialization
		#######################################################
		
		
		# Check whether the user specified a target to begin with. If not, random choice.
		if not self.target:
			self.target = random.choice(self.targets)
		# Get target index.
		self.target_index = self.targets.index(self.target)
		
		# sigma_0 is the initial sigma (always stays the same). 
		# The learner will reset every time he runs into a local
		# minimum. (during conversion, 'current_sigma' gets smaller
		# and smaller. Then, after reset, sigma (the starting value of
		# current_sigma) can be altered. Thus, we have 3 sigmas, of 
		# which sigma_0 will always stay the same, current_sigma changes slowly 
		# with every reset. sigma is changed after every generation.
		# For now, we set both sigma and current_sigma to sigma_0
		sigma = self.sigma_0
		current_sigma = self.sigma_0
		
		# initialize x_mean. x_mean is the current state of our learnt parameters.
		# The learner_pars will be updated using x_mean.
		# Offspring will be generated around xmean with a gaussian distribution of
		# sigma!
		# For initialisation: Doesn't matter which vowel.. Used /schwa/ here.
		# Convert to list (from numpy), in order to make it of type deque() 
		# deque: (fast datatype, like list)
		
		x_mean = self.learner_pars_rel[self.target][self.i_pars_to_learn]
		x_mean = x_mean.tolist()
		
		# recent x_mean
		x_recent = []
		x_recent.append(x_mean)
		
		# recent fitness
		sorted_fitness_recent = list()
		
		
		# Some variables needed for the learning.
		# ------------------------------------------------------------------------------------------------
			#(self.population_size is the same as lambda_ in Murakami's code.)
		mu_ = self.population_size / 2.0			 # mu_ is float
		mu = int(numpy.floor(mu_))		 # mu is integer = number of parents/points for recombination
		weights = numpy.zeros(mu)
		for i in xrange(mu):			# muXone recombination weights
			weights[i] = numpy.log(mu_+0.5) - numpy.log(i+1)
		weights /= sum(weights)		 # normalize recombination weights array 
		mu_eff = sum(weights)**2 / sum(weights**2)
										# variance-effective size of mu
		
			# window for convergence test
		if not self.user_convergence_interval:
			convergence_interval = int(10+numpy.ceil(30.0*self.N_dim/self.population_size))
		else:
			convergence_interval = self.user_convergence_interval
		i_reset = 0

			# Strategy parameter setting: Adaptation
		c_c = (4.0+mu_eff/self.N_dim) / (self.N_dim+4.0+2*mu_eff/self.N_dim)
										# time constant for cumulation for C
		c_s = (mu_eff+2.0) / (self.N_dim+mu_eff+5.0)
										# time constant for cumulation for sigma control
		c_1 = 2.0 / ((self.N_dim+1.3)**2 + mu_eff) # learning rate for self.rank-one update of C
		c_mu = 2 * (mu_eff-2.0+1.0/mu_eff) / ((self.N_dim+2.0)**2 + 2*mu_eff/2.0)
										# and for self.rank-mu update
		damps = 1.0 + 2*numpy.max([0, numpy.sqrt((mu_eff-1.0)/(self.N_dim+1.0))-1.0]) + c_s
										# damping for sigma
	

		# Initialize dynamic (internal) strategy parameters and constants
		p_c = numpy.zeros(self.N_dim)			   # evolution path for C
		p_s = numpy.zeros(self.N_dim)			   # evolution path for sigma
		B = numpy.eye(self.N_dim)				   # B defines the coordinate system
		D = numpy.eye(self.N_dim)				   # diagonal matrix D defines the scaling
		B_D = numpy.dot(B,D)
		C = numpy.dot(B_D, (B_D).T)			   # covariance matrix
		i_eigen = 0					 # for updating B and D
		chi_N = numpy.sqrt(self.N_dim) * (1.0-1.0/(4.0*self.N_dim) + 1.0/(21.0*self.N_dim**2))
										# expectation of ||self.N_dim(0,I)|| == norm(randn(self.N_dim,1))
		# ------------------------------------------------------------------------------------------------
		
		
		# Initialize Fitness (reward for each sample of one generation). 
		fitness = numpy.zeros(self.population_size)
		
		
		# Some last initialisations..
		# -------------------------------------
		error = False
		fitness_mean = 0.0
		
		i_count = 0
		
		return_dict = dict()

		t_0 = datetime.datetime.now()
		t_reset = t_0
		# -------------------------------------
		
		
		
		#######################################################
		# Generation Loop
		#######################################################

		while True:
			
			print 4*'\n'
			print 40*"-"
			print "Generating new generation of speech gestures.."
			print 40*"-"
			
			
			print "Current target is "+self.target
			
			print "Learnt targets are: "+str(self.targets_learnt)
			
			
			
			# Generate and evaluate lambda offspring. The 'evaluation' function samples around x_mean and synthesizes (while calling 'environment' - both functions below..) sounds.
			# These sounds are evaluated with the ESN from 'hear'. self.confidences is created, which is then used to compute the fitness (reward).
			# Resulting confidences will be zero-valued if we don't yet have a target.
			# --------------------------------------------------------------------------------------------------------------------------
			self.evaluation(x_mean,sigma, B_D)
			"""
			# do the following way, for more than one workers.. (if self.n_workers > 1:)
	  		#self.z_offspring, self.x_offspring, self.confidences, self.energy_cost, self.evaluated_boundary_penalty, N_resampled_trial = self.parallel_evaluation(x_mean, B_D)
			"""
			
			# Each offspring sample counts toward the total count.
			i_count += self.population_size
			self.iteration_stages[self.target_index].append(i_count)
	  		
	  		
	  		
			# Compute the fitness (reward) of the learner parameters.
			# ---------------------------------------------------------------------------------------------------------------------------
										# These values we all got from self.evaluation.
			fitness = -self.confidences.T[self.target_index] + self.energy_factor*self.energy_cost + self.alpha*self.evaluated_boundary_penalty
			
			
			# Fitness will be a vector (one fitness for each sample in our generation.) A little later (after checking target) we see if the fitness
			# of one of these samples is above the convergence threshold..	
			
			
			
			
			# Check if we've reached a new maximum confidence. If so, output to result path.
			if self.save_peak and min(fitness) < -self.peak[self.target][0]: # Index null is the reward, 1 the corresp.parameters
				print "New peak "+self.target
				self.new_peak(fitness)
				
				# If this step is done (or, later - if we're over the convergence threshold), then the learner pars will be the ones with maximum
				# reward. If this step is not done, then the parameters are updated at the end, with the computed x_mean
			
			
			
			
			
			"""
			# Now, in this part of the loop, we evaluate various cases.
			# 	- are we already over the convergence threshold? In this case, we've learnt the current target. > On to the next..
			#	- are we still below that threshold? (else:) > Update x_mean and other parameters, start loop again..
			# (each part is separated by some empty lines)
			"""
			
			
			
			
			
			
			
			# If we are above the reward (convergence) threshold in one of the samples..
			# ---------------------------------------------------------------------------------------------------------------------------
			if (fitness < -self.convergence_thresholds[self.target]).any() and not self.must_converge:		#  - remember: we defined fitness as negative.
				
				print "Learnt target "+self.target
				self.targets_learnt.append(self.target)
				
				
				
				# These next three steps should theoretically already have been done in self.new_peak, since, being over the convergence threshold means that
				# we must have a maximum reward (since we haven't crossed the threshold for this target yet before).
				
				# Which sample is the one above convergence threshold?
				i_argmax = fitness.argmin()
				
				# Take the parameters used for that sample as x_mean.
				x_mean = self.x_offspring[i_argmax][:]
				
				self.update_learner_pars(x_mean)
				
				
			
				return_dict[self.target+'_steps'] = i_count-i_reset
				return_dict[self.target+'_time'] = datetime.datetime.now()-t_reset
				return_dict[self.target+'_reward'] = -fitness[i_argmax]

				i_reset = i_count
				t_reset = datetime.datetime.now()

				
				
				
				
				# We've reached a target. Now check if there are still some to learn.
				if not(set(self.targets)==set(self.targets_learnt)) and self.intrinsic_motivation:
					# ..which means, the algorithm will look for the next one, in the next iteration.
					p_c = numpy.zeros(self.N_dim)		
					p_s = numpy.zeros(self.N_dim)		
					B = numpy.eye(self.N_dim)			 
					D = numpy.eye(self.N_dim)   
					B_D = numpy.dot(B,D)		
					C = numpy.dot(B_D, (B_D).T)
					i_eigen = 0
					current_sigma = self.sigma_0			   
					sigma = current_sigma
					i_reset = 0

				else:
					print 'terminating.'
					print 'i_reset:', i_reset, ', confidence:', -fitness[i_argmax]
					#tag = int(i_count/(self.n_workers))
					#for i_worker in xrange(1,self.n_workers):
					#	comm.send((None,None,None,None,None), dest=i_worker, tag=tag)
					
					
					# Save state! (For all targets, save current reward, sigma (0, if not selected as current aim of the learner) and the parameters.)
					# --------------------------			
					self.save_snapshot(sigma,fitness)
					
					# End while loop!
					break
				
				
				
			
			
			
			
			
			
			
			
			
			
			
			# If we are NOT below the convergence threshold..
			# ---------------------------------------------------------------------------------------------------------------------------	
			else:
				
				# Sort by fitness and compute weighted mean into x_mean
				indices = numpy.arange(self.population_size)
				to_sort = zip(fitness, indices)
											# minimization
				to_sort.sort()
				sorted_fitness, indices = zip(*to_sort)
				sorted_fitness = numpy.array(sorted_fitness)
				indices = numpy.array(indices)
				x_mean = numpy.zeros(self.N_dim)
				z_mean = numpy.zeros(self.N_dim)
				fitness_mean = 0.0
				for i in xrange(mu):
					x_mean += weights[i] * self.x_offspring[indices[i]]
											# recombination, Eq. 39
					z_mean += weights[i] * self.z_offspring[indices[i]]
											# == D^-1 * B^T * (x_mean-x_old)/sigma
					fitness_mean += weights[i] * sorted_fitness[indices[i]]
				
				# Update learner pars after saving a snapshot (at the end..)
				
				#self.output_write.write(str(datetime.datetime.now()-t_0)+'  '+str(i_count)+'  '+str(-fitness_mean)+'  '+str(self.target_index)+'  ')
				#for confidence in self.confidences[0]:
				#	self.output_write.write(str(confidence)+'  ')
				#self.output_write.write(str(self.energy_cost[0])+'  '+str(self.evaluated_boundary_penalty[0])+'  '+str(sigma)+'  '+str(N_resampled_trial)+'\n')
				#self.output_write.write('	rel coords: '+str(self.learner_pars_rel)+'\n')
				#self.output_write.write('	abs coords: '+str(self.learner_pars)+'\n\n')
				#self.output_write.flush()
				
				
				# Cumulation: Update evolution paths
				p_s = (1.0-c_s)*p_s + (numpy.sqrt(c_s*(2.0-c_s)*mu_eff)) * numpy.dot(B,z_mean)
											# Eq. 40
				h_sig = int(numpy.linalg.norm(p_s) / numpy.sqrt(1.0-(1.0-c_s)**(2.0*i_count/self.population_size))/chi_N < 1.4+2.0/(self.N_dim+1.0))
				p_c = (1.0-c_c)*p_c + h_sig * numpy.sqrt(c_c*(2.0-c_c)*mu_eff) * numpy.dot(B_D,z_mean)
											# Eq. 42

	
				# Adapt covariance matrix C
				# -----------------------------------------------------------------------------
			   
				C_new = (1.0-c_1-c_mu)*C + c_1*(numpy.dot(p_c,p_c.T) + (1.0-h_sig)*c_c*(2.0-c_c)*C) + c_mu*numpy.dot(numpy.dot((numpy.dot(B_D, self.z_offspring[indices[:mu]].T)),numpy.diag(weights)),(numpy.dot(B_D, self.z_offspring[indices[:mu]].T)).T)

				if not (numpy.isfinite(C_new)).all():
					print 'Warning! C contains invalid elements!'
					error = True
				else:
					C = C_new			   # regard old matrix plus self.rank one update plus minor correction plus self.rank mu update, Eq. 43
				
				# Adapt step-size sigma
				# -----------------------------------------------------------------------------
				sigma = sigma * numpy.exp((c_s/damps) * (numpy.linalg.norm(p_s)/chi_N - 1.0))
											# Eq. 41
				
				
				# Update B and D from C
				# -----------------------------------------------------------------------------
				if i_count - i_eigen > self.population_size/(c_1+c_mu)/self.N_dim/10.0:
											# to achieve O(self.N_dim**2)
					i_eigen = i_count
					C_new = numpy.triu(C) + numpy.triu(C,1).T
											# enforce symmetry
					cond = numpy.linalg.cond(C_new)
					if not (numpy.isfinite(C_new)).all():# or (C_new < 0.0).any()):
						print 'Warning! C contains invalid elements!'
						print 'C:', C_new
						print 'repaired to C=', C
						print 'conditioning number of C:', cond
						error = True
					else:
						C = C_new

					if (numpy.iscomplex(C)).any():
						print 'Warning! C contains complex elements!'
						print 'C:', C
						print 'conditioning number of C:', cond
						error = True

					D, B = numpy.linalg.eig(C) # eigen decomposition, B==normalized eigenvectors?
					if (D < 0.0).any():
						print 'Warning! D contains negative elements!'
						for i in xrange(len(D)):
							if D[i] < 0.0:
								D[i] = -D[i]
								print -D[i], 'repaired to', D[i]
					D = numpy.diag(numpy.sqrt(D)) # D contains standard deviations now
					B_D = numpy.dot(B,D)


				# Escape flat fitness
				# -----------------------------------------------------------------------------
				if False:#sorted_fitness[0] == sorted_fitness[int(numpy.ceil(0.7*self.population_size))]:
					sigma *= numpy.exp(0.2+c_s/damps)
					print 'Warning: flat fitness, consider reformulating the objective'
					print 'Fitness (highest first):', sorted_fitness
					
				
				# Do we have anough data points to even thing about convergence (and termination)?
				enough = False
				while len(x_recent) > convergence_interval - 1:
					x_recent.pop(0)
					enough = True
				while len(sorted_fitness_recent) > convergence_interval - 1:
					sorted_fitness_recent.pop(0)
					enough = True
					
				x_recent.append(x_mean)
				sorted_fitness_recent.append(fitness_mean)
				
				# Compute termination criteria (C_matrix_criterium not dependant on how many data points)
				cond = numpy.linalg.cond(C)
				C_matrix_criterium = (cond > self.conditioning_maximum)
				if enough:
					recent_params_range = numpy.ptp(x_recent, axis=0)
					converged_params = (recent_params_range < self.range_for_convergence).all()
					converged_fitness = (numpy.ptp(sorted_fitness_recent) < self.range_for_convergence)
				else:
					converged_params = False
					converged_fitness = False
				
				termination = converged_fitness or converged_params or C_matrix_criterium
				
				
				if termination:
					print 'convergence criterion reached.'
					if (sorted_fitness[0] > -self.convergence_thresholds[self.target]): # confidence worse than desired
						print 'reward too low. resetting sampling distribution.'
						print 'reward', -sorted_fitness[0], '<', self.convergence_thresholds[self.target]
						p_c = numpy.zeros(self.N_dim)		
						p_s = numpy.zeros(self.N_dim)		
						B = numpy.eye(self.N_dim)			 
						D = numpy.eye(self.N_dim)
						B_D = numpy.dot(B, D)		   
						C = numpy.dot(B_D, (B_D).T)
						i_eigen = 0
						if self.random_restart and self.targets_learnt:
							if current_sigma < 0.9 and not self.keep_sigma_constant:
								current_sigma += 0.05			  
							sigma = current_sigma
							
							# Preferably chose a already learned target
							random_target = random.choice(self.targets_learnt)
						
							print 'Agent chose to restart search of current target from learnt parameters of target /%s/'%random_target
								
							x_mean = self.learner_pars_rel[random_target][self.i_pars_to_learn]
								

						else:
							x_mean = self.neutral_pars_rel[self.i_pars_to_learn]
							print 'Agent chose to restart search of current target from neutral parameters.'
							if current_sigma < 0.9 and not self.keep_sigma_constant:
								current_sigma += 0.1
								sigma = current_sigma
								print "Sigma increased!"
								
						
						

					else:
						print 'reward:', -sorted_fitness[0], ', i_reset:', i_reset
						self.targets_learnt.append(self.target)
	
						if (len(self.targets_learnt) < self.n_targets+1) and self.intrinsic_motivation:
							p_c = numpy.zeros(self.N_dim)		
							p_s = numpy.zeros(self.N_dim)		
							B = numpy.eye(self.N_dim)			 
							D = numpy.eye(self.N_dim)		   
							C = numpy.dot(B_D, (B_D).T)	
							i_eigen = 0
							current_sigma = self.sigma_0
							print "sigma0: "+str(self.sigma_0)		   
							sigma = current_sigma
							i_reset = 0
						else:
							print 'terminating.'
							print 'i_reset:', i_reset, ', confidence:', -sorted_fitness[0]
							tag = int(i_count/(self.n_workers-1))
							
							# Save state! (For all targets, save current reward, sigma (0, if not selected as current aim of the learner) and the parameters.)
							# --------------------------			
							self.save_snapshot(sigma,fitness)
							
							break
			
			
			
			
			
			
			
			print 'Iteration:',i_count,', time elapsed:', datetime.datetime.now()-t_0, ', target:', self.target, ', reward:%.2f'%(-fitness_mean)
			
			
			
			# Do the following for all generations! (Whether below or above convergence criterium) / of course, only if not broken already (not all targets found)
			# ---------------------------------------------------------------------------------------------------------------------------
			if error:
				print 'Critical error occurred!\nTerminating simulations.'
				#tag = int(i_count/(self.n_workers))
				#
				#for i_worker in xrange(1,self.n_workers):
				#	comm.send((None,None,None,None,None), dest=i_worker, tag=tag)
				break
			
			
			# Update for next batch
			self.update_learner_pars(x_mean)
			
			
				
			# Save state! (For all targets, save current reward, sigma (0, if not selected as current aim of the learner) and the parameters.)
			# --------------------------			
			self.save_snapshot(sigma,fitness)
						
			# Get next target vowel.. Intrinsic motivation means: The learner picks the target which yielded the maximum confidence in the last round.
			# ---------------------------------------------------------------------------------------------------------------------------
			if self.intrinsic_motivation or self.target in self.targets_learnt:
				
				self.get_next_target()
				# Update the learner pars of the new target now!
				self.update_learner_pars(x_mean)
		
				
			
			
			#  <----  <----  <----  RESTART LOOP, until learnt.   <----  <----  <----  <----  <----  <----  <----
			
		#	
		#	
		#	
		#	
		#	
		#
		#
		
		
		
		
		
		
		
		
		
		# At the end...
		# ---------------------------------------------------------------------------------------------------------------------------
		return_dict['time'] = datetime.datetime.now()-t_0
		return_dict['steps'] = i_count
		
		
		# Synthesize learnt speech sounds.
		# -----------------------------------------------------------------------------------------------
		for learnt_target in self.targets_learnt:
			
			if not type(self.learner_pars[learnt_target]) == list:
				self.learner_pars[learnt_target] = self.learner_pars[learnt_target].tolist()
				
			input_dict = {	'params':self.learner_pars[learnt_target], #The final parameters
							'gesture':'learnt_'+learnt_target, #The params will be written into that gesture (e.g. <shape... 'learnt_a'..>) in the speaker file
							'group_speaker':self.amb_speech.sp_group_name+' '+str(self.learner), 
							'pitch_var':self.learner_pitch_rel,
							'verbose':False }
			
			paths = {		'wav_folder':self.result_folder, 'wav_path':self.result_folder+'/learnt_%s.wav'%learnt_target}
		
			synthesize.main(input_dict,paths)
			wav_path = synthesize.wav_path
		
			# Plot the mean sample votes for the learnt vowels.
			confidence[learnt_target],mean_sample_vote[learnt_target] = self.get_confidence(self.wav_path,True)
		
		
		# Main function returns final data.
		return return_dict
	
	
	#############################################################################################################################################


















																						# HELP FUNCTIONS FOR THE MASTER FUNCTION



	#############################################################################################################################################
	
	def evaluation(self,x_mean,sigma, B_D):
		"""
		evaluate rewards if only one worker exists
		-> interface of agent and environment, called during each environment evaluation
		-> executed only by master
		 - arguments:
		  - x_mean: numpy.array of length self.N_dim, mean of current sampling distribution
		  - sigma: float, width of current sampling distribution
		  - B_D: numpy.array of shape (self.N_dim,self.N_dim), covariance matrix of current sampling distribution
		  - i_count: int, current iteration step
		  - self.target: int, index of current target vowel	  
		 - globals:
		  - self.n_workers: int, number of worker (=slaves+master)
		  - self.verbose: bool, for printed stuff
		  - self.n_targets: int, total number of target vowels
		 - outputs:
		  - self.z_offspring: numpy.array of shape (lambda, self.N_dim), sampled z values of each slave for each coordinate "COS" = Current offspring
		  - x: numpy.array of shape (lambda, self.N_dim), corresponding parameter values of each slave for each coordinate
		  - self.confidences: numpy.array of shape (lambda, self.n_targets+1), corresponding confidence levels for each target vowel + null class
		  - self.energy_cost: numpy.array of length lambda, corresponding energy penalty for each slave's sample
		  - self.evaluated_boundary_penalty: numpy.array of length lambda, corresponding boundary penalty for each slave's sample
		"""

		
		
		self.confidences = numpy.zeros([self.population_size,len(self.ESN_std_sequence)])
		self.energy_cost = numpy.zeros(self.population_size)
		self.evaluated_boundary_penalty = numpy.zeros(self.population_size)
		self.z_offspring = numpy.zeros([self.population_size, self.N_dim])
		self.x_offspring = numpy.zeros([self.population_size, self.N_dim])
		N_resampled = -self.population_size
		
		for i in xrange(self.population_size):   # offspring generation loop

			invalid = True
			if self.resample['normal'] and not (self.resample['penalty'] and self.resample['specific']):
				
				while invalid:
					N_resampled += 1
					z_i = numpy.random.randn(self.N_dim)	  # standard normally distributed vector
					x_i = x_mean + sigma*(numpy.dot(B_D, z_i))  # add mutation, Eq. 37
					invalid = (x_i < 0.0).any() or (x_i > 1.0).any()

				boundary_penalty_i = 0.0
			elif self.resample['penalty'] and not (self.resample['normal'] and self.resample['specific']):
				N_resampled = 0
				z_i = numpy.random.randn(self.N_dim)		  # standard normally distributed vector
				x_i = x_mean + sigma*(numpy.dot(B_D, z_i)) # add mutation, Eq. 37
				boundary_penalty_i = 0.0
				if (x_i<0.0).any() or (x_i>1.0).any(): # check boundary condition
					if self.verbose:
						print 'boundary violated. repairing and penalizing.'
					x_repaired = x_i.copy()	   # repair sample
					for i_component in xrange(len(x_i)):
						if x_i[i_component] > 1.0:
							x_repaired[i_component] = 1.0
						elif x_i[i_component] < 0.0:
							x_repaired[i_component] = 0.0
					boundary_penalty_i = numpy.linalg.norm(x_i-x_repaired)**2
												# penalize boundary violation, Eq. 51
					x_i = x_repaired[:]
					
			elif self.resample['specific'] and not (self.resample['penalty'] and self.resample['normal']):
				

				# This is similar to 'get noisy parameters' in ambient_speech_functions.py
				invalid = numpy.ones(len(x_mean), dtype=bool)
				x_i = numpy.array(x_mean[:])
				boundary_penalty_i = 0.0
				while invalid.any():
					numpy.random.seed()
					z_i = numpy.random.randn(self.N_dim)	  # standard normally distributed vector
					perturbation = sigma*(numpy.dot(B_D, z_i))  # add mutation, Eq. 37
					perturbation[numpy.logical_not(invalid)] = 0 # only those that are invalid will be (re-) 'noised'
					x_i = x_i + perturbation
					
					# Which ones are invalid? (invalid is a matrix!)
					# The samples have to be in relative coordinates between 0 and 1!
					invalid = numpy.logical_or((x_i < 0.0) , (x_i > 1.0))
					if invalid.any():
						boundary_penalty_i += numpy.linalg.norm(abs(x_i[invalid]-0.5)-0.5)**2 #This is essentially the distance from the boundaries 0 or 1.
						x_i[invalid] = numpy.array(x_mean)[invalid] #take away the noise where invalid
				
				
			else:
				raise RuntimeError('Specify only one resample method (set only one to True) (Set this parameter in get_params.py)')
			
			
				
			
			self.z_offspring[i] = z_i[:]
			self.x_offspring[i] = x_i[:]
			self.evaluated_boundary_penalty[i] = boundary_penalty_i
	 		
	 		
	 		# Things to do if you already have a target vowel.
	 		if self.target:
	 			# Now use the sampled parameters to update our learner parameters (of the current target)
				self.update_learner_pars(x_i) #change
				
				# .. and see how well the parameters actually fit a class (using the ESN network (flow) from the previous step in the project 'hear')
				# the quality of the sample is given in a confidence vector for each generation (basically a sample vote from the ESN network)
				self.confidences[i] = self.environment()
				
				# get energy cost of the current learner parameters (from their distance to neutral position). A high energy cost will have a bad influence
				# on the fitness of the learner.
				self.energy_cost = norm(self.learner_pars_rel[self.target] - self.neutral_pars_rel)
			
				

			#           ^
			#           ^
			#           ^
			#           ^
			#           ^			
			#           ^
			# Called in | upper function

	def environment(self):
		"""
		Environment feeds the parameters which are being sampled in 'evaluate' into the environment.
			- speech sounds are produced using Vocal Tract Lab API.
			- these speech sounds are then 'heard' and classified by our Echo State Network which we trained in 'hear'. (flow)
			- we can then plot the actual reservoir states, which gives the user a handy way of checking on the progress of the learner
				(quality of classification is seen there) -> How 'close' we are to a sound we want to learn.
		"""
		#self.plot_learner_params()
		"""
		# Clear folder of wav etc files of last sample..
		system('rm -r '+self.output_folder+'/*')
		"""
		
		# Synthesize main (from api_class) won't accept numpy as parameters..
		if type(self.learner_pars[self.target]) == numpy.ndarray:
			self.learner_pars[self.target]=self.learner_pars[self.target].tolist()
			
		
		# Synthesize sound using VTL_API
		# -----------------------------------------------------------------------------------------------------------------------------------------
		
		# Reset synthesize instance
		
		input_dict = {	'params':self.learner_pars[self.target],
						'group_speaker':self.amb_speech.sp_group_name+' '+str(self.learner),
						'pitch_var':self.learner_pitch_rel,
						'verbose':False }
		paths = {		'wav_folder':self.output_folder}
		
		synthesize.main(input_dict,paths)
		wav_path = synthesize.wav_path
		
		
		# If the gesture was 'airtight', return 0 confidence for all targets.
		# This distinction simply skips the last part (which would also do the job for silent gestures). > Speedup!
		if not synthesize.sound_is_valid:
			
			# Return a penalty as confidence.
			penalty = numpy.zeros([len(self.ESN_std_sequence)])
			
			penalty[-1] = 1
			
			return penalty
		
		
		else:
			# Call the function, right below..
			confidence,mean_sample_vote = self.get_confidence(wav_path,plot=False)
		
		
			return confidence
			
			

				#           ^
				#           ^
				#           ^
				#           ^
				#           ^			
				#           ^
				# Called in | upper function
			
			
			
	def get_confidence(self,wav_path,plot):
		"""
		Processes sound (using functions from ambient speech) and used the ESN to produce a sample vote (plot to see what it is..), and the confidences.
		"""
		
		# Process sound
		# -----------------------------------------------------------------------------------------------------------------------------------------
		# Call a function from ambient speech to process sound for the hearing (don't dump the cochlear activation into a file, use it directly).
			# First, we need to know how the function must be called (sampling dictionary. see more infos in ambient_speech_functions)
		sampling = {'n_channels':self.amb_speech.sampling['n_channels'] , 'compressed':True} #Standard values: n_channels: 50, compressed: True
		cochlear_activation = self.amb_speech.process_sound_for_hearing(wav_path,sampling,dump=False)

		
		# Evaluate how 'good' the speech sound was (classification)
		# -----------------------------------------------------------------------------------------------------------------------------------------                  	

		sample_vote = self.flow(cochlear_activation)                       # evaluate trained self.verbose units' responses for current item

		normalize = True
		def normalize_activity(x):
			#Define a neat function which is called right below.. =)
	
			x_normalized = x.copy()
			minimum = x.min()
			maximum = x.max()
			range_ = maximum - minimum
			bias = abs(maximum) - abs(minimum)
			x_normalized -= bias/2.0
			x_normalized /= range_/2.0
			return x_normalized
		
		def shift_and_normalize_activity(x):
			
			x_shifted = x - x.min()
			x_shifted /= x.max()
			return x_shifted
		
		#sample_vote = normalize_activity(sample_vote)
		sample_vote = shift_and_normalize_activity(sample_vote)
		
		# Change sequence of nodes in the sample vote if the ESN has a special sequence of nodes 
		#	(e.g. ['null','i','u','a'] instead of self.vowels.append('null') )
		if self.ESN_sequence and not self.ESN_sequence==self.ESN_std_sequence:
			
			if len(self.ESN_sequence) != len(self.ESN_std_sequence):
				raise RuntimeError("Too many vowels initialized in ambient speech setup for this ESN you are using! Reinitialize with only those vowels before using 'learn'!")
			else:
				new = numpy.array(sample_vote)*0
				for col in range(numpy.size(sample_vote,1)): #Go throught the columns
					correct_col = self.ESN_std_sequence.index(self.ESN_sequence[col])		# ESN seq. for instance:  ['null','i','u','a']
					new[:,correct_col] = sample_vote[:,col]
				sample_vote = new
		
		
		# Average each neurons' response over time
		mean_sample_vote = numpy.mean(sample_vote, axis=0)
		
		
		# Get confidences
		# -----------------------------------------------------------------------------------------------------------------------------------------
		def get_confidence(vote):
			#Define a neat function which is called right below.. =)
	
			confidence = numpy.zeros(len(vote)) #length of the vote is n_classes
			confidence = numpy.exp(vote)
			confidence /= numpy.sum(confidence)
			return confidence

		confidence = get_confidence(mean_sample_vote)
		
		# Plot reservoir states (this is done after the learning and skipped during learning)
		# -----------------------------------------------------------------------------------------------------------------------------------------
		if plot:
			if not type(plot) == str:
				raise RuntimeError('plot must be either False or string (e.g. learnt vowel name)')
			else:
				self.plot_reservoir_states(sample_vote,plotname=plot)
		
		
		#print "Confidence = %s"%str(confidence[:])

			
		return confidence,mean_sample_vote

		
		
		
		
		
		
		"""	
		
		The generations could be assessed in a parallel way.
		(Steps: evaluation, environment, get_confidence)
		
		These steps could be written in a separate skript which would take parameters like x_mean as arguments, and return confidences.
		
		The rest (cma-es learning loop) would not be run in parallel. 
		
		On MPI run as follows:
		-------------------------------------------------------------------
		$ salloc -p sleuths -n (lambda/int) mpirun python parallel_eval_envir_confidence.py [arguments]
		-------------------------------------------------------------------
		
		
		
		
		
		
	def parallel_evaluation(self,x_mean, B_D, i_count):


		items_broadcast = x_mean,sigma,B_D,i_count,self.target_index
											# whatever the master distributes to the slaves
		tag = int(i_count/(self.n_workers-1))	# each transmission carries a specific tag to identify the corresponding slave

		#??? What is N_reservoir? Where is it defined?
		#if N_reservoir > 20:
		self.confidences = numpy.zeros([n_workers-1,n_vowels+1])
		#else:
		#	self.confidences = numpy.zeros([n_workers-1,n_vowels])
		self.energy_cost = numpy.zeros(self.n_workers-1)
		boundary_penalty = numpy.zeros(self.n_workers-1)
		z = numpy.zeros([self.population_size, self.N_dim]) #??? Explain lambda_?
		x = numpy.zeros([self.population_size, self.N_dim])
		self.N_dim_resampled = numpy.zeros([self.population_size, self.N_dim], dtype=int)

		print 'current tag (master):', tag


		for i_worker in xrange(1,self.n_workers):
			comm.send(items_broadcast, dest=i_worker, tag=tag)
		for i_worker in xrange(1,self.n_workers):

			z[i_worker-1],x[i_worker-1],self.confidences[i_worker-1],self.energy_cost[i_worker-1],boundary_penalty[i_worker-1],N_resampled[i_worker-1] = comm.recv(source=i_worker, tag=tag)

		N_resampled_sum = N_resampled.sum()
		if self.verbose:
			print N_resampled_sum, 'samples rejected'

		return z, x, self.confidences, self.energy_cost, boundary_penalty, N_resampled_sum
		"""
		
	
		"""
	def generation_sampling(self):
		"""
		#Sample and return confidences to master
		"""
		
		if not self.load_state == None:
			self.i_start = comm.recv(source=0)
		i = int(self.i_start/(self.n_workers-1))
		
		while True:
			print 'current tag (slave):', i

			x_mean,sigma,B_D,i_count,self.target_index = comm.recv(source=0, tag=i)
			if x_mean == None:
				break

			invalid = True
			if self.rank==0:
				print 'sampling parameters...'
			if self.resample:
			  N_resampled = -1
			  while invalid:
				N_resampled += 1
				z = numpy.random.randn(self.N_dim)	  # standard normally distributed vector
				x = x_mean + sigma*(numpy.dot(B_D, z))  # add mutation, Eq. 37
				invalid = (x < 0.0).any() or (x > 1.0).any()
			  self.evaluated_boundary_penalty = 0.0
			else:
			  N_resampled = 0
			  z = numpy.random.randn(self.N_dim)		  # standard normally distributed vector
			  x = x_mean + sigma*(numpy.dot(B_D, z)) # add mutation, Eq. 37
			  self.evaluated_boundary_penalty = 0.0
			  if (x<0.0).any() or (x>1.0).any(): # check boundary condition
				if self.verbose:
					print 'boundary violated. repairing and penalizing.'
				x_repaired = x.copy()	   # repair sample
				for i_component in xrange(len(x)):
					if x[i_component] > 1.0:
						x_repaired[i_component] = 1.0
					elif x[i_component] < 0.0:
						x_repaired[i_component] = 0.0
				self.evaluated_boundary_penalty = numpy.linalg.norm(x-x_repaired)**2
											# penalize boundary violation, Eq. 51
				x = x_repaired
	 
			self.update_learner_pars(x)
			self.energy_cost = numpy.fabs(self.learner_pars_rel[self.target] - self.neutral_pars_rel)


			confidences = envir.evaluate(self.learner_pars, simulation_name=self.subfolder['learn'], output_path=self.output_path,
				i_target=self.target_index, rank=self.rank, speaker=self.learner, n_vow=self.n_targets, normalize=normalize)

			if self.rank==0 and self.verbose:
				print 'z:', z, ', x:', x, ', confidences:', confidences, ', energy costs:', self.energy_cost, ', boundary penalties:',\
					self.evaluated_boundary_penalty

			send_back = z,x,confidences,self.energy_cost,self.evaluated_boundary_penalty,N_resampled
			comm.send(send_back, dest=0, tag=i)
			i += 1
		"""
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	#############################################################################################################################################
	#############################################################################################################################################
	# General functions, used to save states of learning, plot, etc....
	#############################################################################################################################################
	#############################################################################################################################################
	
	
	def get_next_target(self):
		"""
		# Get next target
		# --------------------------------------------------------------------------------------------------------
		The aim of this function is to get our next target. We don't want any already learnt targets anymore. Target
		with maximum confidence is to be chosen.
		We can't use the reward here, since we're thinking about the whole generation and all target confidences.
		"""
		# Example format for confidences:
		# confidences = numpy.array([0.2(a),0.1(e),0.3(i),0.4(null)], [next generation],...] (generation)
		# 1. Get maximal confidences across the samples in the generation, for each target.
		generation_maxima = numpy.amax(self.confidences,axis=0)
		
		unlearnt_target_indices = [i for i in range(len(self.targets)) if self.targets[i] not in self.targets_learnt]
		
		max_confidence_unlearnt = numpy.amax(generation_maxima[unlearnt_target_indices])
		
		# Get the right index out of generation_maxima
		
		self.target_index = numpy.where(generation_maxima==max_confidence_unlearnt)[0][0]
		
		next_target = self.targets[self.target_index]
		
		if next_target != self.target:
			"\nChanging objective! Found higher confidence samples for target %s!\n"%next_target
		
		self.target = next_target
		
		if self.target in self.targets_learnt:
			# This might still happen (if, e.g. more than one value of max_confidence_unlearnt is found in generation_maxima)
			# For example: generation maxima is: array([ 0.,  0.,  0.,  0.,  0.,  1.])
			# In this case, simply pick a random non-learnt target
			self.target = random.choice([t for t in self.targets if self.targets.index(t) in unlearnt_target_indices])
			self.target_index = self.targets.index(self.target)			
		
	
	
	
	
	def update_learner_pars(self,x):
		"""
		Updates the shape parameters of the learner.
		(both rel and abs)
		"""
		
		if not self.target:
			raise RuntimeError("Cannot update learner parameters! We don't know what the current target is!")
		
		params = self.target_pars_rel[self.target].copy()
		
		if self.flat_tongue:
			params[-4:-1] = 0.5
		
		params[self.i_pars_to_learn] = x[:]
		
		
		self.learner_pars_rel[self.target] = numpy.array(params[:])
		
		# Call function from ambient speech in order to transform parameters from relative to absolute parameter space
		self.learner_pars[self.target] = self.amb_speech.transf_param_coordinates('relative_to_absolute',self.learner_pars_rel[self.target],self.pars_top,self.pars_low)
		self.learner_pars[self.target] = numpy.array(self.learner_pars[self.target])
		
		
		
	
	
	def save_snapshot(self,sigma,fitness):
	
		"""
		Save a snapshot of current target learning progress 
		- graphic, and pickle dump!
		(In the result folder..)
		"""
		self.target_index = self.targets.index(self.target)
		
		
		# Put rewards of sub -0.1 values to -0.1. (fitness is -reward)
		fitness = numpy.array(fitness)
		fitness[fitness>0.1] = 0.1
		
		# Save state...
		# ---------------------------------------------------------------------
		self.reward_history[self.target].append(-numpy.amin(fitness))
		self.sigma_history[self.target].append(sigma)
		self.learner_pars_rel_history[self.target].append(self.learner_pars_rel[self.target])
		self.learner_pars_history[self.target].append(self.learner_pars[self.target])
		
		# Save graphic
		# ----------------------------------------------------------------------
		snapshot = plt.figure()
		try:
			graph = snapshot.add_subplot(111)
			threshold = [self.convergence_thresholds[self.target]] * len(self.iteration_stages[self.target_index]) # A bar in the graph, indicating the reward threshold.
			graph.plot([x+5 for x in self.iteration_stages[self.target_index]],self.reward_history[self.target],'ro-',label='reward')
			graph.plot([x+5 for x in self.iteration_stages[self.target_index]],threshold,'r--',label='reward_threshold')
			graph.plot([x+15 for x in self.iteration_stages[self.target_index]],self.sigma_history[self.target],'bd-',label='sigma')
			for i in self.i_pars_to_learn:
				graph.plot(self.iteration_stages[self.target_index],numpy.array(self.learner_pars_rel_history[self.target]).T[i],'.--',alpha=0.3)
			plt.xlabel('Iteration Number')
			plt.ylabel('Value')
			graph.set_title('CMA-ES Parameters over Generation Number')
			plt.legend()
			snapshot.savefig(self.result_folder+'/snapshot/snapshot_for_target_%s.png'%self.target)
		except ValueError:
			debug()
			
		plt.close()
		
		
		# Save state (pickle)
		# ----------------------------------------------------------------------
		self.output_dict = {	'learner parameters relative':self.learner_pars_rel_history,
								'rewards':self.reward_history,
								'sigma':self.sigma_history,
								'current_peak':self.peak							}
						
		f = open(self.result_folder+'/history.pickle', 'w+')
		f.truncate()
		pickle.dump(self.output_dict,f)
		f.close()
		
		
		
		
		
		
		
	def new_peak(self,fitness):
		"""
		Check if one of the current samples is the best sample ever of that target.
		If yes, output to data path.
		"""
		
		# Which sample is the highest? (remember, fitness is defined negatively)
		i_argmax = fitness.argmin()
		
		# Take the parameters used for that sample as x_mean.
		x_mean = self.x_offspring[i_argmax][:]
	
		self.update_learner_pars(x_mean)
		
		if type(self.learner_pars[self.target]) == list:
			self.peak[self.target] = (-fitness[i_argmax],self.learner_pars[self.target])  # Initialisation was a tuple: (0,numpy.zeros([len(self.target_pars[vowel])]))
		else:
			self.peak[self.target] = (-fitness[i_argmax],self.learner_pars[self.target].tolist())
		
		# Save in data folder
		input_dict = {	'params':self.peak[self.target][1], #The current peak parameters
							'group_speaker':self.amb_speech.sp_group_name+' '+str(self.learner), 
							'pitch_var':self.learner_pitch_rel,
							'verbose':False }
		
		paths = {		'wav_folder':self.output_folder+'/current_peak', 'wav_path':'%s/current_peak/peak_%s.wav'%(self.output_folder,self.target)}
		
		synthesize.main(input_dict,paths)
		
		
	
		
	
	
	def plot_reservoir_states(self, sample_vote,plotstring='unknown'):
		"""
		Plot the states of the reservoir (and the classification strengths). This function is very similar to 'plot_prototypes' in hear_functions.
		
		New: Changed completely, made simpler.
		"""
		
		debug()
		
		f = plt.figure()
		plt.title("Class activations using (adult|infant) ESN")
		f_s  = plt.subplot(111)
		f_s.matshow(sample_vote.T)
		plt.ylabel("Class")
		plt.xlabel('')
		ylabels = self.vowels[:]
		ylabels.append('null')
		plt.yticks(range(self.n_vowels), ylabels)
		plt.xticks(range(0, 35, 5), numpy.arange(0.0, 0.7, 0.1))
		f.savefig(self.result_folder+'/Class_activations_learnt_'+plotstring)
		
		
