


# Class imports
	# Import the class containing the parameters and arguments for this function.
from parameters.get_params import parameters as params

	# Import VTL_API needed.
from src.VTL_API.api_class import VTL_API_class
	#	.. make a class instance, or: "synthesize is-a VTL_API_class"
	#	Call synthesize like this: synthesize.main(input_dict,paths)
synthesize = VTL_API_class()


# General imports
import os
from os import path,system,listdir
import matplotlib
matplotlib.use('Agg')				  # for use on clusters
import numpy
from numpy.linalg import norm
import argparse

import Oger
import pylab

from brian import kHz, Hz, exp, isinf
from brian.hears import Sound, erbspace, loadsound, DRNL
from scipy.signal import resample

#from mpi4py import MPI #now use simpler parallel computing from joblib
from joblib import Parallel, delayed  
import multiprocessing

import datetime
from datetime import date
from time import *

from collections import deque

import cPickle
import pickle
import random

# Python debugger
from pdb import set_trace as debug



class functions(params):
	"""
	Provides functions, called in "src/learn/learn.py"
	Includes main learning function for the Echo State Network.
	"""

	
	#############################################################################################################################################
	# Functions called directly from 'learn' ####################################################################################################
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
		
			# Targets to be learnt (vowels)
		if self.targets == "all":
			self.targets = self.vowels[:]
			if '@' in self.targets:
				self.targets.pop(self.targets.index('@'))
		
		# Targets will be moved to this array when learnt:
		self.targets_learnt = []
		self.n_targets = len(self.targets)
			
		# Now that we've finished with the targets, add schwa to vowels.
		if not '@' in self.vowels:
			self.vowels.append('@')
		# That mean, we want /schwa/ in vowels but not in targets!	
		
		
			# 1. Take over target parameters (dict format) of all vowels from the ambient speech setup.
			# 2. Get schwa parameters as neutral position (used for energy cost etc.)
		self.target_pars = dict()
		self.target_pars_rel = dict()
		for vowel in self.targets:
			self.target_pars[vowel] = self.amb_speech.pars[vowel][0,:]
			self.target_pars_rel[vowel] = self.amb_speech.pars_rel[vowel][0,:]
		
			# Get schwa parameters as neutral position (used for energy cost etc.)
		self.neutral_pars = self.amb_speech.pars['@'][0,:]
		self.neutral_pars_rel = self.amb_speech.pars_rel['@'][0,:]
		
			# look up dictionary for parameter indices
		self.par_names = self.amb_speech.par_names
		
		
		# Take over the parameter-coordinate-system-transformation-function from ambient speech setup.
			# This includes taking over upper and lower boundaries for the vowel parameters.
		self.pars_top = self.amb_speech.pars_top[0][:]
		self.pars_low = self.amb_speech.pars_low[0][:]

		
		# Parameters from ambient speech
			# Get the relative pitch of our learner (will be added to abs pitch 52)
		if type(self.learner)==int:
			# the number of the learner is also the right index in speaker_pitch_rel - list.
			index = self.learner 
			self.learner_pitch_rel = self.amb_speech.speaker_pitch_rel[index]
		else:
			self.learner_pitch_rel = 5###!!!
		
		
		# Verbosity
		self.verbose = self.be_verbose_in['learn']
		
		# Initialize paths
		# ---------------------------------------------------------------------------------------------------------
			# We have the same main folder, so..
		self.base_path = self.amb_speech.base_path
			#Get the path to the .flow file of the current auditory system
		self.ESN_path = self.hear.ESN_output_path
			#Store general output, trained speaker etc. in an output_folder
		self.output_folder = self.base_path+'/data/output/learn'
			#Store results of the reinforement learning
		self.result_folder = self.base_path+'/results/learn'
		if self.subfolder['learn']:
			self.output_folder = self.output_folder+'/'+self.subfolder['learn']
			self.result_folder = self.output_folder+'/'+self.subfolder['learn']
			
			# Where to find the learner (speaker file +'.speaker')
		if type(self.learner)==int:
			self.learner_path = self.base_path+'/data/speakers_gestures/srangefm/'+str(self.learner)
		else:
			self.learner_path = self.base_path+'/data/speakers_gestures/standard/'+self.learner
		
		
	def setup_folders(self):
		"""
		Sets up the appropriate folders for the group "self".
		(.. and cleans them up.)
		"""
		
		if self.rank==0:
			#Put the learner into a separate folder
			#-----------------------------------------------------
			old_learner_path = self.learner_path
			self.learner_path = self.base_path+'/data/speakers_gestures/learner/'+str(self.learner)
			system('cp '+old_learner_path+'.speaker '+self.learner_path+'.speaker')
			
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
			
			#Create folder for results (plots etc.)
			#-----------------------------------------------------
			if not path.isdir(self.result_folder):
				system('mkdir --parents '+self.result_folder)
			elif not(listdir(self.result_folder) == []):
				system('rm -r '+self.result_folder+'/*')
				
			
			
	
		"""
	def open_result_write(self):
		outputfile = outputfolder+'out.dat'
		results = outputfolder+'results.dat'
		os.system('rm '+outputfile)   
		os.system('rm '+results)
		os.system('touch '+outputfile)
		os.system('touch '+results)
		self.output_write = open(outputfile, 'w')
		self.results_write = open(results, 'w')
		
	def close_result_write(self):
		if self.rank==0:
			self.output_write.close()
			self.results_write.close()
			for record in records:
				record.close()
		"""
				
	
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
		- Change values where we want to learn to /@/ values
		"""
		self.learner_pars = self.target_pars.copy()
		self.learner_pars_rel = self.target_pars_rel.copy()
		
		
		for vowel in self.targets:
			self.learner_pars[vowel][self.i_pars_to_learn] = self.neutral_pars[self.i_pars_to_learn]
			self.learner_pars_rel[vowel][self.i_pars_to_learn] = self.neutral_pars_rel[self.i_pars_to_learn]
			
	
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
				#  dimenions (see Hansen)					
			self.population_size = lambda_list[self.N_dim-1]
	
	
	def save_state(self,state, flag):
		save_file = self.output_folder+'save.'+flag
		os.system('rm '+save_file)
		os.system('touch '+save_file)
		save_file_write = open(save_file, 'w')
		cPickle.dump(state, save_file_write)
		save_file_write.close()
	


	def load_saved_state(self):
		inputfile_dynamic = open(self.load_state+'.dyn', 'r')
		inputfile_static = open(self.load_state+'.stat', 'r')
	   
		self.static_params = cPickle.load(inputfile_static)
		self.dynamic_params = cPickle.load(inputfile_dynamic)

		inputfile_static.close()
		inputfile_dynamic.close()

	
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
																	#
																	#
																	#
																	#	
	#############################################################################################################################################
	# Small function for cmaes and generation_sampling ##########################################################################################
	#############################################################################################################################################
	
	def get_next_target(self):
		# Get next target
		# --------------------------------------------------------------------------------------------------------
		# Example format for confidences:
		# confidences = numpy.array([0.2(a),0.1(e),0.3(i),0.4(o)], [next generation],...] (generation)
		# 1. Get maximal confidences across the samples in the generation, for each vowel.
		generation_maxima = numpy.amax(self.confidences,axis=0)
		
		# 2. Get all learnt indices
		i_learnt = [i for i in range(len(self.vowels)) if self.vowels[i] in self.targets_learnt]
		
		# 3. Get the highest non-learnt confidence, and make the index our new target index.
		generation_maxima[i_learnt] = 0
		self.target_index = generation_maxima.tolist().index(max(generation_maxima))
		
		self.target = self.targets[self.target_index]
		
		if self.verbose:
			print 'confidences:', self.confidences
			print 'targets_learnt:', self.targets_learnt
			print 'next target:', self.target
	
	
	
	
	
	def update_learner_pars(self,x):
		"""
		Updates the shape parameters of the learner.
		(both rel and abs)
		"""
		if not self.target:
			raise RuntimeError("Cannot update learner parameters! We don't know what the current target is!")
		params = self.target_pars_rel[self.target]
		if self.flat_tongue:
			for i in xrange(-4,0):
				params[i] = 0.5

		for i in xrange(len(x)):
			params[self.i_pars_to_learn[i]] = x[i]
		
		
		self.learner_pars_rel[self.target] = params
		
		# Call function from ambient speech in order to transform parameters from relative to absolute parameter space
		self.learner_pars[self.target] = self.amb_speech.transf_param_coordinates('relative_to_absolute',self.learner_pars_rel[self.target],self.pars_top,self.pars_low)
		
		
		
		
	def plot_reservoir_states(self, flow, sample_vote):
		"""
		Plot the states of the reservoir (and the classification strengths). This function is very similar to 'plot_prototypes' in hear_functions.
		"""
		
		
		# reservoir activity for most recent item
		current_flow = flow[0].inspect()[0].T
		# reservoir size
		N = flow[0].verbose_dim
		
		n_subself.plots_x, n_subself.plots_y = 2, 1   # arrange two self.plots in one column
		pylab.subplot(n_subself.plots_x, n_subself.plots_y, 1)
		                                    # upper plot
		y_min = y.min()
		y_max = y.max()
		if abs(y_min) > y_max:              # this is for symmetrizing the color bar
			vmin = y_min                    # -> 0 is always shown as white
			vmax = -y_min
		else:
			vmax = y_max
			vmin = -y_max

		class_activity = pylab.imshow(y.T, origin='lower', cmap=pylab.cm.bwr, aspect=10.0/(n_vow+1), interpolation='none', vmin=vmin, vmax=vmax)
		                                    # plot self.verbose activations, adjust to get uniform aspect for all n_vow
		pylab.title("Class activations")
		pylab.ylabel("Class")
		pylab.xlabel('')
		ylabels = self.vowels[:]
		ylabels.append('null')
		pylab.yticks(range(self.n_vowels), ylabels)
		pylab.xticks(range(0, 35, 5), numpy.arange(0.0, 0.7, 0.1))
		cb = pylab.colorbar(class_activity)

		n_subself.plots_x, n_subself.plots_y = 2, 1
		pylab.subplot(n_subself.plots_x, n_subself.plots_y, 2)
		                                    # lower plot
		current_flow_min = current_flow.min()
		current_flow_max = current_flow.max()
		if abs(current_flow_min) > current_flow_max:
		        vmin_c = current_flow_min   # symmetrizing color, see above
		        vmax_c = -current_flow_min
		else:
		        vmax_c = current_flow_max
		        vmin_c = -current_flow_max

		reservoir_activity = pylab.imshow(current_flow, origin='lower', cmap=pylab.cm.bwr, aspect=10.0/N, interpolation='none', vmin=vmin_c, vmax=vmax_c)
		                                    # plot reservoir states of current prototype,
		                                    #  adjust to get uniform aspect for all N
		pylab.title("Reservoir states")
		pylab.xlabel('Time (s)')
		pylab.xticks(range(0, 35, 5), numpy.arange(0.0, 0.7, 0.1))
		pylab.ylabel("Neuron")
		pylab.yticks(range(0,N,N/7))
		cb2 = pylab.colorbar(reservoir_activity)

		pylab.savefig(self.output_folder+'/plots/vowel_'+self.target+'_'+str(self.rank)+'.pdf')

		pylab.close('all')
	
	
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
																	#
																	#
																	#
																	#	
	#############################################################################################################################################
	# MASTER FUNCTION ###########################################################################################################################
	#############################################################################################################################################
	
	
	def cmaes(self):						# actual CMA-ES part
		"""
		Documentation?
		"""


		

		
		#######################################################
		# Initialization
		#######################################################
		
		
		# sigma_0 is the initial sigma (always stays the same). 
		# The learner will reset every time he runs into a local
		# minimum. (during conversion, 'current_sigma' gets smaller
		# and smaller. Then, after reset, sigma (the starting value of
		# current_sigma) can be altered. Thus, we have 3 sigmas, of 
		# which sigma_0 will always stay the same, sigma changes slowly 
		# with every reset. Current sigma is changed after every generation.
		# For now, we set both sigma and current_sigma to sigma_0
		sigma = self.sigma_0
		current_sigma = sigma
		
		# initialize x_mean. x_mean is the current state of our learnt parameters.
		# The learner_pars will be updated using x_mean.
		# Offspring will be generated around xmean with a gaussian distribution of
		# sigma!
		# For initialisation: Doesn't matter which vowel.. Used /schwa/ here.
		# Convert to list (from numpy), in order to make it of type deque() 
		# deque: (fast datatype, like list)
		x_mean = self.neutral_pars_rel[self.i_pars_to_learn]
		x_mean = deque(x_mean.tolist())
		
		# recent x_mean
		x_recent = deque()
		x_recent.append(x_mean)
		
		# recent fitness
		fitness_recent = deque()
		
		mu_ = self.population_size / 2.0			 # mu_ is float
		mu = int(numpy.floor(mu_))		 # mu is integer = number of parents/points for recombination
		weights = numpy.zeros(mu)
		for i in xrange(mu):			# muXone recombination weights
			weights[i] = numpy.log(mu_+0.5) - numpy.log(i+1)
		weights /= sum(weights)		 # normalize recombination weights array 
		mu_eff = sum(weights)**2 / sum(weights**2)
										# variance-effective size of mu
		
		# window for convergence test
		convergence_interval = int(10+numpy.ceil(30.0*self.N_dim/self.population_size))
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

		# Initialize arrays
		fitness = numpy.zeros(self.population_size)


		#######################################################
		# Output preparation
		#######################################################

		"""
		self.output_write.write('initial conditions: time=('+str(datetime.datetime.now())+') N='+str(self.N_dim)+', lambda='+str(self.population_size)+', x=')
		for x_ in x_mean:
			self.output_write.write(str(x_)+' ')
		self.output_write.write(', distance='+str(norm(self.learner_pars_rel-self.target_pars_rel[vowel])))
		self.output_write.write(', sigma='+str(sigma))
		self.output_write.write(', energy_factor='+str(self.energy_factor))
		self.output_write.write(', alpha='+str(self.alpha))
		self.output_write.write(', convergence_threshold='+str(self.convergence_threshold))
		self.output_write.write(', conditioning_maximum='+str(self.conditioning_maximum))
		self.output_write.write('\n')
		self.output_write.write('time	sampling step   mean fitness   i_target   ')
		for i in xrange(self.n_targets):
			self.output_write.write('confidence['+str(i)+']   ')
		for i in xrange(self.n_targets):
			self.output_write.write('motor deviation['+str(i)+']   ')
		self.output_write.write('energy_cost boundary_penalty sigma N_resampled\n')
		"""
		"""
		self.static_params = [self.N_dim, self.verbose, self.learner_pars_rel, self.i_pars_to_learn, self.sigma_0, self.target_pars_rel, self.convergence_threshold, self.intrinsic_motivation, 
			self.n_targets, self.energy_factor, self.alpha, self.output_path, self.random_restart, self.conditioning_maximum, self.flat_tongue]
		self.save_state(self.static_params, 'stat')
		"""

		#######################################################
		# Generation Loop
		#######################################################

		error = False
		fitness_mean = 0.0

		i_count = 0

		"""
		if not self.load_state == None:
		  print 'loading state', self.load_state
		  [current_time, i_count, self.target_index, p_s , p_c, C, i_eigen, sigma, x_recent, fitness_recent, i_reset, current_sigma, self.learner_pars,\
			self.targets_learnt, B_D, x_mean] = self.dynamic_params
		  print 'current_time:', current_time, '\ni_count:', i_count, '\ni_target:', self.target_index, '\np_s:', p_s, '\np_c:', p_c, '\nC:', C,\
			'\ni_eigen:', i_eigen, '\nsigma:', sigma, '\nx_recent:', x_recent, '\nfitness_recent:', fitness_recent, '\ni_reset:',\
			i_reset, '\ncurrent_sigma0:', current_sigma, '\nlearner_parameters:', x_learnt, '\nlearnt_targets:', self.targets_learnt, '\nB_D:', B_D,\
			'\nx_mean:', x_mean
		  self.i_start = i_count
		  print 'i_start = i_count =', self.i_start
		  for i_worker in xrange(1,self.n_workers):
			  comm.send(self.i_start, dest=i_worker)
		"""

		return_dict = dict()

		t_0 = datetime.datetime.now()
		t_reset = t_0
		
		# Main loop:
		# 
		while True:

			# Generate and evaluate lambda offspring.
			# Resulting confidences will be zero-valued if we don't yet have a target.
			# ---------------------------------------------------------------------------------------------------------------------------
			self.evaluation(x_mean,sigma, B_D)
			# do the following way, for more than one workers.. (if self.n_workers > 1:)
	  		#self.z_offspring, self.x_offspring, self.confidences, self.energy_cost, self.evaluated_boundary_penalty, N_resampled_trial = self.parallel_evaluation(x_mean, B_D)
				
			
			# Each offspring sample counts toward the total count.
			i_count += self.population_size
	  		
	  		
	  		
			# If we have a target vowel (e.g. if we're aiming for /a/), compute the fitness (reward) of the learner parameters.
			# ---------------------------------------------------------------------------------------------------------------------------
			if self.target:
												# These values we all got from self.evaluation.
				fitness = -self.confidences.T[self.target_index]+self.energy_factor*self.energy_cost+self.alpha*self.evaluated_boundary_penalty
				
			
			
			
			# If we don't yet have a target vowel, get one! Now we have to start from the beginning again..
			# ---------------------------------------------------------------------------------------------------------------------------
			if not self.target:
				idle = True
				self.get_next_target()
			
				
				
			
			# If we are below the convergence threshold.. (Found a local minimum below the last one!)..
			# ---------------------------------------------------------------------------------------------------------------------------
			if (fitness < -self.convergence_threshold).any(): #(and self.no_convergence_criterion:)
				i_argmax = fitness.argmin()
				x_mean = self.x_offspring[i_argmax]
				self.targets_learnt.append(self.target)
				
				
				raw_input("Found target "+self.target+"!! Continue?")
				
				self.update_learner_pars(x_mean)
				
				"""
				self.results_write.write(self.target+'	'+str(i_count)+'	'+str(-fitness[i_argmax])+'	'+strftime("%d %b %H:%M:%S", localtime())+'\n')
				self.results_write.write('relative coordinates:\n '+str(self.learner_pars_rel)+'\n')
				self.results_write.write('absolute coordinates:\n '+str(self.learner_pars)+'\n\n')
				self.results_write.flush()
				"""

			
				return_dict[self.target+'_steps'] = i_count-i_reset
				return_dict[self.target+'_time'] = datetime.datetime.now()-t_reset
				return_dict[self.target+'_reward'] = -fitness[i_argmax]

				i_reset = i_count
				t_reset = datetime.datetime.now()

				
				print 'iteration:',i_count,', now:', datetime.datetime.now(), ', i_target:', self.target_index, ', reward:', -fitness[i_argmax], ', parameter:',self.x_offspring[i_argmax]
				
				pdb.set_trace()
				
				# We've reached a target. Now check if there are still some to learn.
				if not(self.targets==self.targets_learnt) and self.intrinsic_motivation:
					self.target = False
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
					tag = int(i_count/(self.n_workers-1))
					for i_worker in xrange(1,self.n_workers):
						comm.send((None,None,None,None,None), dest=i_worker, tag=tag)
					break
			
			
			
			
			
			
			
			
			
			
			
			# If we are NOT below the convergence threshold.. (The last minimum was better)
			# ---------------------------------------------------------------------------------------------------------------------------	
			else:

				# Sort by fitness and compute weighted mean into x_mean
				indices = numpy.arange(self.population_size)
				to_sort = zip(fitness, indices)
											# minimization
				to_sort.sort()
				fitness, indices = zip(*to_sort)
				fitness = numpy.array(fitness)
				indices = numpy.array(indices)
				x_mean = numpy.zeros(self.N_dim)
				z_mean = numpy.zeros(self.N_dim)
				fitness_mean = 0.0
				for i in xrange(mu):
					x_mean += weights[i] * self.x_offspring[indices[i]]
											# recombination, Eq. 39
					z_mean += weights[i] * self.z_offspring[indices[i]]
											# == D^-1 * B^T * (x_mean-x_old)/sigma
					fitness_mean += weights[i] * fitness[indices[i]]
				
				for i in xrange(len(x_mean)):
					self.learner_pars_rel[self.i_pars_to_learn[i]] = x_mean[i]

				#self.get_('deviations')

				# Output
				self.update_learner_pars(x_mean)
				
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
			   
				C_new = (1.0-c_1-c_mu)*C + c_1*(numpy.dot(p_c,p_c.T) + (1.0-h_sig)*c_c*(2.0-c_c)*C) + c_mu*numpy.dot(numpy.dot((numpy.dot(B_D, self.z_offspring[indices[:mu]].T)),numpy.diag(weights)),(numpy.dot(B_D, self.z_offspring[indices[:mu]].T)).T)

				if not (numpy.isfinite(C_new)).all():
					print 'Warning! C contains invalid elements!'
					error = True
				else:
					C = C_new			   # regard old matrix plus self.rank one update plus minor correction plus self.rank mu update, Eq. 43
	
				# Adapt step-size sigma
				sigma = sigma * numpy.exp((c_s/damps) * (numpy.linalg.norm(p_s)/chi_N - 1.0))
											# Eq. 41

				# Update B and D from C
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


				# Escape flat fitness, or better terminate?
				print 'fitness:', fitness
				if fitness[0] == fitness[int(numpy.ceil(0.7*self.population_size))]:
					sigma *= numpy.exp(0.2+c_s/damps)
					print 'warning: flat fitness, consider reformulating the objective'


				while len(x_recent) > convergence_interval - 1:
					x_recent.popleft()
				while len(fitness_recent) > convergence_interval - 1:
					fitness_recent.popleft()
				x_recent.append(x_mean)
				fitness_recent.append(fitness_mean)

				cond = numpy.linalg.cond(C)


				if self.no_reward_convergence:
					termination = (numpy.ptp(x_recent, axis=0) < self.ptp_stop).all() or (cond > self.conditioning_maximum)
				else:
					termination = ((numpy.ptp(x_recent, axis=0) < self.ptp_stop).all()) and (numpy.ptp(fitness_recent) < self.ptp_stop) or (cond > self.conditioning_maximum)

				if termination:
					print 'convergence criterion reached.'
					if (fitness[0] > -self.convergence_threshold): # confidence worse than desired
						print 'reward too low. resetting sampling distribution.'
						print 'reward', -fitness[0], '<', self.convergence_threshold
						p_c = numpy.zeros(self.N_dim)		
						p_s = numpy.zeros(self.N_dim)		
						B = numpy.eye(self.N_dim)			 
						D = numpy.eye(self.N_dim)
						B_D = numpy.dot(B, D)		   
						C = numpy.dot(B_D, (B_D).T)
						i_eigen = 0

						if self.random_restart:
							if current_sigma < 0.9 and not self.keep_sigma_constant:
								current_sigma += 0.05			  
							sigma = current_sigma
							print 'sigma set to', sigma
					
							random_learnt_target = random.choice(self.targets_learnt)
							x_mean = self.learner_pars[random_learnt_target]
							print 'agent chose to restart search from learnt parameters of '+random_learnt_target

						else:
							if current_sigma < 0.9 and not self.keep_sigma_constant:
								current_sigma += 0.1			  
							sigma = current_sigma
							print 'sigma set to', sigma


					else:
						print 'confidence:', -fitness[0], ', i_reset:', i_reset
						self.targets_learnt.append(self.target)
	
						if (len(self.targets_learnt) < self.n_targets+1) and self.intrinsic_motivation:
							self.target = False
							p_c = numpy.zeros(self.N_dim)		
							p_s = numpy.zeros(self.N_dim)		
							B = numpy.eye(self.N_dim)			 
							D = numpy.eye(self.N_dim)		   
							C = numpy.dot(B_D, (B_D).T)	
							i_eigen = 0
							current_sigma = self.sigma_0			   
							sigma = current_sigma
							i_reset = 0
						else:
							print 'terminating.'
							print 'i_reset:', i_reset, ', confidence:', -fitness[0]
							tag = int(i_count/(self.n_workers-1))
							for i_worker in xrange(1,self.n_workers):
								comm.send((None,None,None,None,None), dest=i_worker, tag=tag)
							break
			
			
			# Do the following for all generations!
			# ---------------------------------------------------------------------------------------------------------------------------
			if error:
				print 'Critical error occurred!\nTerminating simulations.'
				tag = int(i_count/(self.n_workers-1))
				for i_worker in xrange(1,self.n_workers):
					comm.send((None,None,None,None,None), dest=i_worker, tag=tag)
				break
			if self.verbose:
				print 'x:', self.x_offspring
			print 'iteration:',i_count,', time elapsed:', datetime.datetime.now()-t_0, ', target:', self.target, ', reward:', -fitness_mean, ', parameter:',x_mean
			
			"""
			self.save_state(self.dynamic_params, 'dyn')
			"""
			
		# At the end..
		# ---------------------------------------------------------------------------------------------------------------------------
		return_dict['time'] = datetime.datetime.now()-t_0
		return_dict['steps'] = i_count

		x_min = x_mean
		if self.verbose:
			print 'x:', self.x_offspring
			print 'x_mean:', x_mean
		
		
		#		input_dict = {	'gesture':vowel,
		#						'group_speaker':group_speaker,
		#						'pitch_var':self.speaker_pitch_rel[speaker],
		#						'verbose':True }
		#		paths = {		'input_folder':self.speaker_path,
		#						'wav_folder':self.output_path+'/'+vowel }
		#		
		#		synthesize.main(input_dict,paths)
		#		
		#		create_speaker_finish(speaker, outputfolder) #??? How to do this?
		
		return return_dict
	
	
	#############################################################################################################################################





















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

		
		
		self.confidences = numpy.zeros([self.population_size,self.n_targets+1])
		self.energy_cost = numpy.zeros(self.population_size)
		self.evaluated_boundary_penalty = numpy.zeros(self.population_size)
		self.z_offspring = numpy.zeros([self.population_size, self.N_dim])
		self.x_offspring = numpy.zeros([self.population_size, self.N_dim])
		N_resampled = -self.population_size
		
		for i in xrange(self.population_size):   # offspring generation loop

			invalid = True
			print 'sampling parameters...'
			if self.resample:
				while invalid:
					N_resampled += 1
					z_i = numpy.random.randn(self.N_dim)	  # standard normally distributed vector
					x_i = x_mean + sigma*(numpy.dot(B_D, z_i))  # add mutation, Eq. 37
					invalid = (x_i < 0.0).any() or (x_i > 1.0).any()

				boundary_penalty_i = 0.0
			else:
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
					x_i = x_repaired

			self.z_offspring[i] = z_i
			self.x_offspring[i] = x_i
			self.evaluated_boundary_penalty[i] = boundary_penalty_i
	 		
	 		
	 		# Things to do if you already have a target vowel.
	 		if self.target:
	 			# Now use the sampled parameters to update our learner parameters (of the current target)
				self.update_learner_pars(x_i)
				# .. and see how well the parameters actually fit a class (using the ESN network (flow) from the previous step in the project 'hear')
				# the quality of the sample is given in a confidence vector for each generation (basically a sample vote from the ESN network)
				self.confidences[i] = self.environment()
				
				# get energy cost of the current learner parameters (from their distance to neutral position). A high energy cost will have a bad influence
				# on the fitness of the learner.
				self.energy_cost = norm(self.learner_pars_rel[self.target] - self.neutral_pars_rel)
			
			
	
			
			












	def environment(self):
		"""
		Environment feeds the parameters which are being sampled in 'evaluate' into the environment.
			- speech sounds are produced using Vocal Tract Lab API.
			- these speech sounds are then 'heard' and classified by our Echo State Network which we trained in 'hear'. (flow)
			- we can then plot the actual reservoir states, which gives the user a handy way of checking on the progress of the learner
				(quality of classification is seen there) -> How 'close' we are to a sound we want to learn.
		"""
		
		# Synthesize sound using VTL_API
		# -----------------------------------------------------------------------------------------------------------------------------------------
		
		input_dict = {	'gesture':self.target,
						'params':self.learner_pars[self.target],
						'learning_speaker':self.amb_speech.sp_group_name+' '+str(self.learner),
						'pitch_var':self.learner_pitch_rel,
						'verbose':False }
		paths = {		'input_path':self.learner_path,
						'wav_folder':self.output_folder }
		
		valid = synthesize.main(input_dict,paths)
		wav_path = synthesize.wav_path
		
		"""
		# This part is commented, because there's no need for it. The idea was to return zero confidences for a silent sample. 
		# But since we included some silent samples in the null classes in ambient speech, those will be classified as null and
		# a low confidence will be returned anyway. This makes everything a bit more realistic, since we're not simply skipping 
		# the ESN part...
		# If the gesture was 'airtight', return 0 confidence for all targets.
		if not valid:
			return numpy.zeros([self.n_targets+1])
		"""
		
		# Process sound
		# -----------------------------------------------------------------------------------------------------------------------------------------
		# Call a function from ambient speech to process sound for the hearing (don't dump the cochlear activation into a file, use it directly).
			# First, we need to know how the function must be called (sampling dictionary. see more infos in ambient_speech_functions)
		sampling = {'n_channels':self.amb_speech.sampling['n_channels'] , 'compressed':True} #Standard values: n_channels: 50, compressed: True
		cochlear_activation = self.amb_speech.process_sound_for_hearing(wav_path,sampling,dump=False)
		
		
		# Evaluate how 'good' the speech sound was (classification)
		# -----------------------------------------------------------------------------------------------------------------------------------------

		# Get flow (classifier)
		flow_file = open(self.ESN_path, 'r')    
		flow = cPickle.load(flow_file)      	
		flow_file.close()                   	

		sample_vote = flow(cochlear_activation)                       # evaluate trained self.verbose units' responses for current item
		
		# Normalize activity?
		normalize = False
		def normalize_activity(x):
			"""Define a neat function which is called right below.. =)"""
			x_normalized = x.copy()
			minimum = x.min()
			maximum = x.max()
			range_ = maximum - minimum
			bias = abs(maximum) - abs(minimum)
			x_normalized -= bias/2.0
			x_normalized /= range_/2.0
			return x_normalized
		
		if normalize:
			sample_vote = normalize_activity(sample_vote)
		
		# Average each neurons' response over time
		mean_sample_vote = numpy.mean(sample_vote, axis=0)
		
		
		# Get confidences
		# -----------------------------------------------------------------------------------------------------------------------------------------
		
		def get_confidence(vote):
			"""Define a neat function which is called right below.. =)"""
			confidence = numpy.zeros(len(vote)) #length of the vote is n_classes
			confidence = numpy.exp(vote)
			confidence /= numpy.sum(confidence)
			return confidence
		
		confidence = get_confidence(mean_sample_vote)
		
		# Plot reservoir states
		# -----------------------------------------------------------------------------------------------------------------------------------------
		#self.plot_reservoir_states(flow, sample_vote)
		
		return confidence

		
		
		
		
		
		
		
		
		
		
		
		
		
		
		
		
		
		
		
		
		
		
		
		
		
		
		
		
		
		
		
		
		
		
		
		
		
		
		
		
		
		
		
		
		
		"""	
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
	#############################################################################################################################################
	# SLAVE FUNCTION ############################################################################################################################
	#############################################################################################################################################
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
		
		
