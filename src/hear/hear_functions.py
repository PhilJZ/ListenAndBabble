


# Class imports
# -------------------------------------------------------
	# Import the class containing the parameters and arguments for this function.
from parameters.get_params import parameters as params


# (From older version): Module imports
# -------------------------------------------------------
	# Confusion Matrix was not altered (original state, programmed by Murakami)
from src.hear.confusion_matrix import ConfusionMatrix

# General imports
# -------------------------------------------------------
from datetime import date
import os
from os import system
import cPickle
import pickle						# the difference to cPickle: slower! but can handle classes.
import gzip
import random
import mdp
import scipy as sp
import numpy
import random
import pdb

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

#global Oger
import Oger

#global pylab
import pylab

from pdb import set_trace as debug


class functions(params):
	"""
	Provides functions, called in "src/hear/hear.py"
	Includes main learning function for the Echo State Network.
	"""
	


	
#######################################################################################################################	
# SETUP FUNCTIONS                                   ###################################################################
#######################################################################################################################
	
	def __init__(self):
		"""
		Initialize from params:
		Import all relevant parameters as self. (from 'params.variable' to 'self.variable')
		Get hear and ambient speech parameters (ambient speech parameters are also used when hearing ambient speech)
		"""		
		# Import all relevant parameters/functions as self. (from 'params.variable' to 'self.variable')
		params.__init__(self)
		self.get_hear_params()
		self.get_ambient_speech_params()
		
		# Reload required class instances from previous steps
		# ---------------------------------------------------
		self.amb_speech = pickle.load(file('data/classes/ambient_speech_instance.pickle','r'))
		
		
		
		# Initialize group specific parameters
		self.n_speakers = int(self.size) if self.speakers=="all" else len(self.speakers)#size of the group (how many tot. speakers)
		
		self.speakers = self.amb_speech.speakers[:]
		
		
		self.vowels = self.amb_speech.vowels[:]
		self.vowel_data_n = self.amb_speech.sampling['n_samples']
		self.null_data_n = self.amb_speech.sampling_null['n_samples']
		
		if '@' in self.vowels:
			self.vowels.pop(self.vowels.index('@'))
		
		# Total number of vowels learned
		self.n_vowels= len(self.vowels)
		
		# Get number of trained networks
		self.n_trains = self.n_workers * self.trains_per_worker
		
		# Prepare lists for errors of each network size
		self.errors = dict()
		self.errors['leaky'] = numpy.zeros([self.trains_per_worker, len(self.reservoir_sizes)])
		if self.do_compare_leaky:
			self.errors['non-leaky'] = self.errors['leaky'].copy()
			
		self.error_matrix = dict()	
		self.error_matrix['leaky'] = numpy.zeros(len(self.reservoir_sizes))
		if self.do_compare_leaky:
			self.error_matrix['non-leaky'] = self.error_matrix['leaky'].copy()
		
		# Create empty list for confusion matrices for each network size
		self.c_matrices = dict()
		self.c_matrices['leaky'] = numpy.zeros([len(self.reservoir_sizes), self.n_vowels+1, self.n_vowels+1])
		if self.do_compare_leaky:
			self.c_matrices['non-leaky'] = self.c_matrices['leaky'].copy()
		
		# Final errors and cmatrices (only used by master)
		if self.rank==0:
			self.final_errors = dict()
			self.final_cmatrices = dict()
			self.final_stds = dict()
		
		# Moved full initialisation of sample dicts to get_samples
		self.labeled_samples = dict()
		self.only_for_test_labeled_samples = dict()
		self.labeled_null_samples = dict()
		self.only_for_test_labeled_null_samples = dict()
		self.labels = dict()
		self.null_labels = dict()
		
		
		# Training and test sets
		self.training_set = list()
		self.test_set = list()
		self.training_set = [ ( numpy.array([]), numpy.array([]) ) ]
		self.test_set = [ ( numpy.array([]), numpy.array([]) ) ]
		
		# Verbosity
		self.verbose = self.be_verbose_in['hear']
		
		# A list of ESN candidates, of which the user will be asked to chose one to be used by 'learn' and analyzed for confidence thresholds.
		self.ESN_candidates = []
		
		# Initialize paths
		self.current_path = os.path.dirname(os.path.abspath(__file__)) #Get path of this script
		self.base_path = os.getcwd()
		self.input_path = self.base_path+'/data/output/ambient_speech/'+self.sp_group_name+'/samples'
		self.proto_input_paths = self.amb_speech.output_paths # [vowel][speaker] (dict(list(tuples)))
		# why this format? Because we have 2 input paths for each vowel and speaker, one for the .wav files and one for the .dat.gz files! (2nd part of the tuple)
		
			# The output path is where the ESN gets stored. We'll use the same user-defined subfolder as in the result folder.
		self.output_path = self.base_path+'/data/output/hear/'+self.sp_group_name+'/'+self.subfolder['hear']+'/'+'worker_'+str(self.rank)
		self.ESN_output_path = self.base_path+'/data/output/hear/'+self.sp_group_name+'/'+self.subfolder['hear']+'/'+'current_auditory_system.flow'
			# The result path is where master (rank 1) will store all the interesting plots, (confusion matrices etc)
		self.result_path = self.base_path+'/results/hear/'+self.sp_group_name
		if self.subfolder['hear']:
			self.result_path = self.result_path+'/'+self.subfolder['hear']
		
		# delete (large) ambient speech class instance
		self.amb_speech = []
		
		
		
	def setup_folders(self):
		"""
		Sets up the appropriate folders for the group "self".
		(.. and cleans them up.)
		"""
		
		if self.rank == 0:
			#Create folder Output
			if os.path.isdir(self.output_path):
				system('rm -rf '+self.output_path)
			else:
				system('mkdir --parents '+self.output_path)
			
			#Create folder for results (plots etc.)
			if not os.path.isdir(self.result_path):
				system('mkdir --parents '+self.result_path)
			elif not(os.listdir(self.result_path) == []):
				system('rm -r '+self.result_path+'/*')
			
			#Create folder for prototype plots
			if self.do_plot_hearing:
				if not os.path.isdir(self.result_path+'/prototype_plots'):
					system('mkdir '+self.result_path+'/prototype_plots')
				elif not(os.listdir(self.result_path+'/prototype_plots') == []):
					system('rm -r '+self.result_path+'/prototype_plots/*')
			
			#Create an empty result file
			size = str(not len(self.reservoir_sizes)==1)
			filename = str(self.n_vowels)+'vow_'+str(self.n_channels)+'chan_'+size+'size_'+str(self.do_compare_leaky)+'_compare.out'
			
			self.result_file = self.result_path+'/' + filename
			
	
	def make_Oger_inspectable(self):
		# numpy random seed w.r.t. global runtime
		numpy.random.seed()
		numpy.random.seed(numpy.random.randint(256) * (self.rank+1))
		
		# numpy random seed w.r.t. worker
		random.seed(numpy.random.randint(256) * (self.rank+1))
		
		# random seed w.r.t. worker
		Oger.utils.make_inspectable(Oger.nodes.LeakyReservoirNode)
		
		# make reservoir states inspectable for plotting
		Oger.utils.make_inspectable(Oger.nodes.ReservoirNode)
	

	def get_samples(self):
		"""
		Get all the samples in the sample directory. Returns them shuffled.
		These are then used for setting up a test set and a training set for the ESN.
		self.labeled_samples initialized as an empty dictionary in __init__.
		
		E.g. self.sample["a"] could look like this: [...,( [..i-th sound...],[i-th label (timerange x n_vowel+1)] ),([], [ [][] ]),...]
		
		Basic procedure: Do everything for each vowel and then in the end also for the null class. 
		"""
		
		
		
		# Empty sample matrices. Here, the samples (read by numpy) and their corresponding
		# labels will be stored, - then mixed up randomly and a test & training set is picked
		# from them. ( in get_samples() and get_sets())
		
		for vowel in self.vowels:
			self.labeled_samples[vowel] = [ ( [], numpy.array([]) ) ]
			self.only_for_test_labeled_samples[vowel] = [ ( [], numpy.array([]) ) ]
			
		self.labeled_null_samples = [ ( [], numpy.array([]) ) ]
		self.only_for_test_labeled_null_samples = [ ( [], numpy.array([]) ) ]
		

		
		
		
		
		# The timerange included in the classification (36 slots in the label plot for each vowel)
		self.n_timesteps = 36
		self.timerange = range(self.n_timesteps)
		
		# Index of null samples in the lables
		i_null = self.n_vowels
		
		if self.verbose:
			print "Getting samples from the following vowels:"
			print self.vowels
		
	
		# Create a proto label for the vowel
		protolabel = -numpy.ones([self.n_timesteps, self.n_vowels+1])
		# Create and initialize labels
		for vowel in self.vowels:	
			self.labels[vowel] = protolabel.copy()
		self.null_labels = protolabel.copy()
		
		# Putting the right lines to -1 (fully activated)
		for vowel in self.vowels:
			self.labels[vowel][:, self.vowels.index(vowel)] *= -1.
		self.null_labels[:, i_null] *= -1.
		
		# Representative samples:
		# -----------------------------------------------------------------
		for vowel in self.vowels:
			
			# Path of the vowel samples. Get a list of all the files in the dir.
			current_path = self.input_path+'/'+vowel
			files = os.listdir(current_path)
			
			for item in files:
				if '.dat.gz' in item:
					# Get the speaker from the item.
					name,dat,gz=item.split('.')
					# All these samples were produced by sampling round an original vowel (even if now classified differently)
					# Plus, the names of the samples have the speaker number in them.
					try:
						original_vowel,n_sample = name.split('_')
					except ValueError:
						nulltoken,original_vowel,n_sample = name.split('_')
						
					speaker = int(n_sample)%len(self.speakers)
					
					if speaker in self.omitted_test_speakers:
						# Use these samples only for the testing.
						self.only_for_test_labeled_samples[vowel].append((numpy.load(gzip.open(current_path+'/'+item)),
												self.labels[vowel].copy()))
						
					else:
						# (self.labeled_samples[vowel] is a list. Append a tuple to it.)
						self.labeled_samples[vowel].append((numpy.load(gzip.open(current_path+'/'+item)),
													self.labels[vowel].copy()))
						# format: list of tuples of numpy arrays: [ ( numpy.array([]), numpy.array([]) ) ]
					
		# -----------------------------------------------------------------
			
			
			
		
		# Null samples:
		# -----------------------------------------------------------------
		# Path of the null samples. Get a list of all the files in the dir.
		current_path = self.input_path+'/null'
		files = os.listdir(current_path)
	
		for item in files:
			if '.dat.gz' in item:
			
				# Get the speaker name, from which the sample was produced. 
				name,dat,gz=item.split('.')
				try: # Look at the files in data!
					original_vowel,n_sample = name.split('_')
				except ValueError:
					nulltoken,original_vowel,n_sample = name.split('_')
				speaker = int(n_sample)%len(self.speakers)
				
				
				
				
				if speaker in self.omitted_test_speakers:
					# Use these samples only for the testing.
					self.only_for_test_labeled_null_samples.append((numpy.load(gzip.open(current_path+'/'+item)),
												self.null_labels.copy()))
					
				else:
					# (self.labeled_null_samples is a list. Append a tuple to it.)
					self.labeled_null_samples.append((numpy.load(gzip.open(current_path+'/'+item)),
											self.null_labels.copy()))
				
		# -----------------------------------------------------------------
		
		
	
	
	
	def get_sets(self):
		"""
		Get a training and a test set for the ESN learning.
		These are picked from all the samples we got in 'get_samples'
		"""
		
		# Refresh initialisation
		self.training_set = []
		self.test_set = []
		
		
		
		# In the normal case, where both test and samples are taken from all speakers:
		# -------------------------------------------------------------------------------------------------------------
		# -------------------------------------------------------------------------------------------------------------
		if not self.omitted_test_speakers:
			# We created representative samples of /a/, /e/ etc. as well as null samples in ambient_speech.py.
			# Now, we'll see how many null, or /a/, or /e/ sounds we want.
			n_train = self.n_samples['train'] * self.n_speakers
			# (remember: self.n_samples[...] was defined as 'per speaker'.
		
			# Same for test samples
			n_test = self.n_samples['test'] * self.n_speakers
		
		
			for vowel in self.vowels:
			
				# Get representative part of training/test set.
				random.shuffle(self.labeled_samples[vowel])
			
				# If many vowels were moved out of a folder, our list of labeled samples might
				# be quite short - shorter even than training_samples + test_samples!
				for sample in range(n_train):
					try:
						self.training_set.append(self.labeled_samples[vowel][sample])
					except IndexError:
						raise RuntimeError('Too few '+vowel+' samples!')
						#self.training_set.append(random.choice(self.labeled_samples[vowel]))
					
			
				for sample in range(n_train,n_train+n_test):
					try:
						self.test_set.append(self.labeled_samples[vowel][sample])
					except IndexError:
						raise RuntimeError('Too few '+vowel+' samples!')
						#self.test_set.append(random.choice(self.labeled_samples[vowel]))
		
		
			# Get non-representative (null) part of training/test set.
			random.shuffle(self.labeled_null_samples)
			for sample in range(n_train):
				try:
					self.training_set.append(self.labeled_null_samples[sample])
				except IndexError:
					print 'Too few null samples!'
					raw_input('Proceed?')
		
			for sample in range(n_train,n_train+n_test):
				try:
					self.test_set.append(self.labeled_null_samples[sample])
				except IndexError:
					print 'Too few null samples!'					
					raw_input('Proceed?')
		
		
		# -------------------------------------------------------------------------------------------------------------
		
		
		
		
		
		
		# In the case, where some speakers are omitted from the training samples. Take those speakers as test samples!	
		# -------------------------------------------------------------------------------------------------------------
		# -------------------------------------------------------------------------------------------------------------	
		else:
			for vowel in self.vowels:
				
				for labeled_sample in self.labeled_samples[vowel]:
					self.training_set.append(labeled_sample)
				
				for labeled_sample in self.only_for_test_labeled_samples[vowel]:
					self.test_set.append(labeled_sample)
					
			for labeled_sample in self.labeled_null_samples:
				self.training_set.append(labeled_sample)
				
			for labeled_sample in self.only_for_test_labeled_null_samples:
				self.test_set.append(labeled_sample)
			
		# -------------------------------------------------------------------------------------------------------------
		
		
		
		
		# Shuffle representative with non-represantive training elements.
		random.shuffle(self.training_set)
		random.shuffle(self.test_set)
		
		
		def check_for_rotten_eggs(x):#
			"""
			Check for 'rotten eggs' in set
			"""
			i_rotten=0
			i = 0
			while i in range(len(x)):
				if x[i][0] == []:
					i_rotten+=1
					x.pop(i)
				else:
					i+=1
			print '('+str(i_rotten)+' rotten samples found in set)'
		
		check_for_rotten_eggs(self.training_set)
		check_for_rotten_eggs(self.test_set)








#######################################################################################################################	
# MAIN FUNCTIONS                                    ###################################################################
#######################################################################################################################
	
	def simulate_ESN(self):
		""" main function simulating both leaky and non-leaky ESNs
			globals:
			- self.reservoir_sizes: list of reservoir sizes
			- self.n_vowels: number of vowels used
			- self.do_plot_hearing: boolean defining if plots are created
			- self.separate: boolean defining if infant samples are used as test data
			- self.n_channels: number of used channels
		"""
		# Compare leaky networks with non-leaky networks?
		# ------------------------------------------------------------------------
	
		for j in xrange(len(self.reservoir_sizes)):     # loop over network sizes
			self.current_reservoir = j
			for train in xrange(self.trains_per_worker):
				
				# Leaky (done anyway)
				# ...........................................................
				
				print('worker', self.rank, 'of', self.n_workers, 'simulating leaky network of size', self.reservoir_sizes[self.current_reservoir], 
																							'('+str(train+1)+'/'+str(self.trains_per_worker)+')')
																							
				# call learn function
				error_leaky, c_matrix_leaky = self.learn(output_dim=self.reservoir_sizes[self.current_reservoir], leaky=True)

				if (train==0) and self.do_plot_hearing and (self.rank == 0):
					self.plot_prototypes(N=self.reservoir_sizes[self.current_reservoir])
				# ...........................................................
				
				
				if self.do_compare_leaky:
					# Non-leaky (optional)
					# ...........................................................
					
					print('worker', self.rank, 'of', self.n_workers, 'simulating non-leaky network of size', self.reservoir_sizes[self.current_reservoir], 
																							'('+str(train+1)+'/'+str(self.trains_per_worker)+')')
					
					# call learn function to execute one simulation run
					error_nonleaky, c_matrix_nonleaky = self.learn(output_dim=self.reservoir_sizes[self.current_reservoir], leaky=False)
																			
					if (train==0) and self.do_plot_hearing and (self.rank == 0):
						self.plot_prototypes(N=self.reservoir_sizes[self.current_reservoir])
					# ...........................................................
				
				
				
				
				# Collect current error rates and append current confusion matrix.
				# ----------------------------------------------------------------
					self.errors['non-leaky'][train][self.current_reservoir] = error_nonleaky
					self.c_matrices['non-leaky'][self.current_reservoir] += c_matrix_nonleaky
					
				self.errors['leaky'][train][self.current_reservoir] = error_leaky
				self.c_matrices['leaky'][self.current_reservoir] += c_matrix_leaky
				# ----------------------------------------------------------------
				
				# Print current cmatrices?
				if self.verbose and self.do_compare_leaky:
					print('c_matrix_leaky:', c_matrix_leaky)
					print('c_matrix_nonleaky:', c_matrix_nonleaky)
					
				elif self.verbose:
					print('c_matrix_leaky:', c_matrix_leaky)
					
					
					
						

		# Divide by number of trains per worker
		if self.do_compare_leaky:
			self.c_matrices['leaky'] /= self.trains_per_worker
			self.c_matrices['non-leaky'] /= self.trains_per_worker
		else:
			self.c_matrices['leaky'] /= self.trains_per_worker
		
		
		# Print cmatrices?
		if self.do_compare_leaky and self.verbose:
			print('total_c_matrices_leaky:', self.c_matrices['leaky'])
			print('total_c_matrices_nonleaky:', self.c_matrices['non-leaky'])
		elif self.verbose:
			print('total_c_matrices_leaky:', self.c_matrices['leaky'])
		


	
	def learn(self,output_dim,leaky):
		""" function to perform supervised learning on an ESN
			data: data to be learned (ndarray including AN activations and teacher signals) OLD VERSION
			self.n_vowels: total number of vowels used
			self.reservoir_sizes: size of ESN
			leaky: boolean defining if leaky ESN is to be used
			self.do_plot_hearing: boolean defining if results are to be plotted
			self.verbose: boolean defining if progress messages are to be displayed
			testdata: provide test data for manual testing (no cross validation) OLD VERSION
			self.n_channels: number of channels used
			classification: boolean defining if sensory classification is performed instead of motor prediction
		"""
	

		# get "self.test_set" and "self.training_set"
		self.get_sets()
		
		
		print('Training samples: = '+str(len(self.training_set)))
		print('Test samples: = '+str(len(self.test_set)))
		
		
		N_classes = self.n_vowels+1					# number of classes is total number of vowels + null class
		input_dim = self.n_channels					# input dimension is number of used channels

		
		print('Constructing reservoir and training.\n'+50*'-'+'\n'+'\t........')
		
		
		# construct individual nodes
		if leaky:                           # construct leaky reservoir
			reservoir = Oger.nodes.LeakyReservoirNode(  input_dim=input_dim, output_dim=output_dim, input_scaling=1., 
														spectral_radius=self.spectral_radius, leak_rate=self.leak_rate)
														# call LeakyReservoirNode with appropriate number of input units and 
														#  given number of reservoir units
		else:                               # construct non-leaky reservoir
			reservoir = Oger.nodes.ReservoirNode(input_dim=input_dim, output_dim=output_dim, input_scaling=1.)
									# call ReservoirNode with appropriate number of input units and given number of reservoir units

		if self.logistic:
			readout = Oger.nodes.LogisticRegressionNode()
		else:
			readout = Oger.nodes.RidgeRegressionNode(self.regularization)
											# construct output units with Ridge Regression training method

		self.flow = mdp.Flow([reservoir, readout])  			# connect reservoir and output nodes
	


		if self.verbose:
			print("Training...")
		
		
		# train self.flow with input files provided by file iterator
			# self.training_set has following format: list(tuple(nparray))
		self.flow.train([[], self.training_set])
		
		
		
		
		
		ytest = []                          # initialize list of test output

		if self.verbose:
			print("Applying to testset...")

		losses = []                         # initiate list for discrete recognition variable for each test item
		ymean = []                          # initiate list for true class of each test item
		ytestmean = []                      # initiate list for class vote of trained flow for each test item

		for i_sample in xrange(len(self.test_set)):       # loop over all test samples
			if self.verbose:
				print('testing with sample '+str(i_sample))

			xtest = self.test_set[i_sample][0]
							                # load xtest and ytarget as separate numpy.array([])s
			ytarget = self.test_set[i_sample][1]
			ytest = self.flow(xtest)             # evaluate trained output units' responses for current test item

			mean_sample_vote = mdp.numx.mean(ytest, axis=0)
							                # average each output neurons' response over time
			if self.verbose:
				print('mean_sample_vote = '+str(mean_sample_vote))
			target = mdp.numx.mean(ytarget, axis=0)
							                # average teacher signals over time
			if self.verbose:
				print('target = '+str(target))

			argmax_vote = sp.argmax(mean_sample_vote)
							                # winner-take-all vote for final classification
			ytestmean.append(argmax_vote)   # append current vote to votes list of all items
			argmax_target = sp.argmax(target)
							                # evaluate true class of current test item
			ymean.append(argmax_target)     # append current true class to list of all items

			loss = Oger.utils.loss_01(mdp.numx.atleast_2d(argmax_vote), mdp.numx.atleast_2d(argmax_target))
							                # call loss_01 to compare vote and true class, 0 if match, 1 else
			if self.verbose:
				print('loss = '+str(loss))
			losses.append(loss)             # append current loss to losses of all items

			xtest = None                    # destroy xtest, ytest, ytarget, current_data to free up memory
			ytest = None
			ytarget = None

		error = mdp.numx.mean(losses)       # error rate is average number of mismatches

		if self.verbose:
			print("error: "+str(error))
			print('ymean: '+str(ymean))
			print('ytestmean: '+str(ytestmean))

		ytestmean = numpy.array(ytestmean)     # convert ytestmean and ymean lists to numpy.array([]) for confusion matrix
		ymean = numpy.array(ymean)
	
		
		confusion_matrix = ConfusionMatrix.from_data(N_classes, ytestmean, ymean) # 10 classes
					                        # create confusion matrix from class votes and true classes
		c_matrix = confusion_matrix.balance()
					                        # normalize confusion matrix
		c_matrix = numpy.array(c_matrix)

		if self.verbose:
			print('confusion_matrix = '+str(c_matrix))


		self.save_flow(leaky)

		return error, c_matrix              # return current error rate and confusion matrix







#######################################################################################################################
# SAVE AND PLOTS                                    ###################################################################
#######################################################################################################################


	def plot_prototypes(self,N):
		""" 
		Function to visualize output neurons' states and reservoir activations
		for prototypical vowels of each speaker. ('How do neuron states look like
		for the 'precise' vowels produced by each speaker?)
		"""


		if self.verbose:
			print('worker', self.rank, 'plotting prototypes')
		
		plot_vowels = self.vowels[:]
		if plot_vowels[-1] == '@':
			plot_vowels.pop()
		for vowel in plot_vowels:             # loop over all syllables of all speakers
			for speaker in self.speakers:
				
				# Get correct paths for 1. where to save the plot and 2. which audiofile should be loaded..
					#1.
				result_file_plot = self.result_path+'/prototype_plots/speaker='+str(speaker)+'_vowel='+vowel+'_N='+str(N)+'.png'
					#2.
				input_path = self.proto_input_paths[vowel][speaker][1]
				
										        # name of corresponding activation file
				if self.verbose:
					print('loading '+input_path)
				if not os.path.exists(input_path):
												# end loop if input_file not found
					print('Prototype file not found!')
					break
					
				input_file = gzip.open(input_path, 'rb')
										        # open current input_file in gzip read mode
				current_data = numpy.load(input_file)  # load numpy.array([]) from current input_file
				input_file.close()                  # close input_file
				

				xtest = current_data
										        # read activations from input array
				ytest = self.flow(xtest)             # get activations of output neurons of trained network

				current_flow = self.flow[0].inspect()[0].T
				
				
				numpy.array([ytest.T, current_flow]).dump(result_file_plot+'.np')

				n_subplots_x, n_subplots_y = 2, 1
										        # arrange two plots in one column
				pylab.subplot(n_subplots_x, n_subplots_y, 1)
										        # upper plot

				ytest_min = ytest.min()
				ytest_max = ytest.max()
				if abs(ytest_min) > ytest_max:
					vmin = ytest_min
					vmax = -ytest_min
				else:
					vmax = ytest_max
					vmin = -ytest_max

				if self.compressed_output:
					class_activity = pylab.imshow(ytest.T, origin='lower', cmap=pylab.cm.bwr, aspect=10.0/(self.n_vowels+1), interpolation='none', vmin=vmin, vmax=vmax)
				else:
					class_activity = pylab.imshow(ytest.T, origin='lower', cmap=pylab.cm.bwr, aspect=10000.0/self.n_vowels, interpolation='none', vmin=vmin, vmax=vmax)
										        # plot output activations, adjust to get uniform aspect for all self.n_vowels
				pylab.title("Class activations of speaker "+str(speaker)+', gesture '+vowel)
				pylab.ylabel("Class")
				pylab.xlabel('')
				ylabels = self.vowels[:]
				ylabels.append('null')
				pylab.yticks(range(self.n_vowels+1), ylabels)

				if self.compressed_output:
					pylab.xticks(range(0, 35, 5), numpy.arange(0.0, 0.7, 0.1))
				else:
					pylab.xticks(range(0, 35000, 5000), numpy.arange(0.0, 0.7, 0.1))

				pylab.colorbar(class_activity)


				 # plot confusion matrix (balanced, each class is equally weighted)

				n_subplots_x, n_subplots_y = 2, 1
				pylab.subplot(n_subplots_x, n_subplots_y, 2)
										        # lower plot

				current_flow_min = current_flow.min()
				current_flow_max = current_flow.max()
				if abs(current_flow_min) > current_flow_max:
					vmin_c = current_flow_min
					vmax_c = -current_flow_min
				else:
					vmax_c = current_flow_max
					vmin_c = -current_flow_max

				if self.compressed_output:
					reservoir_activity = pylab.imshow(current_flow, origin='lower', cmap=pylab.cm.bwr, aspect=10.0/N, interpolation='none', vmin=vmin_c, vmax=vmax_c)
				else:
					reservoir_activity = pylab.imshow(current_flow, origin='lower', cmap=pylab.cm.bwr, aspect=10000.0/N, interpolation='none', vmin=vmin_c, vmax=vmax_c)
										        # plot reservoir states of current prototype,
										        #  adjust to get uniform aspect for all N
				pylab.title("Reservoir states")
				pylab.xlabel('Time (s)')

				if self.compressed_output:
					pylab.xticks(range(0, 35, 5), numpy.arange(0.0, 0.7, 0.1))
				else:
					pylab.xticks(range(0, 35000, 5000), numpy.arange(0.0, 0.7, 0.1))

				pylab.ylabel("Neuron")
				if N < 6:
					pylab.yticks(range(N))

				pylab.colorbar(reservoir_activity)

				pylab.savefig(result_file_plot)   # save figure
				pylab.close('all')

		xtest = None                    # destroy xtest and ytest to free up memory
		ytest = None
		class_activity = None
		reservoir_activity = None



	
	def write_and_plot_results(self):
		"""
		Simply writes the errors to a neat result file and plots the cmatrices!
		
		Keep in mind: Final confusion matrices and errors have just been gathered by master in our main function in 'hear.py'.
		
		The formats are as such: (dicts)
		self.final_errors['leaky'] = comm.gather(self.errors_leaky, root=0)
		self.final_errors['non-leaky'] = comm.gather(self.errors_nonleaky, roo=0) if self.compare_leaky else []
		self.final_cmatrices['leaky'] = comm.gather(self.c_matrices_leaky, root=0)
		self.final_cmatrices['non-leaky'] = comm.gather(self.c_matrices_nonleaky, root=0) if self.compare_leaky else []
	
		"""
		
		# See 'setup_folders()' for path info.
		if self.verbose:
			'Results printed to: \n'+self.result_file
		
		outputfile = open(self.result_file, 'w') 
		
		# If the user wanted to compare leaky reservoirs with non-leaky reservoirs, there will be two
		# errors, as well as two confusion matrices! leaky_states is either only leaky, or (if compared)
		# both leaky and non-leaky. Write out all the results for both cases.
		leaky_states = ['leaky']
		if self.do_compare_leaky:
			leaky_states.append('non-leaky')
		# for both (or only leaky)
		for leaky_state in leaky_states:
			
			
			# Make the output a bit 'nicer'.
			# ---------------------------------------------------------------------------
				#Reshape errors and create stds.
			errors = numpy.array(self.final_errors[leaky_state])
			errors = errors.reshape([self.n_trains, len(self.reservoir_sizes)])
			self.final_errors[leaky_state] = numpy.average(errors, axis=0)
			self.final_stds[leaky_state] = numpy.std(errors, axis=0)
				#Average confusion matrices
			self.final_cmatrices[leaky_state] = numpy.array(self.final_cmatrices[leaky_state])
			#self.averaged_final_cmatrices[leaky_state] = numpy.average(self.final_cmatrices[leaky_state], axis=0)
			# ---------------------------------------------------------------------------

			
			
			outputfile.write(leaky_state+':\n\n')
			
			for i in xrange(len(self.reservoir_sizes)):  # loop over all network sizes
				# Write errors to file
				# ---------------------------------------------------------------------------
				outputfile.write(str(			self.reservoir_sizes[i])+'     '+
												str(self.final_errors[leaky_state][i])+'     '+
												str(self.final_stds[leaky_state][i])+'\n')
								                # record errors and standard deviations for each network size
				outputfile.flush()
				# ---------------------------------------------------------------------------

				# If wanted, plot confusion matrices
				# ---------------------------------------------------------------------------
				if self.do_plot_hearing:                       # plots for current network size
					C_Matrix = self.final_cmatrices[leaky_state][i]

					C_Matrix = ConfusionMatrix(C_Matrix, labels=range(self.n_vowels+1))
										        # convert to confusion matrix object

					pylab.figure()              # plot confusion matrix
					pylab.title('Confusion matrix of leaky reservoir of size '+str(self.reservoir_sizes[i]))
					labels = self.vowels[:]
					labels.append('null')
					pylab.xticks(numpy.arange(self.n_vowels+1), labels)
					pylab.yticks(numpy.arange(self.n_vowels+1), labels)
					pylab.xlabel('classified as')
					pylab.ylabel('sample')
					if self.verbose:
						print 'current C_Matrix:', C_Matrix
					self.plot_confusion_matrix(C_Matrix, suffix=leaky_state, nn=self.reservoir_sizes[i])
				# ---------------------------------------------------------------------------
				
		if self.do_plot_hearing:
			self.plot_errors()
				
		
		outputfile.close()
		print 'done'


	
	def save_flow(self, leaky):
		if self.rank == 0:
			filename = self.output_path+'reservoir'+str(self.reservoir_sizes[self.current_reservoir])+'_leaky'+str(leaky)+'.flow'
			with open(filename, 'wb') as flow_file:
				cPickle.dump(self.flow, flow_file)
				self.ESN_candidates.append(filename)
			




	def plot_confusion_matrix(self, matrix, suffix, nn):
		""" function to visualise a balanced confusion matrix
			 modified version of the predefined Oger.utils.plot_conf function
			 additional arguments: outputfile_ for plot files, nn for reservoir size"""

		result_file_plot = self.result_file+'_'+str(nn)+'_'+suffix+'.png'
		
		index = 2
		while os.path.exists(result_file_plot):
			result_file_plot = self.result_file+'_'+str(nn)+'('+index+')_'+suffix+'.png'
			index+=1
		
		numpy.asarray(matrix).dump(result_file_plot+'.np')

		res = pylab.imshow(numpy.asarray(matrix), cmap=pylab.cm.jet, interpolation='nearest')
		for i, err in enumerate(matrix.correct):
								            # display correct detection percentages
								            # (only makes sense for CMs that are normalised per class (each row sums to 1))
			err_percent = "%d%%" % round(err * 100)
			pylab.text(i-.2, i+.1, err_percent, fontsize=14)

		pylab.colorbar(res)

		pylab.savefig(result_file_plot)

	
	
	
	
	def plot_errors(self):
		"""
		function to plot the errors of the classification over various network sizes
		"""
		system('mkdir --parents '+self.result_path+'/error_plots')
		f = plt.figure()
		plt.title('ESN classification errors for various network sizes')
		sf = f.add_subplot(111)#(121)
		
		leaky, = sf.plot(self.reservoir_sizes,self.final_errors['leaky'],'b-',label='leaky')
		
		if self.do_compare_leaky:
			non_leaky, = sf.plot(self.reservoir_sizes,self.final_errors['non-leaky'],'r-',label='non-leaky')
			plt.legend([leaky,non_leaky],['leaky','non-leaky'])
		else:
			plt.legend([leaky],['leaky'])
			
		plt.xlabel('Network size')
		plt.ylabel('Error rate')
		plt.ylim([0,1])
		f.savefig(self.result_path+'/error_plots/errors.png')
			

	def plot_error_matrix(self):
		"""
		Only executed for self.do_sweep_omitted_speakers = True
		"""
		
		
		system('mkdir --parents '+self.result_path+'/error_plots')
		
		for leakystate in ['leaky','non-leaky']:
		
			fig = plt.figure()
			plt.clf()
			plt.title('ESN classification errors over speakers omitted in training')
			ax = fig.add_subplot(1,1,1)
			ax.set_aspect('equal')
			plt.imshow(self.error_matrix[leakystate], interpolation='nearest', cmap=plt.cm.coolwarm)
		
			ax.set_yticks(range(len(self.omitted_group_labels)))
			ax.set_yticklabels(self.omitted_group_labels)
			ax.set_xticks(range(len(self.reservoir_sizes)))
			ax.set_xticklabels(self.reservoir_sizes,rotation='vertical')
			plt.ylabel('omitted speaker')
			plt.xlabel('reservoir sizes')
			try:
				plt.colorbar()
			except RuntimeError:
				pass
		
			fig.savefig(self.result_path+'/error_plots/error_matrix_%s.png'%leakystate)
		
	


#######################################################################################################################
# ESN CHOICE AND ANALYSIS                           ###################################################################
#######################################################################################################################


	def choose_final_ESN(self):
		"""
		The User choses which ESN is to be analyzed further (and used in learn).
		"""
		
		print "Please chose one of the following trained ESNs, that will be further analyzed and used in learning:"
		for i in range(len(self.ESN_candidates)):
			print "ESN Nr. %d:\n%s"%(i,self.ESN_candidates[i])
		candidate_index = int(raw_input("Enter the ESN-Number:\n\t>>"))
	
		
		system('cp '+self.ESN_candidates[candidate_index]+' '+self.ESN_output_path)
		






	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
"""
	def exclude_speakers_from_data(self,excluded):
		"""
#A function that lets us remove certain speakers from the ESN - training data and saves them in a "excluded_samples" directory in "hear".
#Also, the original samples are saved in a compressed file in data/output/hear.
"""
	
		# Each sample is labeled with a sample number. To get the right sample numbers to be excluded, we can look at how these samples were produced.
		# From this we gather, that sample 0 was produced by speaker 0, sample 1 by speaker 1, etc until sample 21. After that sample 22 was produced
		# by speaker 0 again, 23 by 1, 24 by 2, etc. Thus:
		# 													speaker_number = sample_number modulo 22 (or the size of the group) (len(self.speakers))
		# Thus: sample_numbers = speaker_number + n*22 (n from 0 to n_samples per speaker)
		# ----------------------------------------------------------------------------------------------------------------------------------------------
	
	
		from math import *
	
		excluded_samples = list()
	
		for n in range(self.vowel_data_n):
		
			for speaker in excluded:
				debug() # Check if all datatypes are int (next step)
				excluded_samples.append( len(self.speakers)*n + speaker )
			
	
	
	
		# Look through all the samples, and remove those that are in the excluded samples.
		# ----------------------------------------------------------------------------------------------------------------------------------------------
	"""
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	

