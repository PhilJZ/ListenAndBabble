
# Global imports
import numpy




class parameters(object):
	"""
	The idea is to keep all control parameters here, - Thus, results can be reproduced in a neat way.
	Warning: Before introducing new parameters, make sure that the name is not yet used (e.g. in a
	function executed before.)
	If you want to use one variable for two things, you might consider using a dictionary / class structure.
	"""
	
	
	def __init__(self):
		"""
		# ===============================================================================================================
		# ===============================================================================================================
		# Shared Parameters used in 'ambient_speech', 'hear' and 'learn' (and subfunctions)
		# ===============================================================================================================
		# ===============================================================================================================
		"""
		
			#Special result sub-folders?
		self.subfolder = {'hear':'','learn':''} #format : "xxxx/"
			#Provide explaining output?
		self.be_verbose_in = {'hear':False,'learn':True} #Verbosity anyway True in ambient speech.
		
		
		
		
		# Parallel Processing (MPI4PY)
		"""
		from mpi4py import MPI
		# Parameters used in parallel processing
		comm = MPI.COMM_WORLD                   		# setup MPI framework
		self.n_workers = comm.Get_size()             	# total number of workers / parallel processes
		self.rank = comm.Get_rank()              		# id of the worker of this specific instance of the parameter class
		"""
		# Temporary solution
		self.n_workers = 1
		self.rank = 0
		
	
	
	
	
	
	
	def get_ambient_speech_params(self):
		"""
		# ===============================================================================================================
		# ===============================================================================================================
		# Parameters that are used in 'ambient_speech' (and functions)
		# ===============================================================================================================
		# ===============================================================================================================
		"""
		# Jobs to be executed when calling ambient_speech.py.
			# Warning: Executing a job will generally delete data from the last time that job was performed!
		# --------------------------------------------------------------------------------------------------------------
			#Setup (directories, fundamental frequencies, glottal parameters in speaker files, etc.)
		self.do_setup = True
			#Synthesize? Shall the gestures included in the speakerfiles be synthesized? (These must first be set in VTL!) A VTL library is called, and airflow simulated. > .wav file produced
			#These wav files are used as prototypes of that specific gesture sound. (see "learn")
		self.do_make_proto = False
			#Shall those set-up speakers and their synthesized gestures (.wav files) be analyzed (formants, parameter developement of shapes over years e.g.)?
		self.do_setup_analysis = False
			#Shall speech samples (used to train the auditory system) be produced? (Takes a lot of time, depending on how many samples are produced..)
		self.do_make_samples = False
			#Shall the user be given the chance to change classifications? Make a backup of the samples, before executing this..
		self.do_user_check = False
			#Shall those speech samples be analysed? (Look at formants of good samples vs bad samples (meaning (non-) or representative)? )
		self.do_sample_analysis = False # Ideally do the sample analysis after the user check.

		
		
		
		
		
		
		
		
		# The name of the speaker group. This requires that a certain speaker group already be setup (as VTL '.speaker' files) in the right directory (data/spekers_gestures/"speaker_group")
		self.sp_group_name = 'srangefm_2'
	
		# The size of the speaker group (not nescessarily how many speakers are chosen from that group!)
		self.size = 22
	
		# The shape gestures we want to look at..
		self.vowels="all"
	
		# The speakers in the speaker group we want to look at.. If you want the whole speaker group, simply put "all".
		self.speakers="all"
		
		# F0-parameters:
		# --------------------------------------------------------------------------------------------------------------
		# Various F0-parameters can be used for each speaker. Chose which one.. (must be 1 parameter for each speaker)
		# Real parts: Male, Imag parts: Female. This format makes sense for speakers where we have one male and one female for each age!
			# Standard f0s, originally used in "srangefm".
		f0s_standard = numpy.array([505.202393494200,   355.126913609294,    293.202429182816, 274.930607250920   , 267.403576928634  ,  262.018639542581 ,253.342004157396  ,  239.229107613903   , 221.338079761552 ,201.538876040741 ,   181.701451891867]) + 1j* numpy.array([505.148497150820,    355.220897386812 ,   292.548958250982 ,274.320068337208  ,  268.628877776761  ,  266.442538954839 ,261.729637743741  ,  252.139959167463   , 239.579141365103 ,226.189890942510 ,   214.114914505535])
		
			# F0s taken from a Lee et al.
		f0s_Lee = 	numpy.array([505.,355.,293.,270.,268.,250.,235.,175.,135.,140.,140.]) + 1j* numpy.array([505.,355.,292.,271.,268.,250.,236.,237.,230.,240.,235.])

		self.f0s = f0s_Lee
		
		# Correct pitch slightly to yield right frequencies. (problem in VTL_API?)
		self.pitch_corr = -1.0
		
		
		# Introduce some noise in the f0?
		self.f0_sigma = 0
	
	
		# Parameters used in sampling: (null are the non-representative null samples > train audit. system to classify those as not-syllables!)
		# --------------------------------------------------------------------------------------------------------------
			# speech samples for each vowel for each speaker.
			# noise-sigma for the shape parameters > speech samples that are true vowel sounds or not representative (higher sigma)
			# larger noise-sigma for shape parameters > speech samples that shouldn't represent true vowel sounds
		self.sampling = 	   {'n_samples':12, # roughly 100 samples per cathegory required for good training..
									'sigma':0.01, # Sigma used for most vowels. With vowel /u/, sigma will be reduced thus: sigma_u = sigma * 0.7
									'process sound':True, 
									'n_channels':50,
									'compressed':True}

		self.sampling_null = {'n_samples':4, # The amount of null sampling of each vowel must add up to the amount of sampling for one vowel in total. 
											 # That way, we have for instance 100 (a good number for training) /a/ samples, 100 /e/ samples and 100 null samples.
									'sigma':0.2 , # Sigma used for most vowels. With vowel /u/, sigma will be reduced thus: sigma_u = sigma * 0.7
									'process sound':True, 
									'n_channels':50, 
									'compressed':True}
		
	
		# ===============================================================================================================
		
		
		
		
		
		
		
		
		
		
		
	
	def get_hear_params(self):
		"""
		# ===============================================================================================================
		# ===============================================================================================================
		# Parameters that are used in 'hear' (and functions)
		# ===============================================================================================================
		# ===============================================================================================================
		"""
		# Jobs to be done..?
		# --------------------------------------------------------------------------------------------------------------
				#Compare leaky networks with non-leaky networks?
		self.do_compare_leaky = True
				#Turns plotting on
		self.do_plot_hearing = True
				#After chosing a final output ESN, analyze it (thresholds for later rewards e.g.) ?
		self.do_analyze_output_ESN = True
				# Provide (explaining) output during execution?
		self.be_verbose_in['hear'] = False
		
	
	
	
		# Network parameters
		# --------------------------------------------------------------------------------------------------------------
			#Network sizes for variation [default: 10,20,50]
		self.reservoir_sizes = [1,10,100,1000]#[1,10,20,50,80,100,120,140,200,500,1000,2000] # 2000 is about the maximum for my pc.
			#Number of simulations per worker. [default: 1]
		self.trains_per_worker = 20
			#Leak rate of leaky reservoir neurons. [default: 0.4]
		self.leak_rate = 0.4
			#Spectral radius of leaky reservoir. [default: 0.9]
		self.spectral_radius = 0.9
			#Regularization parameter. [default: 0.001]
		self.regularization = 0.001
	
		
		
		# Training & Testing
		# --------------------------------------------------------------------------------------------------------------
		self.n_samples = {
			'train' 	: 	8,
			'test'		:	2
									}
		# Use:
		# For example: self.n_samples['train'] is the amount of null or /a/ or /o/ samples for each speaker which are 
		# used to train the ESN.
		
		# In order to evaluate the Quality of our ESN, we 'keep' some null, and vowel samples to plug into the ESN and
		# see if they are categorized correctly.. How many 'test' samples? > self.n_samples['test']
	
	
		# SPEAKER GENERALISATION
		# --------------------------------------------------------------------------------------------------------------
		# Speaker generalisation poses the question: If I omit the samples of this speaker (these speakers), and use them
		# in my test set instead - How high will the error rate be for the ESNs for different sizes?
		self.omitted_test_speakers = []
		
		
		# If you want to have a plot of which speakers, when omitted, yield (when used as a test set) which errors, 
		# for which network sizes, put True.
		# 
		# matrix:         							[len(self.speakers)]
		# 
		# [len(self.reservoir_sizes)]
		self.do_sweep_omitted_speakers = True
		self.omitted_groups = [[0,1,2,3],[4,5,6,7],[8,9,10,11],[12,13,14,15],[16,17,18,19,20,21]]
		self.omitted_group_labels = ['ages 0-2','ages 4-6','ages 8-10','ages 12-14','ages 16-20']
		#self.sweep_gap_size = 3
		
		

		# FLAGS
		# --------------------------------------------------------------------------------------------------------------
		#Use compressed DRNL output?
		self.compressed_output =  True
		
		#Train with logistic regression instead of ridge regression?
		self.logistic = False

		# Inferred and static variables
		self.n_channels = 50
		self.flow = None
	
		# ===============================================================================================================
		
		
		
		
		
		
		
		
		
		
		
	
	def get_learn_params(self):
		"""
		# ===============================================================================================================
		# ===============================================================================================================
		# Parameters that are used in 'learn' (and functions)
		# ===============================================================================================================
		# ===============================================================================================================
		"""
		# Output-subfolder and verbosity: See __init__()
		
		# Steering the reinforcement learner..
			# Either an integer (0 for instance) or a standard speaker ('infant'/'adult')
		self.learner = 'infant'
			# Is the learner motivated by random success in producing something 
			# like any vowel and then steering towards that one (intrinsic_motivation)
			#, or do we have a specific target to learn?
		self.intrinsic_motivation = True
			#Target vowel for imitation (default: 'a') "all" = self.vowels
		self.targets = "all"
			#Initial target (intrinsic motivation - the learner seeks the nearest target if set to False)
		self.target = False
			#vocal tract parameters to learn (default=['TCX']) (other options: 'all' or 'flat')
		self.pars_to_learn = ['TCX']
		
			#step-size = sigma (default=0.4)
		self.sigma_0 = 0.4
			#alpha for constraint penalty
		self.alpha = 1.0
			#energy balance factor
		self.energy_factor = 0.1				
			#threshold for convergence (reward threshold)
		self.convergence_threshold = 0.2
			#maximal conditioning number
		self.conditioning_maximum = 1e14
			#convergence range
		self.ptp_stop = 0.001
			#number of trials for averaging?
		self.n_trials = 1
			#???
		self.N_reservoir = 5 #random number under 20
		
			#use softmax reward?
		self.softmax = False
			#resample invalid motor parameters?
		self.resample = True
			#normalize ESN output?
		self.normalize = True
			#restart search after bad solution from random learnt variables?
		self.random_restart = True
			#ignore reward for convergence?
		self.no_reward_convergence = True
			#turn off convergence criterion?
		#self.no_convergence_criterion = True
			#initialize with predefined configuration?
		self.predefined = True
			#load saved state?
		self.load_state = None
			#simulate flat tongue, i.e. set all TS to 0?
		self.flat_tongue = True
			#keep sigma 0 constant?
		self.keep_sigma_constant = True






		"""
		Even this simple example may take a very long time to finish, or it may
		never stop.
		The most critical factor for this is the quality of the auditory learning.
		So your most important task is to make sure the auditory system is trained
		properly:
		- Make sure that every single ambient speech sample is placed in the
		  correct folder. If it sounds like [a] for you, put it in data/a and
		  so on. If it doesn't sound like any of the target vowels, put it into
		  one of the null folders. All null folders are treated in the same way,
		  so it doesn't exactly make a big difference where you put it. 
		  The idea behind the different null folders is this: If we generate 
		  speech samples in the vicinity of a given prototypical vowel, then 
		  null samples may show up that don't really sound like that vowel but 
		  show some similarity to it. So in a sense, they represent constraints 
		  of the acoustical properties of that vowel, which greatly helps the 
		  auditory system to learn a model of that vowel.
		  In the ideal case, all null folders contain the same number of samples,
		  which should be a third of what each vowel folder holds. The reason
		  is that we'd like to train the auditory system in an unbiased fashion
		  such that the trained auditory system shows no a priori classification
		  preference for any class. Use these rules to achieve this:
		   n_samples is a multiple of the number of vowels,
		   n_training is a multiple of the number of vowels,
		   each vowel folder contains at least n_samples samples,
		   each null folder contains at least n_samples/3 samples.
		  An alternative is to bias auditory learning in favor of the null class.
		  This will have the effect that the trained auditory system is more
		  likely to classify a given sample as a null sample, so the speech
		  sample needs to provide stronger evidence that it is one the target
		  vowels. So introducing such a training bias creates "stricter" auditory
		  systems, which is viable.
		- Increase the number of ambient speech samples for auditory learning. The
		  number of samples that we generated in this example is way too small for
		  efficient learning. Raise that number by at least one order of magnitude
		  to see reasonable learning progress.
		Other ways to improve/accelerate learning:
		- Make use of parallelization. Especially when you're moving to problems that
		  involve more than one articulator, being able to crank out tens or hundreds
		  of speech samples per babbling iteration is a huge advantage and makes
		  these problems feasible in the first place.
		- Run statistics. Because the reservoir of the auditory system is based on
		  random numbers, you can always end up with one that has trouble recognizing
		  one or the other vowel, even if your training parameters are good. Train
		  multiple auditory systems and pick one that performs well.
		- Lower the reward threshold during babbling. Setting the reward threshold 
		  to 0.5 is rather ad hoc. If you find that speech samples with 0.47 are
		  consistently good enough imitations, don't hesitate to lower that threshold.
		  This will make speech evaluation more lenient and the whole imitation
		  process much faster.
	  	"""	
		  	
		  	
		
    
