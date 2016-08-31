
# Global imports
import numpy




class parameters(object):
	"""
	The idea is to keep all control parameters here, - Thus, results can be reproduced in a neat way.
	Warning: Before introducing new parameters, make sure that the name is not yet used (e.g. in a
	function executed before.)
	If you want to use one variable for two things, you might consider using a dictionary / class structure.
	
	Adapting this project's code for your purposes will definitely mean changing the code itself (under src/.../).
	However, try to understand first, how you can control the ambient speech production, the hearing and the training from
	this script first, before making changes to the code.
	
	I tried to document most of these control parameters in this script. However, a future user won't completely understand
	their use, unless he/she double checks in the relevant scripts. For instance, in the subfunction of parameters: get_..
	_ambient_speech_params(self) (just a few lines down), all the control parameters for the first stage of the project are
	listed (namely control parameters for ambient_speech).
	In order to understand what, say, the self.sampling_null dictionary does, go to the script 'ambient_speech_functions.py',
	located in the src folder, and search for self.sampling_null. The context of the code will often make things more clear
	and you won't be in danger of not really knowing what your changing when you change a control parameter.
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
		
		
		# Temporary solution. 'hear' and 'learn' have aspects that can be done in parallel (using MPI for python, i.e.)
		# For now, we only have one worker, who's rank is 0 (master). Slave workers would have ranks 1,2,3...
		self.n_workers = 1
		self.rank = 0
		
		
		# Which main steps shall we do?
		# Here, we tell our project shell in the main directory what do actually do.
		self.execute_main_script = {'ambient_speech':False,'hear':False,'learn':False,'shell_analysis':True}
		
	
	
	
	
	
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

		# Important note:
		# Might need to install plotly library for analysis features
		# Additionally, in Ubuntu 16.04 the praat formants module doesn't seem to work anymore. > Either try debugging (I already tried..), or implement your
		# own formant extraction method. (Burg's method for instance). (Simply google Burg's formant extraction).
		
		
		
		
		
		
		# The name of the speaker group. This requires that a certain speaker group already be setup (as VTL '.speaker' files) in the right directory (data/speakers_gestures/"speaker_group")
		# Ambient speech will be generated using some or all of these speakers (self.speakers = "all" / self.speakers = [2,5,3] for example).
		# Speakers are named with integers. (See speaker group documentation in srangefm group).
		# In order to work, these speakers (of various ages) must have a file called _file_age.txt in the speaker directory, where filename and age and gender are listed in the
		# same format as in srangefm. Only then, can 'ambient_speech' read the ages and genders etc.
		self.sp_group_name = 'srangefm_2'
	
		# The size of the speaker group (not necessarily how many speakers are chosen from that group!) (You could also just read this parameter out of the speaker age file)
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
				#Compare leaky networks with non-leaky networks?
				#If false, only compute leaky networks.
		self.do_compare_leaky = False
				#Turns plotting on
		self.do_plot_hearing = True
				#After chosing a final output ESN, analyze it (thresholds for later rewards e.g.) ?
		self.do_analyze_output_ESN = False # Not yet implemented!
				#Analyze partially trained ESNs? (not including all speakers in the data)
		
	
	
	
		# Network parameters
		# --------------------------------------------------------------------------------------------------------------
			#Network sizes for variation ... default: [10,100,100]
		self.reservoir_sizes = [1000]
			#Number of simulations per worker. [default: 1]
		self.trains_per_worker = 1
			#Leak rate of leaky reservoir neurons. [default: 0.4]
		self.leak_rate = 0.4
			#Spectral radius of leaky reservoir. [default: 0.9]
		self.spectral_radius = 0.9
			#Regularization parameter. [default: 0.001]
		self.regularization = 0.001
	
		# Training & Testing
		
		self.n_samples = {
			'train' 	: 	9,
			'test'		:	1
									}
		# Use:
		# For example: self.n_samples['train'] is the amount of null or /a/ or /o/ samples for each speaker which are 
		# used to train the ESN.
		
		# In order to evaluate the Quality of our ESN, we 'keep' some null, and vowel samples to plug into the ESN and
		# see if they are categorized correctly.. How many 'test' samples? > self.n_samples['test']
	
	
		# Speaker generalisation?
		# The ESN will be trained on all samples above that age and be tested on all samples up to that age.
		# Put False, if ESN should train on all speakers
		self.generalisation_age = 2
		
		

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
		
		# Learner, Targets; etc.
			# An integer (0 for instance) - Which of the speakers shall be the learner?
		self.learner = 0
			# From which group?
		self.sp_group_name = 'srangefm_2'		
			#Target vowel for imitation (default: 'a') "all" = self.vowels
		self.targets = "all"
			#Initial target (can change after 1 iteration, due to intrinsic motivation)
		self.target = "a"
			#vocal tract parameters to learn (other options: 'all' or 'flat')
				# Only the tongue (flat)?
		#self.pars_to_learn = ['TCX','TTX','TTY','TBX','TBY']
				# All except lips and tongue side elevations, jaw
		self.pars_to_learn = ['TCX','TTX','TTY','TBX','TBY','HX','HY','VS'] # left out: ,'JA','LD','LP' (jaw and lip parameters)
				# All. Be careful not to omit the brackets!
		#self.pars_to_learn = ["all"]
			#simulate flat tongue, i.e. set all TS to 0 again and again? This only kicks into action, if you're not learning the tongue side elevations too.
		self.flat_tongue = False
		
				# The ESN we're using:
		# --------------------------------------------------------------
		self.ESN_path = '/data/output/hear/'+self.sp_group_name+'/worker_0reservoir1000_leakyTrue.flow'
		
		#self.ESN_path = raw_input("Please Enter the (relative) path to the reservoir (ESN) which is to be used in the babbling stage!:\n\t>")
		
			# Sequence of nodes:
		#self.ESN_sequence = ['a','i','u','null'] # For the orig. ESNs (Murakami)
		self.ESN_sequence = ["a","e","i","o","u",'null']
		
		
		# Sampling.
		# --------------------------------------------------------------
			#resample invalid motor parameters?
			# Chose one of the three options. 'Normal' simply resamples all parameters if one is below/above boundary (slow!), 'penalty' simply sets
			# the relative parameters to 0 or 1 if above 1 or below 0, and introduces a penalty to the fitness (to the reward). 'specific' only resamples
			# those parameters that went wrong, and keeps the others, also introducing a penalty. This is maybe the most complicated, but should be fast.
			# See 'learn functions' - if self.resample['normal']: etc for more details.
		self.resample = {'normal':False, 'penalty':False, 'specific':True} # Only one of these True!

		

		
		
		# Always save the most advanced speech sound of each cathegory as a sound file in data?
		# If you do this, you won't need to set a realistic convergence threshold. It will simply learn and learn, until the user
		# interrupts. You can always check what the best learnt vowels are in data/output/learn/[speaker group]/peak/..
		self.save_peak = True
		
		# Reward computation and sigma
		# --------------------------------------------------------------
			#step-size = sigma (default=0.4)
		self.sigma_0 = 0.5
			#keep sigma 0 constant? Of course, sigma will still change, according to the change of fitness. But 'current_sigma' in the code is a value, that the (constantly
			# changing) sigma will always go back to. (Go through 'learn_functions' looking for current_sigma and sigma, to see how they interact.
		self.keep_sigma_constant = False
			#alpha for constraint penalty (How badly should step-over-boundaries in the parameters be punished?)
		self.alpha = 1.0
			#energy balance factor. non-efficient motor configurations (e.g. extreme tongue positions) are punished when computing reward.
		self.energy_factor = 0.1
			#restart search after bad solution from random learnt/nonlearnt parameters (see code)?
		self.random_restart = True
			#ignore reward for convergence?. Reward is not enough. We must converge, in order to finish. Turn this on?
		self.must_converge = False
			# Is the learner motivated by random success in producing something 
			# like any vowel and then steering towards that one (intrinsic_motivation)
			#, or do we have a specific target to learn?
			# Intrinsic motivation only makes sense, if you're learning most parameters. That way, 
			# the not-learnt parameters (which are always kept static) will not interfere, when
			# we jump from one target to the next.
		self.intrinsic_motivation = True
		
		# Convergence Criteria.
		# --------------------------------------------------------------
		# threshold for convergence (reward threshold) - The confidence returned from the ESN. (originally 0.5 for all vowels)
		self.convergence_thresholds = {'a':0.5,'e':0.5,'i':0.5,'o':0.5,'u':0.5}
			#maximal conditioning number. (Covariance matrix)
		self.conditioning_maximum = 1e14
			# Parameter or Reward convergence range. This is the size of the parameter/reward window in which, if the last few generations stayed, the program will conclude,
			# that we have converged (found a local minimum in the fitness landscape) and reset (see random_restart) and correct sigma (make bigger)
		self.range_for_convergence = 0.05
			# The convergence interval is the number of datapoints which are checked if they converge in reward or parameters. This value is computed in the algorithm itself, 
			# however, the user can set the interval here. (If automatic, simply put False).
			# Must be above the size of one generation!
		self.user_convergence_interval = 20
			# Number of res
		self.N_reservoir = 5
			# Optional. (If you want to use the recommended amount of samples for each iteration, simply put False.) 
			# If you set population size, convergence interval also must be set (ca. 3 x population size)
		self.population_size = 15
		




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
		  	
		  	
		
    
