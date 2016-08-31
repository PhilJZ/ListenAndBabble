"""
A shell for calling source code which..
	-produces ambient speech for the learner ("ambient speech")
	-trains the ESN network ("hear")
	-and babbles away, exploring vocal tract parameters and learning specific syllables ("learn")
Author: Max Murakami and Philip Zurbuchen
"""

# General imports for this shell script.
# -----------------------------------------------------------------
import sys
import os


# Add the following lines to the shell code (in the beginning) if this directory is not yet part of sys.path!:

# This will enable python to import modules as done below.
# -----------------------------------------------------------------
my_path = os.path.abspath('')
if my_path not in sys.path:
	sys.path.append(my_path)
#print sys.path
# -----------------------------------------------------------------

# Parameter import
# -----------------------------------------------------------------
"""
Shell is steered using a parameter file (control/get_params.py). In this file, all control parameters for
each stage (ambient speech, hear and learn) are found and changed. This enables minimal interaction with the
code itself.
In this script, we only need to know which project steps to execute. 	
	Should we produce ambient speech, or hear, or both?
		.. or do we have a trained auditory system, and just want
		to learn? 
		> Set the corresponding parameters "True" or "False"
		in the parameter script itself.
																		
Importing all the parameters would look like this:
		
		from control.get_params import parameters as params
		params.__init__(self)
		self.get_learn_params()
		self.get_hear_params()
		self.get_ambient_speech_params()
		
For now, we only need what is defined in __init__(self) of params.
"""

# Import parameters class
from control.get_params import parameters as params

# Initialize parameters class
params_inst = params()

print "Shell calls 3 main scripts:"
print "Execute main script 'ambient_speech' | 'hear' | 'learn' | analyze data from shell? 	: "
print str(params_inst.execute_main_script['ambient_speech'])+' | '+str(params_inst.execute_main_script['hear'])+' | '+str(params_inst.execute_main_script['learn'])+' | '+str(params_inst.execute_main_script['shell_analysis'])

raw_input("Continue?")


# Setting up Parallel computing, using MPI for python.
# -----------------------------------------------------------------
"""
from mpi4py import MPI
comm = MPI.COMM_WORLD                   # setup MPI framework
n_workers = comm.Get_size()             # total number of workers / parallel processes
rank = comm.Get_rank() + 1              # id of this worker -> master:
"""
# Temporary solution:
rank = 1






# Import all our main function classes
# -----------------------------------------------------------------
	# ambient_speech.main() being the main function for the ambient speech setup.
from src.ambient_speech.ambient_speech import ambient_speech as ambient_speech_class
	# same for 'hear'
from src.hear.hear import hear as hear_class
	# .. and 'learn'
from src.learn.learn import learn as learn_class






if params_inst.execute_main_script['ambient_speech']:
	# Set up and / or analyze a group of speakers serving as ambient speech for the ESN network (used in the hearing stage)
	# ---------------------------------------------------------------------------------------------------------------------
	ambient_speech_inst = ambient_speech_class() 				#Inherit

	##########################
	ambient_speech_inst.main()
	##########################






if params_inst.execute_main_script['hear']:
	# Perform the ESN learning (train the network to classify speech sounds).
	# ---------------------------------------------------------------------------------------------------------------------
	hear_inst = hear_class()

	##########################
	hear_inst.main()
	##########################






if params_inst.execute_main_script['learn']:
	# Babble away..
	# ---------------------------------------------------------------------------------------------------------------------
	learn_inst = learn_class()

	##########################
	learn_inst.main()
	##########################











######### ANALYSIS ############
# Some small analyzing steps.. Include at will (e.g. load classes again.. look at plots)
# --------------------------------------------------------------------------------------------

if params_inst.execute_main_script['shell_analysis']:
	
	import pickle
	from pdb import set_trace as debug
	import numpy
	from matplotlib import pylab as plt

	# Reload a class of hear data. ESNs trained on 3 vowels only. Using srangefm_2, all speakers 
	ta = pickle.load(file('data/classes/hear_instance_3Vowels_allSpeakers.pickle','r'))
	
	# Reload a class of hear data. ESNs trained on all 5 vowels. Using srangefm_2, all speakers 
	aa = pickle.load(file('data/classes/hear_instance_allVowels_allSpeakers.pickle','r'))
	
	# Reload a class of hear data. ESNs trained on all 5 vowels. Using srangefm_2, all speakers
	# except speakers 0,1,2,3 for training, and using these for testing. (speaker generalisation)
	a2 = pickle.load(file('data/classes/hear_instance_allVowels_SpeakersG4.pickle','r'))

	
	
	class ts_class():
		"""
		Res sizes, Errors and stddevs from M. Murakami (100 iterations, 1-1000 N, 3 Vowels, Standard speakers only)
		"""
		def __init__(self):
			self.reservoir_sizes = [1,5,10,20,50,100,200,500,1000]
			self.final_errors = {'leaky':[0.675119047619,0.53630952381,0.38,0.198214285714,0.0940476190476,0.0794047619048,0.0802380952381,0.0788095238095,0.0772619047619]}
			self.final_stds = {'leaky':[0.0786308397317,0.0801673243075,0.0888270760135,0.0585001259762,0.03102098624,0.0311374958979,0.0278998451706,0.0286100928864,0.0251128405782]}
		
	
	ts = ts_class()
	
	
	
	
	
	print "Example list of class keys:"
	print ta.__dict__.keys()
	
	
	
	
	
	# Get lacking data..
	
	aa.final_errors['leaky'] = numpy.average(aa.errors['leaky'],axis=0)
	aa.final_stds['leaky'] = numpy.std(aa.errors['leaky'], axis=0)
	
	
	print "Reservoir sizes:"
	print ta.reservoir_sizes
	print aa.reservoir_sizes
	print ts.reservoir_sizes
	print a2.reservoir_sizes
	
	print "Final errors:"
	print ta.final_errors['leaky']
	print aa.final_errors['leaky']
	print ts.final_errors['leaky']
	print a2.final_errors['leaky']
	
	print "Final standard deviations of the errors:"
	print ta.final_stds['leaky']
	print aa.final_stds['leaky']
	print ts.final_stds['leaky']
	print a2.final_stds['leaky']
	
			
	plotfile = "results/hear/errors_compared.png"
	print "Plotting to: "+plotfile
	
	
	s = plt.figure()
	sf = s.add_subplot(111)
	
	sf.errorbar(ts.reservoir_sizes, ts.final_errors['leaky'], ts.final_stds['leaky'], marker='o', color="k", label='M. Murakami: 3 Vowels. Std. infant, adult speakers')
	sf.errorbar(ta.reservoir_sizes, ta.final_errors['leaky'], ta.final_stds['leaky'], marker='o', color="b", label='3 Vowels. srangefm_2')
	sf.errorbar(aa.reservoir_sizes, aa.final_errors['leaky'], aa.final_stds['leaky'], marker='o', color="g", label='5 vowels. srangefm_2')
	sf.errorbar(a2.reservoir_sizes, a2.final_errors['leaky'], a2.final_stds['leaky'], marker='o', color="m", label='5 vowels. srangefm_2 train: (4-20yrs) test: (0-2yrs)')
	
	
	sf.set_xscale('log')
	plt.xticks([1,10,100,1000,2000],['1','10','100','1000','2000'])
	plt.yticks([0.,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0],['0%','10%','20%','30%','40%','50%','60%','70%','80%','90%','100%'])
	
	plt.xlabel('Reservoir size')
	plt.ylabel('Error rate')
	
	#sf.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.,prop={'size':6})
	sf.legend(prop={'size':10})
	plt.ylim(0, 1)
	plt.xlim(0.8,3000)
	s.savefig(plotfile)

	











print 30*"-"
print "All shell scripts executed!"
print 30*"-"
print 30*"-"
print "\n\n"


"""
# Change back to normal standard out (to terminal)
sys.stdout = sys.__stdout__
"""

