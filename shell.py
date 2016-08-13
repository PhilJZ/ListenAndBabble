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
In this script, we only need to know, this basic steps we should execute. 	Should we produce ambient speech, or hear, or both?
																			.. or do we have a trained auditory system, and just want
																			to learn? > Set the corresponding parameters "True" or "False"
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
print "Execute main script 'ambient_speech'? 	: "+str(params_inst.execute_main_script['ambient_speech'])
print "Execute main script 'hear'? 	: "+str(params_inst.execute_main_script['hear'])
print "Execute main script 'learn'? : "+str(params_inst.execute_main_script['learn'])


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












print 30*"-"
print "All shell scripts executed!"
print 30*"-"
print 30*"-"
print "\n\n"


"""
# Change back to normal standard out (to terminal)
sys.stdout = sys.__stdout__
"""

