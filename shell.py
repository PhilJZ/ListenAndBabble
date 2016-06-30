"""
A shell for calling source code which..
	-produces ambient speech for the learner ("ambient speech")
	-trains the ESN network ("hear")
	-and babbles away, exploring vocal tract parameters and learning specific syllables ("learn")
Author: Max Murakami and Philip Zurbuchen


On MPI run as follows:
-------------------------------------------------------------------
$ salloc -p sleuths -n (lambda/int) mpirun python shell.py
-------------------------------------------------------------------
"""

# General imports
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



# Write standard output of our programme written into a text file in the results directory.
"""
os.remove("results/shell_output.out")
f = open("results/shell_output.out", 'w')
sys.stdout = f
"""


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







# Set up and / or analyze a group of speakers serving as ambient speech for the ESN network (used in the hearing stage)
# ---------------------------------------------------------------------------------------------------------------------
#ambient_speech_inst = ambient_speech_class() 				#Inherit

##########################
#ambient_speech_inst.main()
##########################






# Perform the ESN learning (train the network to classify speech sounds).
# ---------------------------------------------------------------------------------------------------------------------
hear_inst = hear_class()

##########################
hear_inst.main()
##########################






# Babble away..
# ---------------------------------------------------------------------------------------------------------------------
#learn_inst = learn_class()

##########################
#learn_inst.main()
##########################

















"""
# Change back to normal standard out (to terminal)
sys.stdout = sys.__stdout__
"""

