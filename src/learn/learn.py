


# Class Imports
# -------------------------------------------------------
	# Import the class(es) containing all the called functions and inherit all subfunctions
from src.learn.learn_functions import functions as funcs



# General Imports
import time
import pickle
from os import system



class learn(funcs):
	"""
	learn.main() does the reinforcement learning of shape parameters.
	
	
	CMA-ES: Evolution Strategy with Covariance Matrix Adaptation for
	nonlinear function minimization.

	This code refers to "The CMA Evolution Strategy: A Tutorial" by 
	Nikolaus Hansen (Appendix C).
		
	- changed and adapted by Philip Zurbuchen 2016.
	
	Original code my Max Murakami 2014
	"""
	
	
	def __init__(self):
		"""
		Initialisation (used classes)
		"""
		#Import all functions from learn_functions (imported above) as self.
		funcs.__init__(self)
		
		
	def main(self):
	

		# Preparation..
		# -----------------------------------------------------------------
	
		
		print "\n"
		print 80*"-"
		print "Welcome to 'learn' \n- A reinforcement learner who uses the Auditory System (ESN) produced in 'hear' to learn speech gestures."
		print "The Speech Parameters (16 of them) are as follows: "
		print self.par_names
		print 80*"-"
			
	
		print "\n"
		print 80*"-"
		print "Setting up / cleaning up used directories.."
		print " -- >  calling setup_folders"
		self.setup_folders()
		print 80*"-"


	
		print "\n"
		print 80*"-"
		print "Master sets up the writing files.."
		print " -- >  calling open_result_write"
		#self.open_result_write()
		print 80*"-"

	
		print "\n"
		print 80*"-"
		print "Getting indices of the parameters to learn."
		print " -- >  calling init_par_indices"
		self.init_par_indices_and_dimension()
		print 80*"-"
	
	
		print "\n"
		print 80*"-"
		print "Initializing learner shape parameters."
		print " -- >  calling init_learner_pars"
		self.init_shape_parameters()
		print 80*"-"
	
		
		print "\n"
		print 80*"-"
		print "Setting up multi processing related value: Population size"
		print " -- >  calling get_population_size()"
		if not self.population_size: #If not set by the user.
			self.get_population_size()
		print 80*"-"
		
		
		
		print "\n"
		print 80*"-"
		print "Performing CMA-ES.."
		print " -- >  calling cmaes()"
		result_dict = self.cmaes()
		print 80*"-"

				
				
		print "\n"
		print 80*"-"
		print "Saving class instance"
		f = open('data/classes/learn_instance.pickle', 'w+')
		f.truncate()
		pickle.dump(self,f)
		f.close()
		print 80*"-"	
		
		
		
		print "\n"
		print 80*"-"
		print "Moving peak-data into results folder.."
		system('mv data/output/learn/current_peak results/learn')
		print 80*"-"
		
	
		print "\n"
		print 80*"-"
		print "Thankyou for using the reinforcement learner! You can find your results in results/learn/...!"
		print 80*"-"	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
