


# Class Imports
# -------------------------------------------------------
	# Import the class(es) containing all the called functions and inherit all subfunctions
from src.learn.learn_functions import functions as funcs



# General Imports






class learn(funcs):
	"""
	learn.main() does the reinforcement learning of shape parameters.
	
	
	CMA-ES: Evolution Strategy with Covariance Matrix Adaptation for
	nonlinear function minimization.

	This code refers to "The CMA Evolution Strategy: A Tutorial" by 
	Nikolaus Hansen (Appendix C).


	Of the old version:
	# usage:
	#  $ salloc -p sleuths -n (lambda/int) mpirun python rl_agent_mpi.py [-v] [-n (n_samples/int)] [-f (folder/str)] [-t (target/str)]
	#	 [-p (parameters/str)] [-s (sigma/float)] [-i] [-N (self.n_targets/int)] [-m] [-I] [-T (threshold/float)] [-P] [-e (energy_factor/float)]
	#	 [-a (alpha/float)] [-F]

	# thesis settings:
	#   salloc -p sleuths -n 100 mpirun python rl_agent_mpi.py -f default_output_folder -p all -i -m -I -r -o -A -w -c
	
	
	- changed and adapted by Philip Zurbuchen 2016.
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
	
		if self.rank == 0:
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
		self.get_population_size()
		print 80*"-"
		
		
		
		for i_trial in xrange(self.n_trials):
			
			
			# The reinforcement learning is done quite differently, if we have more than one workers. First, look
			# at the case of the master (rank==0). If we have more than one workers, the rest will act as slaves, 
			# performing generation sampling (see else:)
			# If there's only one worker anyway, that worker will be rank == 0, master.
			if self.rank==0:
				
				# Perform CMA-ES. 
				x_min = self.cmaes()
				
			else:
			
				# If slave worker: Only sample, and return confidences etc. to the master
				#self.generation_sampling()
				pass
				
				
				
		print "\n"
		print 80*"-"
		print "Saving class instance"
		f = open('data/classes/learn_instance.pickle', 'w+')
		f.truncate()
		pickle.dump(self,f)
		f.close()
		print 80*"-"	
		
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
