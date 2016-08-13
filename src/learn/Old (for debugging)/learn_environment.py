



# General Imports
import os
from brian import kHz, Hz, exp, isinf
from brian.hears import Sound, erbspace, loadsound, DRNL
from scipy.signal import resample

import numpy as np
import matplotlib
matplotlib.use('Agg')                  # for use on clusters
import Oger
import pylab
from datetime import date
import cPickle

# Module Imports
#	Module to produce wav files of current learner motor configuration
from src.VTL_API.api_class import VTL_API_class
synthesize = VTL_API_class() #"synthesize is-a VTL_API_class"

class environment(object):




	#*********************************************************


	def plot_reservoir_states(self,flow, y, i_target, n_vow, self.rank):

		""" plot reservoir states"""


		current_flow = flow[0].inspect()[0].T # reservoir activity for most recent item
		N = flow[0].self.verbose_dim              # reservoir size

		n_subself.plots_x, n_subself.plots_y = 2, 1   # arrange two self.plots in one column
		pylab.subplot(n_subself.plots_x, n_subself.plots_y, 1)
		                                    # upper plot
		y_min = y.min()
		y_max = y.max()
		if abs(y_min) > y_max:              # this is for symmetrizing the color bar
		        vmin = y_min                # -> 0 is always shown as white
		        vmax = -y_min
		else:
		        vmax = y_max
		        vmin = -y_max

		class_activity = pylab.imshow(y.T, origin='lower', cmap=pylab.cm.bwr, aspect=10.0/(n_vow+1), interpolation='none', vmin=vmin, vmax=vmax)
		                                    # plot self.verbose activations, adjust to get uniform aspect for all n_vow
		pylab.title("Class activations")
		pylab.ylabel("Class")
		pylab.xlabel('')
		pylab.yticks(range(n_vow+1), self.lib_syll[:n_vow]+['null'])
		pylab.xticks(range(0, 35, 5), np.arange(0.0, 0.7, 0.1))
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
		pylab.xticks(range(0, 35, 5), np.arange(0.0, 0.7, 0.1))
		pylab.ylabel("Neuron")
		pylab.yticks(range(0,N,N/7))
		cb2 = pylab.colorbar(reservoir_activity)

		pylab.savefig(self.output_folder+'plots/vowel_'+str(i_target)+'_'+str(self.rank)+'.pdf')

		pylab.close('all')






	def get_confidences(self,mean_sample_vote):

		n_classes = len(mean_sample_vote)
		confidences = np.zeros(n_classes)
		norm_sum = 0.0
		for i in xrange(n_classes):
		    confidence_i = exp(mean_sample_vote[i])
		    confidences[i] = confidence_i
		    norm_sum += confidence_i
		confidences /= norm_sum

		return confidences






	"""
	def get_reward(self,mean_sample_vote, sound_extended, i_target, speaker, loudness_factor, softmax=False):


		if self.softmax:
		    reward = exp(mean_sample_vote[i_target])
		    other_exp = 0.0
		    for i in xrange(len(mean_sample_vote)):
		        if i!=i_target:
		            other_exp += exp(mean_sample_vote[i])
		    reward /= reward + other_exp

		else:
		    reward = mean_sample_vote[i_target]
		    for i in xrange(len(mean_sample_vote)):
		        if i!=i_target:
		            reward -= mean_sample_vote[i]

		    if speaker == 'adult':
		        target_loudness = [72.77, 65.20, 66.04, 68.37, 68.47]
		    else:
		        target_loudness = [73.78, 68.68, 69.78]

		    level = float(sound_extended.level)
		    if isinf(level):
		        level = 0.0
		    loudness_reward = level - target_loudness[i_target]

		    if loudness_reward > 0.0:
		        loudness_reward = 0.0

		    reward += loudness_factor * loudness_reward



		return reward
	"""




	def normalize_activity(x):

		x_normalized = x.copy()
		minimum = x.min()
		maximum = x.max()
		range_ = maximum - minimum
		bias = abs(maximum) - abs(minimum)

		x_normalized -= bias/2.0
		x_normalized /= range_/2.0

		return x_normalized




	##########################################################
	#
	# Main script
	#
	##########################################################



	def evaluate(self,params, simulation_name, i_target=0, self.rank=1, speaker='srangefm 0', n_vow=5, normalize=False):
		
		
		
		
		############### Sound generation

		if self.verbose:
		 print 'simulating vocal tract'
		
		input_dict = {	'gesture':vowel,
						'group_speaker':group_speaker,
						'pitch_var':self.speaker_pitch_rel[speaker],
						'verbose':True }
		paths = {		'input_path':self.learner_path,
						'wav_folder':self.output_path+'/'+vowel }
		
		synthesize.main(input_dict,paths)
		
		
		wavFile = synthesize_wav.main(	params, speaker, simulation_name,pitch_var=0,len_var=1.0,verbose=self.verbose,self.rank=self.rank,
										different_output_path=self.output_folder)
	#    wavFile = par_to_wav(params, speaker, simulation_name, verbose=self.verbose, self.rank=self.rank) # call parToWave to generate sound file
		if self.verbose:
		 print 'wav file '+str(wavFile)+' produced'

		sound = loadsound(wavFile)          # load sound file for brian.hears processing
		if self.verbose:
		 print 'sound loaded'



		############### Audio processing

		sound = correct_initial(sound)      # call correct_initial to remove initial burst

		sound_resampled = get_resampled(sound)
		                                    # call get_resampled to adapt generated sound to AN model
		sound_extended = get_extended(sound_resampled)
		                                    # call get_extended to equalize duration of all sounds
		sound_extended.save(wavFile)        # save current sound as sound file

		os.system('cp '+wavFile+' '+folder+'data/vowel_'+str(i_target)+'_'+str(self.rank)+'.wav')

		if self.playback:
		    print 'playing back...'
		    sound_extended.play(sleep=True) # play back sound file

		if self.verbose:
		 print 'sound acquired, preparing auditory processing'

		out = drnl(sound_extended)          # call drnl to get cochlear activation



		############### Classifier evaluation

		flow_name = 'data/current_auditory_system.flow'
		flow_file = open(flow_name, 'r')    # open classifier file
		flow = cPickle.load(flow_file)      # load classifier
		flow_file.close()                   # close classifier file

		sample_vote_unnormalized = flow(out)                       # evaluate trained self.verbose units' responses for current item
		if normalize:
		    sample_vote = normalize_activity(sample_vote_unnormalized)
		else:
		    sample_vote = sample_vote_unnormalized
		mean_sample_vote = np.mean(sample_vote, axis=0)
		                                    # average each self.verbose neurons' response over time


		confidences = get_confidences(mean_sample_vote)

		plot_reservoir_states(flow, sample_vote, i_target, n_vow, self.rank)


		return confidences
