		

#ListenAndBabble (forked repo): Introduction
------------------------------------------------------

Attention: This repository is still work in progress.

See README of forked repo for the model published in 'Murakami et al. (2015)': "Seeing [u] aids vocal learning: babbling and imitation of vowels using a 3D vocal tract model, reinforcement learning, and reservoir computing." International Conference on Development and Learning and on Epigenetic Robotics 2015 (in press).'

Code taken from the original project has been adapted and restructured in order to make it more widely applicable (to include whole series of speakers, etc.). 
Also, a whole new step was added to the project: "ambient speech" which
-sets up a speaker group produced in VTL, 
-adjusts various parameters (like voice-pitch of each speaker),
-creates ambient speech sounds (which can be used to train the echo-state network of the reinforcement learner) and 
-analyses various characteristics
	of the speakers, 
	of shape parameters (like 'toungue position when saying /a/') and 
	of the speech sounds themselves (e.g. plotting vowels in formant space).

Much of the code from the original repo has been compartmentalized, making it easier for a user to find his/her way around the code.

A small documentation of the code follows the SETUP chapter (where all the required python packages are listed, in order that the code works).




SETUP:
------------------------------------------------------


The code is written in Python 2.7 and was tested in Ubuntu 16.04. Apart from standard Python libraries, it requires:

- [numpy](http://sourceforge.net/projects/numpy/files/NumPy/)
- [scipy](http://sourceforge.net/projects/scipy/files/scipy/)
- [matplotlib](http://matplotlib.org/downloads.html)
Install all three opening a terminal and typing:
>> sudo apt-get install python-numpy python-scipy python-matplotlib

- [brian](http://brian.readthedocs.org/en/latest/installation.html)
>> sudo apt-get install python-brian python-brian-doc

- [Oger](http://reservoir-computing.org/installing_oger)
>> sudo apt-get install python-mdp  #Getting a dependency
Unpack tarball after downloading from website (see above)
>> cd [unpacked-directory]
>> sudo python setup.py install

- [mpi4py](https://pypi.python.org/pypi/mpi4py) [this current version used the python joblib]
>> sudo apt-get install python-setuptools
>> easy_install joblib

- [docopt](https://pypi.python.org/pypi/docopt) [not used anymore in this fork]
(no need to install)

The following two dependencies are replaceable, as soon as a VTL version comes out that outputs formants of speech sounds. As for now, we get our formants using praat, and praat_formants_python
- [praat](http://www.fon.hum.uva.nl/praat/download_linux.html)
>> sudo apt install praat

- [praat_formants_python](https://github.com/mwv/praat_formants_python)
Unpack tarball (download from git repo)
>> cd [unpacked-directory]
>> sudo python setup.py install

For some of the plotting, I used 'plotly'. In order to use plotly, you have to install, and register (free)!
- [plotly](https://plot.ly/python/getting-started)
Otherwise, deactivate those plots.



Additionally, you'll need the VocalTractLab API for Linux, which you can download [here](http://vocaltractlab.de/index.php?page=vocaltractlab-download).
After downloading it, extract the VocalTractLabApi.so into the src/VTL_API subfolder.

If the user wishes to create and manipulate speakers of his/her own, she/he must download the whole Vocaltractlab program (same link). (Works on windows. Executable on linux using 'wine').



WHAT 'LISTENANDBABBLE' DOES:
------------------------------------------------------

The project currently contains 3 steps: 'ambient_speech', 'hear' and 'learn'

		ambient_speech

In 'ambient_speech', speaker files (created with VocalTractLab), with predefined (vowel) gestures are imported. The user can analyze the speech sounds produced by
the speakers, and 'ambient speech' (used to train the auditory system of the reinforcement learner described in 'ListenAndBabble') is produced in the form of many
samples (.wav files with speech sounds). These can be classified by the user and labeled with the correct label (i.e. '/a/').

		hear
'hear' listens to ambient speech samples (from first step), and trains an Echo state network to classify those sounds correctly. [Supervised learning]

		learn
'learn' applies reinforcement learning [unsupervised] in order to learn to produce "good" vowels.


See ... [Master thesis] for more information and suggestions for improvement / extensions of this project.



A TYPICAL EXPERIMENT..
------------------------------------------------------

1. 	Create a series of speakers of various ages, using VocalTractLab. (Run .exe on Windows, or over wine on Linux.) 
	These, obviously, will have different anatomies. That means, we have to find the right positions for all shape parameters (tongue position, lip, etc).
	Get a feeling of the gesture shapes (e.g. "how does an /a/ look like in a normal speaker) by looking at some of the standard speakers in the VTL API, or
	by looking at previous speaker series. Then tune all the (vowel-) gestures, already using the right fundamental frequency (the one you'll use for each speaker
	in the experiments!). A certain shape may sound much like an /i/ (e.g.) at - say - f0 = 300, but sound very different at lower/higher frequencies!
	
2.	Produce your ambient speech (using 'src/ambient_speech/..' scripts called from shell - see source code documentation). Ambient speech consists of prototype sounds (1 .wav file
	for each vowel and speaker (+ one .txt and a gunzipped file.))
	
3.	Use the prototypes to guide you in making (often nescessary) changes in your shapes.
	
4.	When you are sure about your prototypes, create sample sounds based on those prototypes (gaussian sampling of the parameters used to create the prototype speech).

6.  Now, since 'hear' works with supervised learning: Label all those samples with the right vowel name. 'user_sort_samples' in ambient_speech_functions.
	
5.	Use 'hear' to train the ESN with mult. ESNs of different sizes > Determine best.

6.	Train ESNs of best size until you get a good one.

7.	Using 'learn', use reinforcement learning to learn to reproduce good sounds. The learner can be one of the speakers you created in your speaker group (usually, the youngest, or speaker 0).
	Check the progress in data/output/learn, where you can listen to the best of each produced vowels. In results/learn/ you can see the current progress of the learning for each vowel.
	
	


SOURCE CODE DOCUMENTATION
------------------------------------------------------

		Architecture
First, we must understand how the code is structured. To make this easier to comprehend, I applied the same structure to the task of computing the square root of some arbitrary number. A pdf presenting the code and explaining the structure is found in the subfolder control. This project is basically structured in the same way (though more complicated). Each step (ambient speech, hear, learn) of the project is executed from the shell (.py) and controlled from _control/get params.py_ by the user.
The actual functions, - what actually happenes, are found in (e.g.) src/ambient-speech/ambient-speech-_functions_. Each of those functions are called from one level higher, the script ambient-speech.py (in the same directory). This script, in turn, is called from the shell.


		Parallelization
This branch is not sep up for parallelisation. Executing ambient_speech and hear in parallel doesn't seem worth the effort. However, doing the reinforcement learning on a cluster seems reasonable (as in the original branch). Though not implemented, what seems best is to do the following:

In learn_functions:

The generations could be assessed in a parallel way.
(Steps: evaluation, environment, get_confidence)
These steps could be written in a separate skript which would take parameters like x_mean as arguments, and return confidences.
The rest (cma-es learning loop) would not be run in parallel. 
Then, this separate script could be executed using os.system from the cma-es while loop. Like this, for example:
os.system('salloc -p sleuths -n (lambda/int) mpirun python [name of script, that imcorporates evaluation, environment and confidence].py [arguments] ... ')


		Further documentation
I tried my best to document the code itself extensively where needed. Knowing the project architecture well will also help to understand the code. I advise to read the shell script, then, also most control parameters in control/get-params.py are commented on. Reading the main step scripts ('ambient-speech.py', 'hear.py', 'learn.py') will tell the user, WHICH function is executed WHEN.


		Speaker groups documentation
The subfolder data/[backups VTL speaker groups contains a speaker group documentation (.txt file). In it some information on the speaker group (age of speakers, pitch, ..) - and how to use the speaker files in the code.












