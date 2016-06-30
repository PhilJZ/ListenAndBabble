		

		ListenAndBabble (forked repo): Introduction
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









		SETUP:
		------------------------------------------------------


The code is written in Python 2.7 and was tested in Ubuntu 16.04. Apart from standard Python libraries, it requires:

- [numpy](http://sourceforge.net/projects/numpy/files/NumPy/)
- [scipy](http://sourceforge.net/projects/scipy/files/scipy/)
- [matplotlib](http://matplotlib.org/downloads.html)
#Install all three opening a terminal and typing:
>> sudo apt-get install python-numpy python-scipy python-matplotlib

- [brian](http://brian.readthedocs.org/en/latest/installation.html)
>> sudo apt-get install python-brian python-brian-doc

- [Oger](http://reservoir-computing.org/installing_oger)
>> sudo apt-get install python-mdp  #Getting a dependency
#Unpack tarball after downloading from website (see above)
>> cd [unpacked-directory]
>> sudo python setup.py install

- [mpi4py](https://pypi.python.org/pypi/mpi4py) [this current version used the python joblib]
>> sudo apt-get install python-setuptools
>> easy_install joblib

- [docopt](https://pypi.python.org/pypi/docopt) [not used anymore in this fork]
# no need to install

#The following two dependencies are replaceable, as soon as a VTL version comes out that outputs formants of speech sounds. As for now, we get our formants using praat, and praat_formants_python
- [praat](http://www.fon.hum.uva.nl/praat/download_linux.html)
>> sudo apt install praat

- [praat_formants_python](https://github.com/mwv/praat_formants_python)
#Unpack tarball (download from git repo)
>> cd [unpacked-directory]
>> sudo python setup.py install


Additionally, you'll need the VocalTractLab API for Linux, which you can download [here](http://vocaltractlab.de/index.php?page=vocaltractlab-download).
After downloading it, extract the VocalTractLabApi.so into the src/VTL_API subfolder. (should already be there, though)


Scripts are executed from the shell.py script. On first execution, uncomment the part where the directory (where you saved your fork of the project) is added to sys.path (including subfolders).

Parameters are in the parameter file in params. Edit these if you want to change things (e.g. if you only want to learn one vowel).









		WHAT 'ListenAndBabble' DOES:
		------------------------------------------------------

The project currently contains 3 steps: 'ambient_speech', 'hear' and 'learn'

		ambient_speech
'ambient_speech' sets up a group of speakers (created in VTL_API) for the next steps.
	Notes: 	New speaker groups (standard: srangefm "speaker range (from 0-20yrs) female 
	 	and male") can be added by saving VTL-produces speakers with the same names
	 	["0","1","2",..] and saving them in data/speakers_gestures under the new group
	 	name. (fileage.txt has to be adjusted. see speaker documentation in that folder.)

	 	Speech samples are produced. some are representative for vowels, some not.
	 	The user must check produced samples and - if nescessary (e.g. if a classified
	 	/a/ sample in folder data/ambient_speech/"group name"/a sounds like nothing at
	 	 all) - relocate them to the right folder (in that case to the ../a_null folder)

		hear
'hear' listens to ambient speech samples (from first step), and trains an Echo state network to classify those sounds correctly. [Supervised learning]

		learn
'learn' applies reinforcement learning [unsupervised] in order to learn to produce "good" vowels.
	Notes:	The learner, in this version, is simply the first


See ... [Master thesis] for more information and suggestions for improvement / extensions of this project.

Enjoy!


		A typical experiment guide..
		------------------------------------------------------

1. 	Create a series of speakers of various ages, using VocalTractLab. (Run .exe on Windows, or over wine on Linux.) 
	These, obviously, will have different anatomies. That means, we have to find the right positions for all shape parameters (tongue position, lip, etc).
	Get a feeling of the gesture shapes (e.g. "how does an /a/ look like in a normal speaker) by looking at some of the standard speakers in the VTL API, or
	by looking at previous speaker series. Then tune all the (vowel-) gestures, already using the right fundamental frequency (the one you'll use for each speaker
	in the experiments!). A certain shape may sound much like an /i/ (e.g.) at - say - f0 = 300, but sound very different at lower/higher frequencies!
	
2.	Produce your ambient speech (using 'src/ambient_speech/..' scripts called from shell - see source code documentation). Ambient speech consists of prototype sounds (1 .wav file
	for each vowel and speaker (+ one .txt and a gunzipped file.))
	
3.	Use the prototypes to guide you in making (often nescessary) changes in your shapes.
	
4.	When you are sure about your prototypes, create sample sounds based on those prototypes (much the same formats, only more than one sample per speaker. These samples are not exactly
	like the prototype sounds.
	
5.	Train the ESN with mult. ESNs of the same sizes > Determine best size.

6.	Train ESNs until you get a good one.

7.	Learn
	
	


		Source code documentation..
		------------------------------------------------------





