import os



"""
Ambient speech samples have already been produced. The next step is to train the network. For this, we'll 
1. save the ambient speech in the right locations, 
2. call "learndata.py" (the main ESN-learning script, programmed by Max Murakami), and 
3. save required results in our results folder.

"""

# Remove all preexisting data, in order to make place for the ambient speech samples
os.system('rm -r src/hear/data/*')
# Store ambient speech in folders acceptable for learndata.py
os.system('mv data/output/ambient_speech/%s_samples/* src/hear/data/)



# In order to call learndata.py we have to change the current working directory
main_directory = os.getcwd()
print main_directory
hear_directory = main_directory+'/src/hear'
os.chdir(hear_directory)


# Call learndata function (see doc?)
os.system('python learndata.py 5 --n_samples 6 --n_training 9 --n_reservoirs 1000 --subfolder tutorial')
# trained auditory system stored as current_auditory_system.flow in 'data'
os.chdir(main_directory)

# Move to result directory
#...


