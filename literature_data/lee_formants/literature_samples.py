

import pickle
from os import system,path,listdir,getcwd    		# for filehandling and exec-documentation
import numpy
import pdb
import matplotlib.pyplot as plt

# Extract Formants from male.txt and female.txt
# ----------------------------------------------------------------------------------



#pdb.set_trace()
for gender in ['male','female']:
	
	vowels = []
	formants = {'/a/':[],'/e/':[],'/i/':[],'/u/':[]}
	forma = []
	
	with open(gender+'.txt') as f:
		for line in f:
			if line[0] == '/':
				current_vowel = line[0:3]
				vowels.append(current_vowel)
			else:
				f,dev = line.split()
				dev = dev[1:-1]
				forma.append(tuple((int(f),int(dev))))
				
				if len(forma)==4:
					# Add these formants to the matrix of formants over age.
					formants[current_vowel].append(forma)
					
					forma = []
					
	
	if gender == 'male':
		male_formants = formants.copy()
	else:
		female_formants = formants.copy()
	
	# Since we're looking at the american pronounciation. Eh is not our german /e/
	vowels_phonetic=vowels[:]
	vowels_phonetic[1] = "/eh/"
	
	# Plot parameters
	ages = [5,6,7,8,9,10,11,12,13,14,15,16,17,18,19] # Used for size of circles
	plot_colors = {"/a/":"m", "/e/":"#4682b4", "/i/":"y", "/u/":"b"}
	plt.close("all")
	#Set up the plot
	f = plt.figure()
	plt.title("Vowel formants of %s series (Lee et al.)"%gender)
	fsub = f.add_subplot(111)
	scatters = []
	for vowel in vowels:
		x = []
		y = []
		xerr = []
		yerr = []
		#The things we want to plot..
		for i in range(len(ages)):
			# Extract f1
			x.append(formants[vowel][i][1][0])
			xerr.append(formants[vowel][i][1][1])
			# Extract f2
			y.append(formants[vowel][i][2][0])
			yerr.append(formants[vowel][i][2][1])
			
		
#		# F1-F2 ?
#		for i in range(len(y)):
#			y[i] = y[i]-x[i]
		
		msize = (numpy.array(ages)+7)**2
		new_scatter = fsub.scatter(x,y,marker='o',c=plot_colors[vowel],s=msize,label=vowel,alpha = 0.7)
		fsub.errorbar(x,y,xerr=xerr,yerr=yerr,c=plot_colors[vowel],alpha=0.5)
		scatters.append(new_scatter)
		
	plt.xlim(200,1200)
	plt.ylim(500,3500)
	plt.legend(scatters,vowels_phonetic)
	plt.xlabel("F1 [Hz]")
	plt.ylabel("F2 [Hz]")

	f.savefig('literature_formants_%s_5to19.png'%gender)
"""
print "\n"
print 80*"-"
print "Pickle this class"
f = open('literature_formants.pickle', 'w+')
f.truncate()
pickle.dump(self,f)
f.close()
print 80*"-"
"""
