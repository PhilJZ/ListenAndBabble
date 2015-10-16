import os					# for data handling via shell calls


filename = str(raw_input('Enter wav file to repair: '))
if filename[-4:] != '.wav':		   # automatically append correct file extension
	filename += '.wav'

if not os.path.exists(filename):		   # check if file exists
	print 'Error: '+filename+' not found!'
else:
	backup = filename+'.bkup'	
	os.system('cp '+filename+' '+backup)	# create backup of wav file

	f = open(filename, 'r')			# open wav file for reading
	content = f.read()			# read content of wav file

	f.close()				# close wav file
	os.remove(filename)			# delete wav file
	os.system('touch '+filename)		# create empty file with the same name

	newfile = open(filename, 'w')		# open new file for writing

	header = 'RIFF'+chr(0x8C)+chr(0x87)+chr(0x00)+chr(0x00)+'WAVEfmt'+chr(0x20)+chr(0x10)+chr(0x00)+chr(0x00)+chr(0x00)+chr(0x01)+chr(0x00)+chr(0x01)+chr(0x00)+'"V'+chr(0x00)+chr(0x00)+'D'+chr(0xAC)+chr(0x00)+chr(0x00)+chr(0x02)+chr(0x00)+chr(0x10)+chr(0x00)+'data'
						# define working header

	newcontent = header+content[68:]	# concatenate header with sound data
	newfile.write(newcontent)		# write new file

	newfile.close()				# close new file
	os.remove(backup)			# delete backup
