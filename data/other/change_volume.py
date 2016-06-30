import wave, numpy, struct

# Open
w = wave.open("whitenoise_original.wav","rb")
p = w.getparams()
f = p[3] # number of frames
s = w.readframes(f)
w.close()

# Edit
s = numpy.fromstring(s, numpy.int16) / 10 * 20
s = struct.pack('h'*len(s), *s)

# Save
w = wave.open("whitenoise.wav","wb")
w.setparams(p)
w.writeframes(s)
w.close()
