# import the pyplot and wavfile modules

import matplotlib.pyplot as plt
from scipy.io import wavfile
import numpy as np

# Read the wav file (mono)

samplingFrequency, signalData = wavfile.read('bbar7a_7.wav')

# Plot the signal read from wav file

window = np.hamming(64)

plt.specgram(signalData, NFFT=64, Fs=samplingFrequency, window=window, noverlap=48, cmap='jet')

plt.axis('off')
frame = plt.gca()
frame.axes.get_yaxis().set_visible(False)
frame.axes.get_xaxis().set_visible(False)


plt.savefig("bbar7a_7.jpg", bbox_inches="tight")