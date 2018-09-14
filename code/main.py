from pydub import AudioSegment
import os
import librosa
import matplotlib.pyplot as plt
from scipy.io import wavfile
import numpy as np


def gen_spectrumgram(folder, filename):

    spec_filename = filename[:len(filename) - 4] + ".png"

    samplingFrequency, signalData = wavfile.read(folder + filename)

    # Plot the signal read from wav file
    window = np.hamming(64)

    plt.specgram(signalData, NFFT=64, Fs=samplingFrequency, window=window, noverlap=48, cmap='jet')

    plt.axis('off')
    frame = plt.gca()
    frame.axes.get_yaxis().set_visible(False)
    frame.axes.get_xaxis().set_visible(False)

    #   check directory is existed
    directory = folder + 'specgram/'
    if not os.path.exists(directory):
        os.makedirs(directory)
    plt.savefig(directory + spec_filename, bbox_inches="tight")
    plt.clf()


def audio_seg(folder, filename, start, end, total):

    ds_filename = filename[:len(filename)-4] + "_ds.wav"
    digit = filename[len(filename) - 6: len(filename) - 5]
    seg_filename = filename[:len(filename) - 4] + "_" + str(digit) + ".wav"
    #   audio file down sampling
    y, s = librosa.load(folder + filename, sr=5000)
    librosa.output.write_wav(folder + ds_filename, y, s)
    #   audio file segmentation
    myaudio = AudioSegment.from_file(folder + ds_filename)
    a_len = len(myaudio)
    start_time = int(a_len * start / total)
    end_time = int(a_len * end / total)
    word = myaudio[start_time:end_time]
    #   check directory is existed
    directory = folder + 'result/'
    if not os.path.exists(directory):
        os.makedirs(directory)
    word.export(directory + seg_filename, format="wav")
    #   delete the down sampling file
    os.remove(folder + ds_filename)

def seg_wav_files():
    filepath = "../dataset/"
    foldernames = os.listdir(filepath)
    for folder in foldernames:
        filenames = os.listdir(filepath + folder)
        for filename in filenames:
            if len(filename) > 5 and filename[-4:] == ".wav":
                align_filename = filepath + folder + '/align/' + filename[:len(filename) - 4] + '.align'
                print(align_filename)
                #   read the align file
                try:
                    f = open(align_filename, 'r')
                    content = f.readlines()
                    digits = content[5].split(" ")
                    last = content[7].split(" ")
                    start = digits[0]
                    end = digits[1]
                    total = last[1]

                    print("%s - %s - %s" % (start, end, total))
                    print(filepath + folder + '/')
                    audio_seg(filepath + folder + '/', filename, int(start), int(end), int(total))

                    f.close()

                    digit = filename[len(filename) - 6: len(filename) - 5]
                    seg_filename = filename[:len(filename) - 4] + "_" + str(digit) + ".wav"
                    gen_spectrumgram(filepath + folder + '/result/', seg_filename)

                except IOError as err:
                    print('File Error:' + str(err))

                print(filepath + folder + '/' + filename)


if __name__ == '__main__':
    seg_wav_files()
