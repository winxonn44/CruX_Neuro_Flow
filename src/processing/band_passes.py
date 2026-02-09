import numpy as np
import pandas as pd
import scipy as sp
import mne
import mne_bids
import matplotlib.pyplot as plt

from scipy import signal
from src.processing.FFT import FFT





def bandpass(data, lowcut, highcut, fs, order=5):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    sos = signal.butter(order, [low, high], btype='band', output='sos')
    return signal.sosfiltfilt(sos, data)

filter = bandpass(signal_data, 0.5, 30, fs) #again, fs and signal_data, and t have to be defined, idk for now

plt.figure(figsize=(10, 6))
plt.plot(t, signal_data, label='Original')
plt.plot(t, band_filtered, label='Bandpass 0.5-30 Hz')
plt.legend()
plt.xlabel('Time (s)')
plt.ylabel('Amplitude')
plt.show()