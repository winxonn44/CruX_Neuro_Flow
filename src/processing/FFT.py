import numpy as np
import pandas as pd
import scipy.signal
import scipy as sp
import mne
import mne_bids
import matplotlib.pyplot as plt


# Need to define tasks for NeuroFlow; work shop for CruX did it as tasks = ['eo', 'ec'] for eyes open and closed respectively
tasks = []

bids_root = 'data/raw/eeg_test1' #choose the test we want

raws = {}

for task in tasks:
    bids_path = mne_bids.BIDSPath(subject='2', task=task, suffix='eeg', extension='.edf', root=bids_root)
    raw = mne_bids.read_raw_bids(bids_path=bids_path, extra_params=dict(preload=True))
    raws[task] = raw

    # Drop bad channels, idk which ones are bad
    keep_channels = []

    for ch in raw.ch_names:
        if ch not in keep_channels:
            raw.drop_channels(ch)

data = raw.get_data()
channels = raw.ch_names
sfreq = raw.info['sfreq']
print(data.shape)
print(channels) 
print(sfreq) # Just some sample information and checks

# Now some FFT and signal working; We still need to define 'x' and 'fs' based on the data in the file.

X = np.fft.fft(x)
freqs = np.fft.fftfreq(len(X), 1/fs)

half = len(X)//2 # Only keep positive half of the spectrum. Why? Idk. If it breaks then let's fix it

''' just some code to plot the FFT, in case checks are needed
plt.figure(figsize=(10,4))
plt.plot(freqs[0:half], np.abs(X[0:half]))
plt.title("Frequency-Domain Spectrum")
plt.xlabel("Frequency (Hz)")
plt.ylabel("Magnitude")
plt.grid()
plt.show()

'''

