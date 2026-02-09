import numpy as np
import mne
import mne_bids
import matplotlib.pyplot as plt

from src.processing.FFT import raws
from src.processing.artifacts import epochs

x = []
y= []

for key, epoch in epochs.items():
    # Extract the alpha band power
    psds, freqs = mne.time_frequency.psd_array_welch(epoch.get_data(), sfreq=raws[key].info['sfreq'],fmin=6,fmax=9)
    alpha_band_power = np.mean(psds, axis=-1)

    # Store the alpha band power in a matrix
    x.append(alpha_band_power)

    if key == 'oa_epochs':
        label = 1
    else:
        label = 0

    y.append(np.full(alpha_band_power.shape[0], label))

X = np.vstack(x)
y = np.concatenate(y)

print(x.shape)
print(y.shape)