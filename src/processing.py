import numpy as np
import scipy as sp
import mne 
import mne_bids


# Need to define tasks for NeuroFlow 
tasks = []

bids_root = 'data/raw/bids'

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