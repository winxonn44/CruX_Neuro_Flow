import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt, iirnotch
import glob
import os

def plot_with_highlights(filename, fs=250):
    print(f"Generating Highlighted Graphs for: {filename}...")

    # 1. LOAD DATA
    try:
        df = pd.read_csv(filename, sep='\t', header=None)
        raw_data = df.iloc[:, 1:9].values
    except Exception as e:
        print(f"Error reading {filename}: {e}")
        return

    n_samples, n_channels = raw_data.shape

    # 2. FILTERING
    nyquist = fs / 2
    b_analysis, a_analysis = butter(4, [0.5 / nyquist, 30.0 / nyquist], btype='bandpass')
    b_notch, a_notch = iirnotch(60 / nyquist, Q=30)
    
    clean_data = np.zeros_like(raw_data)
    for i in range(n_channels):
        sig = filtfilt(b_analysis, a_analysis, raw_data[:, i])
        sig = filtfilt(b_notch, a_notch, sig)
        clean_data[:, i] = sig

    # 3. CALCULATE FFT & INDICES
    window_size = 1 * fs
    step_size = 1 * fs
    
    timestamps = []
    workload_indices = []
    accumulated_fft = np.zeros((window_size // 2 + 1, n_channels))
    valid_epochs = 0
    
    FRONTAL = [0, 1, 2] # Fz, F1, F2
    PARIETAL = [3, 4, 5] # Pz, P1, P2

    for start in range(0, n_samples - window_size, step_size):
        end = start + window_size
        epoch = clean_data[start:end, :]
        
        if np.max(np.abs(epoch)) > 100.0: continue 

        # FFT
        window = np.hanning(window_size)
        epoch_mags = []
        for ch in range(n_channels):
            signal = epoch[:, ch] * window
            fft_vals = np.fft.rfft(signal)
            mag = np.abs(fft_vals) / window_size
            epoch_mags.append(mag)
        epoch_mags = np.array(epoch_mags).T
        
        accumulated_fft += epoch_mags
        valid_epochs += 1
        
        # Indices
        freqs = np.fft.rfftfreq(window_size, d=1.0/fs)
        def get_pwr(mags, low, high):
            idx = np.logical_and(freqs >= low, freqs <= high)
            return np.sum(mags[idx]**2)

        f_theta = np.mean([get_pwr(epoch_mags[:, ch], 4, 8) for ch in FRONTAL])
        p_alpha = np.mean([get_pwr(epoch_mags[:, ch], 8, 12) for ch in PARIETAL])
        
        wi = f_theta / p_alpha if p_alpha > 0 else 0
        timestamps.append(end / fs)
        workload_indices.append(wi)

    if valid_epochs == 0: return

    avg_spectrum = accumulated_fft / valid_epochs
    freqs = np.fft.rfftfreq(window_size, d=1.0/fs)

    # --- PLOT 1: SPECTRAL REGIONS ---
    fig, axes = plt.subplots(4, 2, figsize=(15, 12))
    axes = axes.flatten()
    chan_labels = ['Fz (Focus)', 'F1', 'F2', 'Pz (Fatigue)', 'P1', 'P2', 'Cz (Flow)', 'O2 (Visual)']
    
    for i in range(8):
        ax = axes[i]
        ax.plot(freqs, avg_spectrum[:, i], color='#333333', linewidth=1.5)
        ax.set_title(f'Ch {i+1}: {chan_labels[i]}')
        ax.set_xlim(1, 30)
        ax.grid(True, alpha=0.3)
        
        # HIGHLIGHT: Theta (Focus)
        ax.axvspan(4, 8, color='yellow', alpha=0.3, label='Focus (Theta)' if i==0 else "")
        # HIGHLIGHT: Alpha (Relax)
        ax.axvspan(8, 12, color='cyan', alpha=0.2, label='Relax (Alpha)' if i==0 else "")
        
        if i == 0: ax.legend(loc='upper right')

    plt.tight_layout()
    plt.suptitle(f'FFT with Highlighted Regions - {os.path.basename(filename)}', y=1.02)
    plt.show()

    # --- PLOT 2: FOCUS ZONEzS IN TIME ---
    plt.figure(figsize=(12, 5))
    plt.plot(timestamps, workload_indices, label='Workload Index', color='purple', linewidth=2)
    
    # Dynamic Threshold for "High Focus"
    threshold = np.mean(workload_indices) + 0.5 * np.std(workload_indices)
    plt.axhline(threshold, color='green', linestyle='--', label='Focus Threshold')
    
    # HIGHLIGHT: Fill area above threshold
    plt.fill_between(timestamps, workload_indices, threshold, 
                     where=(np.array(workload_indices) >= threshold),
                     interpolate=True, color='green', alpha=0.3, label='Focus Zone')

    plt.title(f'Focus Zones Over Time - {os.path.basename(filename)}')
    plt.xlabel('Time (s)')
    plt.ylabel('Workload Index')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

# Run
files = glob.glob("data\\raw\\*.csv")
for f in files:
    plot_with_highlights(f)
