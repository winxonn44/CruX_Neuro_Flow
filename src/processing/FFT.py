import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import iirnotch, butter, filtfilt
import glob

def plot_linear_fft(filename, fs=250, exclude_channels=None):
    """
    Plots the Standard (Linear) FFT of EEG data.
    
    Parameters:
    - filename: Path to the CSV file.
    - fs: Sampling rate (default 250Hz for OpenBCI).
    - exclude_channels: List of channel numbers to skip (e.g., [1, 2, 8]).
    """
    if exclude_channels is None:
        exclude_channels = []
        
    print(f"Processing {filename} (Excluding Ch: {exclude_channels})...")
    
    # 1. Load Data
    try:
        df = pd.read_csv(filename, sep='\t', header=None)
        raw_data = df.iloc[:, 1:9].values
    except Exception as e:
        print(f"Error reading file: {e}")
        return

    # 2. Filter Data (High-pass + Notch)
    # Remove DC offset and 60Hz noise
    nyquist = fs / 2
    b_hp, a_hp = butter(2, 1/nyquist, btype='highpass')
    b_notch, a_notch = iirnotch(60/nyquist, Q=30)
    
    filtered_data = np.zeros_like(raw_data)
    for i in range(8):
        sig = filtfilt(b_hp, a_hp, raw_data[:, i])
        sig = filtfilt(b_notch, a_notch, sig)
        filtered_data[:, i] = sig

    # 3. Compute FFT & Plot
    n_samples = raw_data.shape[0]
    window = np.hanning(n_samples)
    
    plt.figure(figsize=(10, 6))
    
    # Loop through all 8 channels
    for i in range(8):
        channel_num = i + 1
        
        # --- EXCLUSION LOGIC ---
        if channel_num in exclude_channels:
            continue
            
        # FFT Math
        signal = filtered_data[:, i] * window
        fft_vals = np.fft.rfft(signal)
        freqs = np.fft.rfftfreq(n_samples, d=1.0/fs)
        
        # Calculate Magnitude (Linear Scale)
        magnitude = np.abs(fft_vals) / n_samples
        
        # Standard Plot (Not Semilog)
        plt.plot(freqs, magnitude, label=f'Ch {channel_num}', alpha=0.8)

    plt.title(f'Linear FFT - {filename}')
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Magnitude (uV)')
    
    # Focus on standard EEG range
    plt.xlim(1, 40) 
    
    plt.grid(True, which='both', linestyle='--', alpha=0.5)
    plt.legend(ncol=2)
    plt.tight_layout()
    plt.show()

# --- MAIN EXECUTION ---
if __name__ == "__main__":
    '''
    Channel 1: Fz
    Channel 2: F1
    Channel 3: F2
    Looking for Theta Waves: 4-8Hz, Averaging these three channels for a "Focus Score"

    Channel 4: Pz
    Channel 5: P1
    Channel 6: P2
    Looking for Alpha Waves: 8-12Hz, alpha spikes when not focused, dropping when engaged
    

    Channel 7: Cz
    Looking for high theta: 4-8Hz
    - Moderate SMR: Calm and Flow
    - High Beta: Stressed/Anxious

    Channel 8: O2
    Looking for MAX ALPHA; set P, F, and C filters to max at this value
    '''
    BAD_CHANNELS = [2, 3, 4, 5, 6, 7, 8] 

    for f in glob.glob("data\\raw\\*.csv"):
        plot_linear_fft(f, exclude_channels=BAD_CHANNELS)
