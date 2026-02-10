import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt, welch
import glob
import os

def process_eeg_pipeline(filename, fs=250):
    print(f"Processing Pipeline for: {filename}...")

    try:
        # Load tab-separated, no header (standard OpenBCI format)
        df = pd.read_csv(filename, sep='\t', header=None)
        # Columns 1-8 are EEG data (indices 1 to 9 exclusive)
        raw_data = df.iloc[:, 1:9].values
    except Exception as e:
        print(f"Error reading file: {e}")
        return

    n_samples, n_channels = raw_data.shape
    
    # 2. PRE-PROCESSING
    nyquist = fs / 2
    
    # Filter 1 (0.5 - 30 Hz)
    # REmoves DC Offset (<0.5Hz) and Mains Hum (>60Hz)
    b_analysis, a_analysis = butter(4, [0.5 / nyquist, 30.0 / nyquist], btype='bandpass')
    
    # Gamma Filter (30 - 50 Hz) - Artifact Detection 
    b_gamma, a_gamma = butter(4, [30.0 / nyquist, 50.0 / nyquist], btype='bandpass')

    # Apply Filters
    clean_data = np.zeros_like(raw_data)
    gamma_data = np.zeros_like(raw_data)

    for i in range(n_channels):
        clean_data[:, i] = filtfilt(b_analysis, a_analysis, raw_data[:, i])
        gamma_data[:, i] = filtfilt(b_gamma, a_gamma, raw_data[:, i])

    # 3. EPOCHING & ARTIFACT REJECTION
    window_size = 1 * fs  # 1-second window
    step_size = 1 * fs    # Non-overlapping for standard analysis (or overlapping if desired)
    
    accepted_epochs = []
    rejected_count = 0
    total_epochs = 0
    
    # Thresholds (Tunable based on user baseline)
    AMPLITUDE_THRESH = 100.0  # uV (Blinks usually > 100uV)
    GAMMA_POWER_THRESH = 15.0 # uV^2 (Indicative of muscle tension)
    THETA_POWER_THRESH = 20.0 # uV^2 (High Theta)

    for start in range(0, n_samples - window_size, step_size):
        total_epochs += 1
        end = start + window_size
        
        # Get the epoch data
        epoch_clean = clean_data[start:end, :]
        epoch_gamma = gamma_data[start:end, :]
        
        # --- REJECTION LOGIC ---
        
        # Rule 1: Amplitude Threshold (Blinks/Jaw Clench)
        if np.max(np.abs(epoch_clean)) > AMPLITUDE_THRESH:
            rejected_count += 1
            continue  # Reject and skip to next epoch

        # Rule 2: Simultaneous Gamma & Theta Spike (Muscle Artifact Check)
        # We calculate power in this specific window to check the spectral content
        freqs, psd_clean = welch(epoch_clean, fs=fs, axis=0)
        freqs_g, psd_gamma = welch(epoch_gamma, fs=fs, axis=0)
        
        # Calculate average power in bands
        theta_power = np.mean(psd_clean[(freqs >= 4) & (freqs <= 8)], axis=0)
        gamma_power = np.mean(psd_gamma[(freqs_g >= 30) & (freqs_g <= 50)], axis=0)
        
        # If BOTH Gamma and Theta are high, it's likely muscle noise mimicking Theta
        if np.any((gamma_power > GAMMA_POWER_THRESH) & (theta_power > THETA_POWER_THRESH)):
             rejected_count += 1
             continue # Reject

        # If it passes both checks, keep it
        accepted_epochs.append(epoch_clean)

    # 4. FEATURE EXTRACTION (PSD on Clean Windows)
    if not accepted_epochs:
        print("No valid epochs found. Data too noisy.")
        return

    # Stack all accepted epochs into a 3D array: (n_epochs, n_samples, n_channels)
    accepted_data = np.stack(accepted_epochs, axis=0)
    
    # Calculate Mean PSD across all accepted epochs (Welch's Method)
    # axis=1 computes PSD along the time axis (n_samples)
    freqs, psd_epochs = welch(accepted_data, fs=fs, nperseg=window_size, axis=1)
    
    # Average across epochs to get the final "Clean Feature" for this file
    mean_psd = np.mean(psd_epochs, axis=0)

    print(f"Total Epochs: {total_epochs} | Rejected: {rejected_count} | Accepted: {len(accepted_epochs)}")
    
    # Plotting the Result
    plt.figure(figsize=(10, 6))
    plt.semilogy(freqs, mean_psd)
    plt.title(f'Clean PSD (Feature Extraction) - {os.path.basename(filename)}')
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Power Spectral Density (uV^2/Hz)')
    plt.xlim(0.5, 30) # Show only the filtered range
    plt.legend([f'Ch{i+1}' for i in range(n_channels)], loc='upper right', ncol=2, fontsize='small')
    plt.grid(True, which='both', linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.show()

    return freqs, mean_psd

# --- MAIN EXECUTION ---
if __name__ == "__main__":
    files = glob.glob("data\\raw\\*.csv")
    for f in files:
        process_eeg_pipeline(f)
