import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import welch, iirnotch, butter, filtfilt
import glob
import os

def calculate_band_power(psd, freqs, low, high):
    """Calculates average power in a specific frequency band."""
    idx = np.logical_and(freqs >= low, freqs <= high)
    return np.mean(psd[idx])

def graph_indices(filename, fs=250):
    print(f"Processing Indices for: {filename}...")
    
    # 1. LOAD DATA
    try:
        # Load tab-separated, no header
        df = pd.read_csv(filename, sep='\t', header=None)
        # Extract Cols 1-8 (indices 1-9 exclusive)
        raw_data = df.iloc[:, 1:9].values
    except Exception as e:
        print(f"Error reading {filename}: {e}")
        return

    # 2. FILTER DATA (High-pass 1Hz + Notch 60Hz)
    nyquist = fs / 2
    b_hp, a_hp = butter(2, 1/nyquist, btype='highpass')
    b_notch, a_notch = iirnotch(60/nyquist, Q=30)
    
    filtered_data = np.zeros_like(raw_data)
    for i in range(8):
        sig = filtfilt(b_hp, a_hp, raw_data[:, i])
        sig = filtfilt(b_notch, a_notch, sig)
        filtered_data[:, i] = sig

    # 3. SLIDING WINDOW ANALYSIS
    # We calculate metrics every 1 second (overlapping windows)
    window_size = 2 * fs  # 2-second window for stable PSD
    step_size = 1 * fs    # 1-second step (update rate)
    
    n_samples = filtered_data.shape[0]
    
    # Storage Lists
    time_points = []
    workload_indices = []
    engagement_indices = []
    theta_powers = []
    alpha_powers = []

    # Channel Map (0-based indices)
    # Frontal: Fz(0), F1(1), F2(2)
    # Parietal: Pz(3), P1(4), P2(5)
    frontal_chs = [0, 1, 2]
    parietal_chs = [3, 4, 5]

    for start in range(0, n_samples - window_size, step_size):
        end = start + window_size
        window_data = filtered_data[start:end, :]
        
        # Calculate PSD for this window
        freqs, psd = welch(window_data, fs=fs, nperseg=window_size, axis=0)
        
        # --- CALCULATE BAND POWERS ---
        # Frontal Theta (4-8 Hz)
        f_theta = np.mean([calculate_band_power(psd[:, ch], freqs, 4, 8) for ch in frontal_chs])
        
        # Parietal Alpha (8-12 Hz)
        p_alpha = np.mean([calculate_band_power(psd[:, ch], freqs, 8, 12) for ch in parietal_chs])
        
        # Frontal Beta (12-30 Hz) - For Engagement Index
        f_beta = np.mean([calculate_band_power(psd[:, ch], freqs, 12, 30) for ch in frontal_chs])

        # --- CALCULATE INDICES ---
        # 1. Workload Index = Theta / Alpha
        workload = f_theta / p_alpha if p_alpha > 0 else 0
        
        # 2. Engagement Index = Beta / (Theta + Alpha)
        engagement = f_beta / (f_theta + p_alpha) if (f_theta + p_alpha) > 0 else 0
        
        time_points.append(end / fs) # Timestamp in seconds
        workload_indices.append(workload)
        engagement_indices.append(engagement)
        theta_powers.append(f_theta)
        alpha_powers.append(p_alpha)

    # 4. PLOTTING
    plt.figure(figsize=(12, 10))
    
    # Plot 1: The Indices (The "Score")
    plt.subplot(3, 1, 1)
    plt.plot(time_points, workload_indices, label='Workload Index (Theta/Alpha)', color='purple', linewidth=2)
    plt.plot(time_points, engagement_indices, label='Engagement Index (Beta / Theta+Alpha)', color='orange', alpha=0.7)
    plt.title(f'Cognitive Indices Over Time - {os.path.basename(filename)}')
    plt.ylabel('Index Value')
    plt.legend(loc='upper right')
    plt.grid(True, alpha=0.3)
    
    # Plot 2: Raw Band Powers (Why did the score change?)
    plt.subplot(3, 1, 2)
    plt.plot(time_points, theta_powers, label='Frontal Theta (Focus)', color='red')
    plt.plot(time_points, alpha_powers, label='Parietal Alpha (Relax)', color='blue')
    plt.title('Underlying Band Powers')
    plt.ylabel('Power (uV^2/Hz)')
    plt.legend(loc='upper right')
    plt.grid(True, alpha=0.3)
    
    # Plot 3: Flow Components (Theta vs Alpha)
    # In Flow, you want HIGH Theta (Red) and MODERATE Alpha (Blue)
    plt.subplot(3, 1, 3)
    plt.plot(time_points, theta_powers, label='Focus (Theta)', color='red')
    plt.plot(time_points, alpha_powers, label='Calm (Alpha)', color='green', linestyle='--')
    plt.fill_between(time_points, theta_powers, alpha_powers, where=(np.array(theta_powers) > np.array(alpha_powers)), color='yellow', alpha=0.3, label='High Focus Zone')
    plt.title('Flow Check (Yellow = Focus > Relax)')
    plt.xlabel('Time (Seconds)')
    plt.ylabel('Power')
    plt.legend(loc='upper right')
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()

# --- RUN ON FILES ---
if __name__ == "__main__":
    files = glob.glob("data\\raw\\*.csv")
    for f in files:
        graph_indices(f)
