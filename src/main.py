import time
import requests
import numpy as np
from brainflow.board_shim import BoardShim, BrainFlowInputParams, BoardIds
from brainflow.data_filter import DataFilter, FilterTypes, DetrendOperations

# CONFIG
WINDOW_SECONDS = 2.0
CALIBRATION_DURATION = 10
DIFFICULTY = 0.75
FOCUS_DURATION_SECONDS = 5 * 60  # 5 minutes

# Paste your Discord Webhook URL here
DISCORD_WEBHOOK_URL = "https://discord.com/api/webhooks/1476432420612870274/Q6AlsJt2ct02tVHEJzLfR2K2yyTx9Xv8U0HbnBaRtXkK5UD38X0IWMlaFlCp8gEM5Kq0"

# Example API/webhook function
def send_discord_alert(message):
    # Sends a message to a Discord channel via Webhook.
    if DISCORD_WEBHOOK_URL == "YOUR_DISCORD_WEBHOOK_URL_HERE":
        print("[API] Error: Please set your DISCORD_WEBHOOK_URL in the configuration.")
        return

    payload = {
        "content": message,
        "username": "BrainFlow Bot",
        "avatar_url": "https://cdn-icons-png.flaticon.com/512/2099/2099058.png"
    }
    
    try:
        response = requests.post(DISCORD_WEBHOOK_URL, json=payload)
        response.raise_for_status()
        print(f"\n[DISCORD] Sent: {message}")
    except Exception as e:
        print(f"\n[DISCORD ERROR] Failed to send message: {e}")

def get_focus_metric(data, board_id, fs):
    # Helper function to process data and return the Workload Index.
    eeg_channels = BoardShim.get_eeg_channels(board_id)
    frontal_channels = eeg_channels[0:3]
    parietal_channels = eeg_channels[3:6]

    scale_factor = (4.5 / 24 / (2**23 - 1)) * 1000000.0
    for ch in eeg_channels:
        data[ch] = data[ch] * scale_factor

    for ch in eeg_channels:
        DataFilter.detrend(data[ch], DetrendOperations.CONSTANT.value)
        DataFilter.perform_bandstop(data[ch], fs, 59.0, 61.0, 4, FilterTypes.BUTTERWORTH.value, 0)
        DataFilter.perform_bandpass(data[ch], fs, 1.0, 30.0, 4, FilterTypes.BUTTERWORTH.value, 0)

    if np.max(np.abs(data[eeg_channels])) > 200:
        return None

    f_bands = DataFilter.get_avg_band_powers(data, frontal_channels, fs, False)
    avg_theta = f_bands[0][1]

    p_bands = DataFilter.get_avg_band_powers(data, parietal_channels, fs, False)
    avg_alpha = p_bands[0][2]

    safe_alpha = avg_alpha if avg_alpha > 0.1 else 0.1
    return avg_theta / safe_alpha

def run_calibration(board, board_id, fs, n_samples):
    # Runs calibration and applies the DIFFICULTY weight.
    print("\n" + "="*40)
    print(f"   CALIBRATION (Difficulty: {DIFFICULTY*100}%)")
    print("="*40)
    
    print(f"\nPhase 1: RELAX ({CALIBRATION_DURATION}s)")
    print(">> Close eyes, relax muscles...")
    time.sleep(3) 
    
    relax_ratios = []
    start_time = time.time()
    
    while time.time() - start_time < CALIBRATION_DURATION:
        time.sleep(0.5)
        data = board.get_current_board_data(n_samples)
        if data.shape[1] < n_samples: continue
        
        val = get_focus_metric(data, board_id, fs)
        if val is not None:
            relax_ratios.append(val)
            print(f"\rSamples: {len(relax_ratios)} | Curr: {val:.2f}", end="")
    
    avg_relax = np.percentile(relax_ratios, 40) if relax_ratios else 1.0
    print(f"\n>> Baseline Relax: {avg_relax:.2f}")

    print(f"\nPhase 2: FOCUS ({CALIBRATION_DURATION}s)")
    print(">> Open eyes. Do mental math hard!...")
    input(">> Press ENTER to start Focus calibration...")
    
    focus_ratios = []
    start_time = time.time()
    
    while time.time() - start_time < CALIBRATION_DURATION:
        time.sleep(0.5)
        data = board.get_current_board_data(n_samples)
        if data.shape[1] < n_samples: continue
        
        val = get_focus_metric(data, board_id, fs)
        if val is not None:
            focus_ratios.append(val)
            print(f"\rSamples: {len(focus_ratios)} | Curr: {val:.2f}", end="")
            
    avg_focus = np.percentile(focus_ratios, 60) if focus_ratios else 2.0
    print(f"\n>> Baseline Focus: {avg_focus:.2f}")
    
    focus_range = avg_focus - avg_relax
    
    if focus_range <= 0:
        print("\nWARNING: Focus was not higher than Relax. Defaulting to Relax + 30%")
        threshold = avg_relax * 1.3
    else:
        threshold = avg_relax + (focus_range * DIFFICULTY)

    print("\n" + "-"*40)
    print(f"CALIBRATION COMPLETE")
    print(f"Range: {avg_relax:.2f} -> {avg_focus:.2f}")
    print(f"Computed Threshold: {threshold:.2f}")
    print("-"*40 + "\n")
    return threshold

def real_time_focus_monitor():
    params = BrainFlowInputParams()
    board_id = BoardIds.CYTON_BOARD.value 
    params.serial_port = 'COM5' 
    
    board = BoardShim(board_id, params)
    
    # STATE TRACKING VARIABLES
    focus_timestamps = []
    session_active = False
    session_end_time = 0
    
    try:
        board.prepare_session()
        board.start_stream()
        
        fs = BoardShim.get_sampling_rate(board_id)
        n_samples = int(WINDOW_SECONDS * fs)
        
        time.sleep(2) 
        threshold = run_calibration(board, board_id, fs, n_samples)
        
        print("--- MONITOR RUNNING ---")
        while True:
            time.sleep(0.2) 
            current_time = time.time()
            data = board.get_current_board_data(n_samples)
            
            if data.shape[1] < n_samples:
                continue

            ratio = get_focus_metric(data, board_id, fs)
            if ratio is None:
                continue
            
            is_focused = ratio > threshold
            status_text = "!!! FOCUS !!!" if is_focused else "RELAXED"
            
            # --- FOCUS STATE LOGIC ---
            if is_focused:
                focus_timestamps.append(current_time)
                
            # Keep only timestamps from the last 2 seconds
            focus_timestamps = [t for t in focus_timestamps if current_time - t <= 2.0]
            
            # Check if condition is met (> 5 states in 2 seconds)
            if len(focus_timestamps) > 5:
                if not session_active:
                    send_discord_alert("**Deep Focus Detected!** Starting a 5-minute focus block.")
                    session_active = True
                
                # Set or extend the 5-minute timer
                session_end_time = current_time + FOCUS_DURATION_SECONDS
                focus_timestamps.clear() 
            
            # TIMER CHECK LOGIC ---
            # If a session is active and the timer has expired, toggle off
            if session_active and current_time > session_end_time:
                send_discord_alert("**Focus block complete.** Great work! Standing by for next focus state.")
                session_active = False

            # Simple visualizer
            session_status = "[SESSION ACTIVE]" if session_active else "[WAITING]"
            print(f"Metric: {ratio:.2f} | Thresh: {threshold:.2f} | {status_text} | Hits/2s: {len(focus_timestamps)} {session_status}")

    except KeyboardInterrupt:
        print("\nStopping...")
    finally:
        if board.is_prepared():
            board.stop_stream()
            board.release_session()
            
        if session_active:
            send_discord_alert("**Script stopped.** Focus session ended early.")

if __name__ == "__main__":
    real_time_focus_monitor()
