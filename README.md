# NeuroFlow

**An experimental EEG signal-processing and focus-neurofeedback prototype**

NeuroFlow explores how EEG recordings can be filtered, screened for artifacts,
and summarized as frequency-domain features. Its longer-term goal is to use
personalized EEG baselines to identify focus states and support timed focus
sessions.

This repository is a research prototype, not a validated focus detector or
medical device. It currently contains offline analysis scripts, two sample
recordings, and an unfinished real-time monitoring path.

## Project snapshot

- **Input:** Two tab-separated EEG recordings, each containing eight EEG
  channels and additional device metadata.
- **Offline analysis:** Band-pass filtering, heuristic epoch rejection, Welch
  power spectral density (PSD) estimation, and per-channel PSD plots.
- **Exploratory signal analysis:** FFT plots, frequency-band highlights, and a
  frontal-theta to parietal-alpha workload index.
- **Real-time direction:** A BrainFlow-based script sketches calibration and
  focus-block state logic for an OpenBCI Cyton board.
- **Current evaluation:** The offline preprocessing pipeline ran on both
  bundled recordings. There is no labeled evaluation set, trained classifier,
  or validated focus-accuracy result.

## How the offline pipeline works

The main offline analysis function is
[`process_eeg_pipeline`](src/processing/preprocessing.py). For each recording,
it:

1. Reads a tab-separated, headerless file and selects columns 2–9 as the eight
   EEG channels.
2. Assumes a 250 Hz sampling rate and applies a fourth-order 0.5–30 Hz
   Butterworth band-pass filter. It also filters 30–50 Hz data for its
   artifact heuristic.
3. Divides the recording into non-overlapping one-second epochs.
4. Rejects an epoch if its filtered amplitude exceeds the configured threshold
   of 100, or if both the gamma- and theta-band power heuristics exceed their
   configured thresholds.
5. Estimates a Welch PSD for the accepted epochs, averages the PSD across
   epochs, and plots the resulting spectrum for each channel.

The function returns the frequency bins and mean PSD array; the current script
does not save plots or analysis results to disk by default.

Other processing modules explore linear FFTs, band-power indices, and
time-series plots with highlighted focus zones. These are exploratory
utilities, not a single integrated analysis workflow.

## Results on the bundled recordings

The following counts were produced by running the existing offline
preprocessing function with `fs=250` on each file. Duration and observed rate
are calculated from the recordings' timestamp column. Acceptance percentages
are accepted epochs divided by total epochs.

| Recording | Samples | Duration | Observed rate | Epochs | Rejected | Accepted | Acceptance |
|---|---:|---:|---:|---:|---:|---:|---:|
| `eeg_test1.csv` | 47,695 | 190.467 s | 250.406 Hz | 190 | 5 | 185 | 97.4% |
| `eeg_test2.csv` | 40,526 | 161.842 s | 250.399 Hz | 162 | 7 | 155 | 95.7% |
| **Combined** | **88,221** | **352.309 s** | — | **352** | **12** | **340** | **96.6%** |

These are **pipeline throughput and artifact-screening results only**. They do
not measure focus-detection accuracy, and the rejection thresholds have not
been validated against expert annotations or a labeled benchmark. The
recordings are processed at the script's assumed 250 Hz even though their
timestamps imply approximately 250.4 Hz.

## Repository layout

```text
data/
  raw/                  Bundled EEG recordings (tab-separated CSV files)
  clean/                Reserved for processed data; currently empty
src/
  main.py               Experimental real-time focus-monitoring script
  processing/
    FFT.py              Linear FFT visualization
    artifacts.py        Artifact-processing work in progress
    band_passes.py      Band-pass filtering experiment
    feature_extraction.py
                        PSD feature-extraction experiment
    graph_indices.py    Spectral and workload-index visualizations
    preprocessing.py    Offline filtering, epoch screening, and PSD pipeline
txts/
  algorithm.txt         High-level EEG analysis pipeline outline
  articles.txt          Background literature link
  things_to_build.txt   Early implementation notes
requirements.txt        Python package list for the analysis work
```

## Getting started

### Environment setup

Use Python 3.9 for the most reproducible setup; other Python versions have not
been certified. Install the listed dependencies from the repository root:

```bash
python3 -m venv venv
source venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

On Windows, activate the environment with:

```powershell
venv\Scripts\Activate.ps1
```

### Run the offline analysis

There is not yet a command-line entry point. From the repository root, start
Python and call the existing pipeline for each recording:

```python
from pathlib import Path
from src.processing.preprocessing import process_eeg_pipeline

for recording in sorted(Path("data/raw").glob("*.csv")):
    process_eeg_pipeline(str(recording), fs=250)
```

The pipeline prints epoch counts and opens a PSD plot for each input file.
Close each plot to continue to the next recording.

## Real-time monitoring status

[`src/main.py`](src/main.py) sketches a live focus-monitoring loop: it reads
data from a BrainFlow Cyton board, calibrates relaxed and focus baselines, and
uses a frontal-theta/parietal-alpha ratio to drive focus-block state changes.
The source configures the serial port as `COM5` and includes a five-minute
focus-block duration.

This path is **not currently ready to run end-to-end**: BrainFlow is not
included in `requirements.txt`, and the called `send_discord_alert` function
is not implemented. Hardware setup, alert delivery, and calibration behavior
have not been verified as part of the results above.

## Known gaps

- The repository has no automated test suite or reproducible command-line
  interface.
- The analysis assumes a particular tab-separated device export layout,
  channel order, and sampling rate; these assumptions are not checked by a
  shared input-validation layer.
- Processing thresholds are hand-tuned heuristics. The recordings have no
  focus labels, so the project cannot yet report classification metrics.
- The processing experiments are not integrated: for example,
  `band_passes.py` references example variables that are not defined in the
  module, and `feature_extraction.py` depends on `raws` and `epochs` objects
  that are not currently provided by the referenced modules.
- Some file-discovery code uses Windows-style path strings, which is not
  portable across operating systems.
- The real-time script has unresolved runtime dependencies and alert
  integration, as described above.

## What would make the project stronger

1. **Make one offline workflow reproducible.** Add a portable CLI, validate the
   input schema and sampling rate, move parameters into explicit
   configuration, and save plots plus machine-readable summaries to an
   ignored `results/` directory.
2. **Add tests and continuous integration.** Cover file parsing, filtering,
   epoch counts, artifact rejection, and invalid input; run the checks on
   supported Python versions.
3. **Establish scientific validity.** Document channel names and units, record
   acquisition details and consent/provenance, label focus/rest intervals,
   compare against simple baselines, and report subject/session-separated
   precision, recall, F1, and uncertainty. Avoid claims of focus detection
   until these evaluations support them.
4. **Finish or clearly scope the real-time feature.** Declare optional
   BrainFlow dependencies, remove the hard-coded serial port, implement
   configurable alert delivery, and test the state machine independently of
   physical hardware.
5. **Improve the portfolio presentation.** Include example plots, a concise
   data dictionary, a short architecture diagram, setup screenshots or a
   demo, and a clear statement of each contributor's role and the project's
   license.

## Background

The project notes outline an EEG acquisition-to-classification pipeline in
[`txts/algorithm.txt`](txts/algorithm.txt). A literature link collected during
the project is listed in [`txts/articles.txt`](txts/articles.txt):
[EEG signal processing article (PubMed Central)](https://pmc.ncbi.nlm.nih.gov/articles/PMC10385593/#abstract1).

## Disclaimer

NeuroFlow is an educational and experimental project. Its signal-derived
indices are not a diagnosis, clinical assessment, or substitute for
professional medical advice.
