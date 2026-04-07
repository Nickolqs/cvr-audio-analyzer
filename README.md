# CVR Audio Analyzer

Python desktop application with a simple Tkinter GUI for analyzing CVR-like audio files.

Software was developed as part of the research project “SOFTWARE TOOL FOR TRAINING IN THE AUTOMATED PROCESSING OF FLIGHT DATA RECORDERS”.

## Features
- open audio files through GUI
- choose output folder
- analyze CVR-like audio
- detect events based on energy, pitch, and spectral features
- build plots
- export results to CSV, JSON, and PNG

## Supported formats
- WAV
- MP3
- M4A
- DAT
- BIN
- TRS

## Notes
- For MP3, M4A, and some DAT/BIN/TRS files, ffmpeg may be required in PATH.
- For DAT/BIN/TRS files, RAW mode may be needed.
- If a .trs file is actually a text/XML transcript, the script reports it.

## Run
```bash
python cvr_analyzer.py
