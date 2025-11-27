# Schnitty's Noise Manager

An open-source audio processing toolkit for converting audio files, removing noise with AI models, and adding synthetic/adversarial noise for protection and testing purposes.

**Developed by Max Hieke**

## Features

### Core Capabilities
- **AI-Powered Noise Removal**: Choose between traditional Spectral Gating or advanced AI models
  - DeepFilterNet
  - SpeechBrain
  - Demucs
  - Asteroid
- **Adversarial Noise Addition**: Multiple noise types for audio protection and robustness testing
- **Batch Processing**: Process multiple audio files simultaneously
- **Multi-format Support**: Input/Output for WAV, MP3, FLAC, AIFF, OGG, M4A
- **Advanced Audio Analysis**: LUFS measurement, spectral analysis, statistics export

### Noise Addition Types
1. **FGSM (Fast Gradient Sign Method)** - Single-step adversarial attack
2. **PGD (Projected Gradient Descent)** - Multi-step iterative attack
3. **Spatial Distortion** - Pitch and time shifting
4. **Adversarial Patch** - Localized noise injection
5. **Universal Perturbation** - Consistent noise pattern
6. **Physical Noise** - Reverb and environmental simulation
7. **Semantic Attack** - Gain and speed modifications

### Post-Processing Effects
- High-pass and Low-pass filtering
- Pre-emphasis and De-emphasis
- Parametric EQ
- Dynamic range compression
- Limiter
- Normalization
- Bass mono conversion

### Protection Presets
15 pre-configured protection profiles including:
- Music Protection I-XV (various adversarial techniques)
- Tape Wow/Flutter simulation (vintage & light)
- Psychoacoustic protection
- Anti-fingerprint modes
- Physical world defenses

## Installation

### Prerequisites
```bash
Python 3.8+
```

### Required Dependencies
```bash
pip install numpy scipy librosa soundfile pydub pillow pyloudnorm matplotlib
```

### Core Libraries
- **numpy** - Numerical computing and array operations
- **scipy** - Signal processing (butter, sosfilt filters)
- **librosa** - Audio analysis and manipulation
- **soundfile** - Audio file I/O
- **pydub** - Audio format conversion
- **pillow** - Image processing (for GUI splash screen)
- **pyloudnorm** - LUFS loudness measurement
- **matplotlib** - Plotting and visualization

### GUI Dependencies
- **tkinter** - GUI framework (usually included with Python)
- **ttk** - Themed Tk widgets (part of tkinter)

**Note**: tkinter is typically pre-installed with Python on Windows and macOS. On Linux, you may need to install it separately:

```bash
# Ubuntu/Debian
sudo apt-get install python3-tk

# Fedora
sudo dnf install python3-tkinter

# Arch Linux
sudo pacman -S tk
```

### AI Model Dependencies (Optional)
If you want to use AI-based denoising models, install these separately:

```bash
# DeepFilterNet
pip install deepfilternet

# SpeechBrain
pip install speechbrain

# Demucs
pip install demucs

# Asteroid
pip install asteroid-filterbanks torch-audiomentations
```

### Complete Installation
For a complete setup with all features:

```bash
# Core dependencies
pip install numpy scipy librosa soundfile pydub pillow pyloudnorm matplotlib

# AI models (optional)
pip install deepfilternet speechbrain demucs asteroid-filterbanks torch-audiomentations
```

Or use the provided requirements.txt:

```bash
pip install -r requirements.txt
```

Generate .exe-File

```bash
python -m PyInstaller --noconfirm --onefile --windowed --icon=icon.ico noise_manager.py
```

## Usaga
### GUI Mode
```bash
python audio_processor.py
```

The GUI provides:
- File browser for input selection
- Output folder configuration
- Real-time parameter adjustment
- Visual progress tracking
- Statistics export (CSV/JSON)
- Spectrogram generation

### Configuration Options

#### Input/Output
- **Input Files**: Drag & drop or browse for audio files
- **Output Folder**: Destination for processed files
- **Output Format**: WAV, MP3, FLAC, OGG, M4A (multiple formats via `&` separator)
- **Bitrate**: 128-320 kbps (for lossy formats)
- **Bit Depth**: 16, 24, 32 bit (for WAV/FLAC)

#### Noise Removal
- **Enable**: Toggle noise reduction
- **Strength**: 0-1 (removal intensity)
- **Filter Order**: 0-6 (higher = sharper cutoff)
- **Min/Max Frequency**: Frequency range for filtering
- **Gate Width**: Smoothing window (1-100)
- **Percentile**: Noise threshold (1-100)
- **Epsilon**: Sensitivity factor (0-1)

#### Noise Addition
- **Preset Selection**: Choose from 15+ protection profiles
- **Noise Type**: Select algorithm (FGSM, PGD, Spatial, etc.)
- **Strength**: Overall noise intensity (0-1)
- **Adv. Epsilon**: Maximum perturbation (0-1)
- **Iterations**: PGD optimization steps (1-20)
- **Filter Parameters**: Band-pass filtering for added noise
  - Order (0-6)
  - Min/Max Frequency (Hz)

#### Post-Processing
- **Highpass/Lowpass Filters**: Remove unwanted frequencies
- **EQ**: Boost/cut specific frequency bands
- **Compression**: Dynamic range control
- **Limiter**: Prevent clipping
- **Normalize**: Target peak level adjustment

## Technical Details

### Noise Removal Algorithm
Uses adaptive spectral gating with configurable Butterworth filtering:
- Spectral analysis via STFT
- Noise profile estimation via percentile method
- Time-frequency masking
- Optional band-pass filtering

### Noise Addition Methods
- **FGSM (additive_adversarial)
- **PGD (pgd)**
- **Spatial (spatial)**

### Audio Statistics
- **LUFS**: Integrated loudness (EBU R128)
- **Peak Level**: Maximum sample value
- **RMS**: Root mean square energy
- **Dynamic Range**: Difference between peaks and average

## Export Options

### Processed Audio
- Main output file(s) in selected format(s)
- Optional: Removed noise signal export
- Optional: Added noise signal export

### Statistics
- **CSV**: Tabular data for analysis
- **JSON**: Structured format for parsing
- **Plots**: Spectrogram visualizations (PNG)

### Statistics Include:
- File metadata (size, format, duration)
- Sample rate and bit depth
- LUFS measurements (before/after)
- Peak levels and RMS
- Processing time
- Applied settings

## Use Cases

### Audio File Protection
Protect your audio from unauthorized AI training, reproduction, or fingerprinting:
```
1. Select "Music Protection III" or similar preset
2. Adjust strength to balance protection vs. quality
3. Enable band-pass filtering for targeted frequency protection
4. Process and export
```

### Audio Restoration & Denoising
Remove background noise using AI models or traditional methods:
```
1. Choose denoising model (DeepFilterNet, SpeechBrain, etc.)
2. Enable "Remove Noise" with appropriate parameters
3. Adjust strength conservatively to preserve quality
4. Export cleaned audio and noise profile
```

### Robustness Testing
Test audio ML models against adversarial perturbations:
```
1. Select adversarial noise type (FGSM/PGD)
2. Configure epsilon and iterations
3. Export both clean and adversarial versions
4. Evaluate model performance degradation
```

### Data Augmentation
Create training data variations for ML models:
```
1. Use spatial or physical noise types
2. Apply subtle transformations
3. Batch process entire datasets
4. Export with consistent parameters
```

### Creative Sound Design
Experiment with audio textures and effects:
```
1. Combine multiple noise types
2. Use post-processing effects
3. Export intermediate stages
4. Create unique sonic signatures
```

## Performance

- **Batch Processing**: Multi-threaded for CPU efficiency
- **Memory Management**: Streaming for large files
- **Speed**: ~5-15x real-time (depends on settings)

## Limitations

- No real-time processing (offline only)
- GUI single-instance (no parallel windows)
- Filter order max: 6 (stability constraint)
- Adversarial effectiveness varies by use case

## Troubleshooting

**"Bandpass filter failed"**
```bash
pip install scipy
```

**"No module named 'librosa'"**
```bash
pip install librosa
```

**GUI not starting**
- Ensure tkinter is installed
- Check Python version (3.8+)
- Verify display environment (for Linux)

**Clipping in output**
- Reduce noise addition strength
- Enable limiter in post-processing
- Lower normalization target

## Contributing

Contributions welcome! Areas of interest:
- Additional noise algorithms
- Real-time processing support
- VST plugin integration
- Advanced ML-based methods

## License

This software is provided **"as-is"** for educational, research, and creative purposes only.

### Terms of Use
- ✅ Educational use
- ✅ Research purposes
- ✅ Creative sound design
- ✅ Personal audio processing
- ❌ Malicious attacks on third-party systems
- ❌ Bypassing security measures
- ❌ Unauthorized data manipulation

### Third-Party Components
All third-party models and libraries are used under their respective open-source licenses:
- **DeepFilterNet**: [License](https://github.com/Rikorose/DeepFilterNet)
- **SpeechBrain**: [Apache 2.0](https://speechbrain.github.io/)
- **Demucs**: [MIT License](https://github.com/facebookresearch/demucs)
- **Asteroid**: [MIT License](https://github.com/asteroid-team/asteroid)

Please respect these licenses when using, distributing, or modifying this software.

### Disclaimer
The developer assumes **no responsibility** for:
- Misuse or illegal activities
- Damage caused by use of this tool
- Violations of third-party terms of service
- Copyright infringement

By using this software, you agree to use it responsibly and ethically.

## Citation

If you use this tool in academic research, please cite:
```bibtex
@software{schnitty_noise_manager,
  author = {Hieke, Max},
  title = {Schnitty's Noise Manager: Audio Processing Toolkit},
  year = {2024},
  url = {https://github.com/[your-username]/schnitty-noise-manager}
}
```

## Acknowledgments

Built with:
- [librosa](https://librosa.org/) - Audio analysis and manipulation
- [NumPy](https://numpy.org/) - Numerical computing
- [SciPy](https://scipy.org/) - Signal processing
- [pyloudnorm](https://github.com/csteinmetz1/pyloudnorm) - LUFS measurement

AI Denoising Models:
- [DeepFilterNet](https://github.com/Rikorose/DeepFilterNet) - Deep learning noise suppression
- [SpeechBrain](https://speechbrain.github.io/) - Speech enhancement toolkit
- [Demucs](https://github.com/facebookresearch/demucs) - Music source separation
- [Asteroid](https://github.com/asteroid-team/asteroid) - Audio source separation toolkit

Adversarial techniques based on published research in audio adversarial examples and robustness testing.

## Support

If you find this tool useful and want to support future development:

**PayPal**: madguineapig@googlemail.com

Your support helps maintain and improve this open-source project!

## Contact

**Developer**: Max Hieke

For bug reports, feature requests, or contributions, please open an issue on GitHub.

---

**Schnitty's Noise Manager** - Professional audio processing for protection, restoration, and research.

**Note**: This tool is intended for legitimate audio processing, protection, research, and creative purposes. Users are responsible for ensuring compliance with applicable laws and regulations. The developer assumes no liability for misuse.
