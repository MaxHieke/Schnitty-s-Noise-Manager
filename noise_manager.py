import os
import json
import numpy as np
import soundfile as sf
import matplotlib.pyplot as plt
from pathlib import Path
import time
from datetime import datetime
import librosa
import pyloudnorm as pyln
from scipy import signal
from pydub import AudioSegment
import tkinter as tk
from tkinter import ttk
from PIL import Image, ImageTk
import threading

import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import threading
import shutil
from pathlib import Path
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib.image as mpimg

# WICHTIG: Globale Variablen HIER definieren (vor allen Funktionen)
postprocess_vars = {}
postprocess_entries = {}
pause_flag = False

def apply_postprocess_custom(audio, sr, settings):
	"""
	Flexible post-processing with individual controls
	"""
	from scipy.signal import butter
	audio = np.asarray(audio, dtype=np.float32)
	
	# Stereo-Handling für Bass-Mono
	is_stereo = len(audio.shape) > 1
	original_stereo = None
	
	if is_stereo:
		original_stereo = audio.copy()
		audio = np.mean(audio, axis=1)
	
	# Normalize safe
	audio = audio / np.max(np.abs(audio) + 1e-8)
	y = audio.copy()

	# 1. Bass-Mono (VOR allen anderen Filtern!)
	if settings.get('bass_mono_enabled', False) and is_stereo:
		from scipy.signal import butter, sosfilt
		
		cutoff_freq = settings.get('bass_mono_freq', 300)
		nyquist = sr / 2
		
		sos_low = butter(4, cutoff_freq / nyquist, btype='low', output='sos')
		bass_mono = sosfilt(sos_low, np.mean(original_stereo, axis=1))
		
		sos_high = butter(4, cutoff_freq / nyquist, btype='high', output='sos')
		highs_left = sosfilt(sos_high, original_stereo[:, 0])
		highs_right = sosfilt(sos_high, original_stereo[:, 1])
		
		y_stereo = np.zeros_like(original_stereo)
		y_stereo[:, 0] = bass_mono + highs_left
		y_stereo[:, 1] = bass_mono + highs_right
		
		y = np.mean(y_stereo, axis=1)
		y = y / np.max(np.abs(y) + 1e-8)
	else:
		y_stereo = None

	# 2. High-Pass Filter
	if settings.get('highpass_enabled', False):
		from scipy.signal import butter, sosfilt
		cutoff = settings.get('highpass_freq', 50)
		nyquist = sr / 2
		sos = butter(2, cutoff / nyquist, btype='high', output='sos')
		y = sosfilt(sos, y)

	# 3. Low-Pass Filter
	if settings.get('lowpass_enabled', False):
		from scipy.signal import butter, sosfilt
		cutoff = settings.get('lowpass_freq', 16000)
		nyquist = sr / 2
		sos = butter(2, cutoff / nyquist, btype='low', output='sos')
		y = sosfilt(sos, y)

	# 4. Pre-Emphasis
	if settings.get('preemphasis_enabled', False):
		coef = settings.get('preemphasis_coef', 0.85)
		y = librosa.effects.preemphasis(y, coef=coef)

	# 5. De-Emphasis
	if settings.get('deemphasis_enabled', False):
		coef = settings.get('deemphasis_coef', 0.8)
		y = librosa.effects.deemphasis(y, coef=coef)

	# 6. Parametric EQ
	if settings.get('eq_enabled', False):
		from scipy.signal import butter, sosfilt
		nyquist = sr / 2
		freq = settings.get('eq_freq', 8000) / nyquist
		gain_db = settings.get('eq_gain', 2.0)
		sos = butter(2, freq, btype='high', output='sos')
		high_boost = sosfilt(sos, y) * (10 ** (gain_db / 20) - 1)
		y = y + high_boost

	# 7. Compression
	if settings.get('compression_enabled', False):
		ratio = settings.get('compression_ratio', 1.2)
		y = np.tanh(y * ratio) / np.tanh(ratio)

	# 8. Limiter
	if settings.get('limiter_enabled', False):
		threshold = settings.get('limiter_threshold', 0.95)
		y = np.clip(y, -threshold, threshold)

	# 9. Final Normalization
	if settings.get('normalize_enabled', True):
		target = settings.get('normalize_target', 0.989)
		y = librosa.util.normalize(y) * target
	
	# Falls Bass-Mono aktiv war, Stereo-Version zurückgeben
	if y_stereo is not None:
		y_stereo = y_stereo / np.max(np.abs(y_stereo) + 1e-8)
		
		if settings.get('limiter_enabled', False):
			threshold = settings.get('limiter_threshold', 0.95)
			y_stereo = np.clip(y_stereo, -threshold, threshold)
		
		if settings.get('normalize_enabled', True):
			target = settings.get('normalize_target', 0.989)
			y_stereo = librosa.util.normalize(y_stereo.T).T * target
		
		return y_stereo
	
	return y


class AudioProcessor:
	
	def __init__(self, config, selected_files=None):
		self.config = config
		self.input_folder = Path(config['input_folder'])
		self.output_folder = self.input_folder / 'converted'
		self.plots_folder = self.input_folder / 'plots'
		self.stats = []
		self.selected_files = selected_files or []
		
		self.noise_profile = None
		self.noise_profile_sr = None
		
		if config.get('noise_profile_path'):
			profile_path = Path(config['noise_profile_path'])
			if profile_path.exists():
				self.noise_profile, self.noise_profile_sr = self.load_noise_profile(profile_path)
		
		self.output_folder.mkdir(exist_ok=True)
		self.plots_folder.mkdir(exist_ok=True)
	
	def load_noise_profile(self, noise_file_path):
		"""Lädt eine Noise-Profile Datei"""
		try:
			noise_audio, noise_sr = self.load_audio(noise_file_path)
			
			if len(noise_audio.shape) == 2:
				if noise_audio.shape[1] == 2:
					noise_audio = np.mean(noise_audio, axis=1)
				elif noise_audio.shape[0] == 2:
					noise_audio = np.mean(noise_audio, axis=0)
			
			D_noise = librosa.stft(noise_audio)
			magnitude_noise = np.abs(D_noise)
			noise_profile = np.median(magnitude_noise, axis=1, keepdims=True)
			
			print(f"[INFO] Noise profile loaded: {noise_file_path.name}")
			return noise_profile, noise_sr
		
		except Exception as e:
			print(f"[ERROR] Could not load noise profile: {e}")
			return None, None
	
	def get_audio_files(self):
		"""Findet alle Audio-Dateien"""
		if self.selected_files:
			files = [self.input_folder / filename for filename in self.selected_files]
			files = [f for f in files if f.exists()]
			return files
		
		extensions = ['.wav', '.mp3', '.flac', '.aiff', '.aif']
		files = []
		for ext in extensions:
			files.extend(list(self.input_folder.glob(f'*{ext}')))
			files.extend(list(self.input_folder.glob(f'*{ext.upper()}')))
		files = list(dict.fromkeys(files))
		return files
	
	def load_audio(self, file_path):
		"""Lädt Audio-Datei"""
		try:
			audio, sr = librosa.load(file_path, sr=None, mono=False)
			if len(audio.shape) == 2:
				audio = audio.T
			return audio, sr
		except:
			audio, sr = sf.read(file_path)
			return audio, sr
	
	def remove_noise(self, audio, sr, strength=0.01):
		"""
		Entfernt Rauschen aus einem Audiosignal.
		"""
		import numpy as np
		
		model_name = self.config.get("noise_removal_model", "Spectral Gating").lower().strip()
		enhanced_audio = None

		# === 1. DeepFilterNet ===
		if "deepfilternet" in model_name:
			try:
				import df
				from df.enhance import enhance, init_df
				import soundfile as sf
				import io
				print("[INFO] Using DeepFilterNet for denoising...")
				
				tmp_buf = io.BytesIO()
				sf.write(tmp_buf, audio, sr, format='WAV')
				tmp_buf.seek(0)

				model, df_state, _ = init_df()
				enhanced_audio, _ = enhance(model, tmp_buf, df_state)

				noise = audio - enhanced_audio[:len(audio)]
			except Exception as e:
				print(f"[WARN] DeepFilterNet failed: {e}")
				enhanced_audio = None  # Explizit None setzen

		# === 2. Demucs ===
		elif "demucs" in model_name:
			try:
				import torch
				import numpy as np
				from demucs.pretrained import get_model
				from demucs.apply import apply_model

				print("[INFO] Using Demucs for denoising...")
				device = "cuda" if torch.cuda.is_available() else "cpu"
				model = get_model("htdemucs").to(device).eval()

				if audio.ndim == 1:
					audio_tensor = torch.tensor(audio[None, :], dtype=torch.float32).to(device)
				else:
					audio_tensor = torch.tensor(audio.T[None, :, :], dtype=torch.float32).to(device)

				with torch.no_grad():
					out = apply_model(model, audio_tensor, split=True, overlap=0.25, progress=False)

				enhanced_audio = out.squeeze().cpu().numpy().T
				if enhanced_audio.ndim > 1:
					enhanced_audio = enhanced_audio.mean(axis=1)

				noise = audio - enhanced_audio[:len(audio)]
			except Exception as e:
				print(f"[WARN] Demucs failed: {e}")
				enhanced_audio = None

		# === 3. Spleeter ===
		elif "spleeter" in model_name:
			try:
				from spleeter.separator import Separator
				import numpy as np
				print("[INFO] Using Spleeter for denoising (vocals separation)...")

				separator = Separator("spleeter:2stems")
				out = separator.separate(audio.reshape(-1, 1) if audio.ndim == 1 else audio)

				voice = out["vocals"].mean(axis=1)
				enhanced_audio = voice
				noise = audio - enhanced_audio[:len(audio)]
			except Exception as e:
				print(f"[WARN] Spleeter failed: {e}")
				enhanced_audio = None

		# === 4. SpeechBrain ===
		elif "speechbrain" in model_name:
			try:
				import torch
				import numpy as np
				from speechbrain.pretrained import SpectralMaskEnhancement

				print("[INFO] Using SpeechBrain for denoising...")
				device = "cuda" if torch.cuda.is_available() else "cpu"
				enhancer = SpectralMaskEnhancement.from_hparams(
					source="speechbrain/mtl-mimic-voicebank",
					run_opts={"device": device}
				)

				audio_tensor = torch.tensor(audio, dtype=torch.float32).unsqueeze(0).to(device)
				enhanced_audio = enhancer.enhance_batch(audio_tensor)[0].cpu().numpy()

				noise = audio - enhanced_audio[:len(audio)]
			except Exception as e:
				print(f"[WARN] SpeechBrain failed: {e}")
				enhanced_audio = None

		# === 5. Asteroid ===
		elif "asteroid" in model_name:
			try:
				import torch
				import numpy as np
				from asteroid.models import DPRNNTasNet

				print("[INFO] Using Asteroid DPRNNTasNet for denoising...")
				device = "cuda" if torch.cuda.is_available() else "cpu"

				model = DPRNNTasNet.from_pretrained("JorisCos/DPRNNTasNet-ks2").to(device).eval()
				audio_tensor = torch.tensor(audio, dtype=torch.float32).unsqueeze(0).to(device)

				with torch.no_grad():
					est = model(audio_tensor).cpu().numpy()[0]

				enhanced_audio = est[:len(audio)]
				noise = audio - enhanced_audio
			except Exception as e:
				print(f"[WARN] Asteroid failed: {e}")
				enhanced_audio = None

		# === FALLBACK: Spectral Gating ===
		if enhanced_audio is None:
			print("[INFO] Using Spectral Gating (default)...")
			if len(audio.shape) == 2:
				cleaned = np.zeros_like(audio)
				noise = np.zeros_like(audio)
				for ch in range(audio.shape[1]):
					cleaned[:, ch], noise[:, ch] = self._remove_noise_mono(audio[:, ch], sr, strength)
				enhanced_audio = cleaned
			else:
				enhanced_audio, noise = self._remove_noise_mono(audio, sr, strength)

		return enhanced_audio, noise

	
	def _remove_noise_mono(self, audio, sr, strength):
		"""Rauschentfernung aus Mono-Signal"""
		
		# PRÜFEN: Wenn Noise-Profil vorhanden -> verwenden
		if self.noise_profile is not None:
			if self.noise_profile_sr == sr:
				print("[INFO] Using loaded noise profile for denoising")
				return self._remove_noise_with_profile(audio, sr, self.noise_profile, strength)
			else:
				print(f"[WARN] Noise profile sample rate ({self.noise_profile_sr} Hz) "
					  f"doesn't match audio ({sr} Hz). Using automatic method.")

		from scipy.signal import butter, cheby1, cheby2, ellip, sosfilt

		def bandpass_filter(audio, sr, min_freq, max_freq, order=1, filter_type="butterworth"):
			"""Bandpassfilter mit wählbarem Typ"""
			nyquist = sr / 2
			low = min_freq / nyquist
			high = max_freq / nyquist

			# Map GUI-Auswahl auf Funktion
			filter_type = filter_type.lower()
			if filter_type == "butterworth":
				sos = butter(order, [low, high], btype='band', output='sos')
			elif filter_type == "chebyshev i":
				sos = cheby1(order, 0.5, [low, high], btype='band', output='sos')
			elif filter_type == "chebyshev ii":
				sos = cheby2(order, 20, [low, high], btype='band', output='sos')
			elif filter_type == "elliptic":
				sos = ellip(order, 0.5, 20, [low, high], btype='band', output='sos')
			else:
				raise ValueError(f"Unknown filter type: {filter_type}")

			return sosfilt(sos, audio)

		# Parameter aus config
		min_freq = self.config.get('denoise_min_freq', 0)
		max_freq = self.config.get('denoise_max_freq', sr / 2)
		order = self.config.get("order_remove")
		# filter_type = self.config.get("filter_type_remove")
		
		
		percentile = self.config.get('denoise_percentile', 10)
		gate_width = self.config.get('gate_width', 1.0)
		epsilon = self.config.get('denoise_epsilon', 0.01)

		# STFT
		D = librosa.stft(audio)
		magnitude, phase = np.abs(D), np.angle(D)

		# Noise-Profil auf Basis Bandpass
		audio_for_noise = bandpass_filter(audio, sr, min_freq, max_freq, order)
		D_noise = librosa.stft(audio_for_noise)
		mag_noise = np.abs(D_noise)

		# Schwelle (Percentile) + Epsilon-Feinanpassung
		noise_profile = np.percentile(mag_noise, percentile, axis=1, keepdims=True)
		threshold = noise_profile * (1 + strength * 10) * (1 + epsilon) * gate_width

		# Gate anwenden
		mask = magnitude > threshold
		cleaned_magnitude = magnitude * mask
		noise_magnitude = magnitude * (1 - mask)

		# Rekonstruktion
		cleaned = librosa.istft(cleaned_magnitude * np.exp(1j * phase))
		noise = librosa.istft(noise_magnitude * np.exp(1j * phase))

		# Länge angleichen
		def match_length(arr, length):
			diff = length - len(arr)
			if diff > 0:
				return np.pad(arr, (0, diff))
			elif diff < 0:
				return arr[:length]
			return arr

		cleaned = match_length(cleaned, len(audio))
		noise = match_length(noise, len(audio))

		return cleaned, noise
	
	def add_adaptive_noise(self, audio, sr, strength=0.01, noise_type='additive_adversarial'):
		"""Fügt Rauschen hinzu"""
		if len(audio.shape) == 2:
			if audio.shape[1] <= 2:
				noisy = np.zeros_like(audio)
				noise_signal = np.zeros_like(audio)
				for ch in range(audio.shape[1]):
					noisy[:, ch], noise_signal[:, ch] = self._add_adaptive_noise_mono(
						audio[:, ch], sr, strength, noise_type
					)
				return noisy, noise_signal
			else:
				noisy = np.zeros_like(audio)
				noise_signal = np.zeros_like(audio)
				for ch in range(audio.shape[0]):
					noisy[ch, :], noise_signal[ch, :] = self._add_adaptive_noise_mono(
						audio[ch, :], sr, strength, noise_type
					)
				return noisy, noise_signal
		else:
			return self._add_adaptive_noise_mono(audio, sr, strength, noise_type)

	def _add_adaptive_noise_mono(self, audio, sr, strength, noise_type):
		"""Fügt adaptives oder adversariales Rauschen hinzu"""
		def bandpass_filter(audio, sr, min_freq, max_freq, order=1):
			"""Bandpassfilter mit Butterworth"""
			from scipy.signal import butter, sosfilt
			nyquist = sr / 2
			low = min_freq / nyquist
			high = max_freq / nyquist
						
			sos = butter(order, [low, high], btype='band', output='sos')
			return sosfilt(sos, audio)
		
		# Parameter aus Config
		adv_epsilon = float(self.config.get('adv_epsilon', 0.01))
		adv_iterations = int(self.config.get('adv_iterations', 10))
		adv_alpha = float(self.config.get('adv_alpha', 0.002))
		patch_size = self.config.get('patch_size', 2048)
		
		# Filter-Parameter aus config (mit korrekten Typen)
		min_freq = int(self.config.get('min_freq_add_noise', 1000))
		max_freq = int(self.config.get('max_freq_add_noise', 20000))
		order = int(self.config.get('order_add_noise', 3))
		
		# Noise-Typen die gefiltert werden sollen (verwenden strength oder adv_epsilon)
		filter_noise_types = ['additive_adversarial', 'pgd', 'adversarial_patch', 
							  'universal', 'physical']
		
		# Noise generieren (je nach Typ)
		noise = None
		
		if noise_type == 'additive_adversarial':
			# Einfacher adversarialer Angriff (FGSM)
			grad = np.sign(np.gradient(audio))
			noise = adv_epsilon * grad
		
		elif noise_type == 'pgd':
			# Projected Gradient Descent Angriff (iterativ)
			noisy = np.copy(audio)
			for _ in range(adv_iterations):
				grad = np.sign(np.gradient(noisy))
				noisy = np.clip(noisy + adv_alpha * grad, audio - adv_epsilon, audio + adv_epsilon)
			noise = noisy - audio
		
		elif noise_type == 'spatial':
			# Räumliche Verzerrung (Pitch + Time)
			pitch_shift = float(self.config.get('spatial_pitch_shift', 1))
			time_shift = int(self.config.get('spatial_time_shift', 50))
			try:
				noisy = np.roll(librosa.effects.pitch_shift(y=audio, sr=sr, n_steps=pitch_shift), time_shift)
			except TypeError:
				noisy = np.roll(librosa.effects.pitch_shift(audio, sr, n_steps=pitch_shift), time_shift)
			noise = noisy - audio
		
		elif noise_type == 'adversarial_patch':
			# Lokales adversariales Patch-Rauschen
			patch_size = min(patch_size, len(audio) - 1)
			start = np.random.randint(0, len(audio) - patch_size)
			noise = np.zeros_like(audio)
			noise[start:start+patch_size] = np.random.randn(patch_size) * strength
		
		elif noise_type == 'universal':
			# Globales Noise-Profil (einmalig für alle Samples)
			if not hasattr(self, 'universal_noise'):
				np.random.seed(42)
				self.universal_noise = np.random.randn(len(audio)) * strength
			noise = self.universal_noise[:len(audio)]
		
		elif noise_type == 'physical':
			# Physikalisch motiviertes Rauschen (Reverb + Umgebung)
			reverb = librosa.effects.preemphasis(audio)
			env_noise = np.random.randn(len(audio)) * strength * 0.2
			noise = reverb * 0.1 + env_noise
		
		elif noise_type == 'semantic':
			# Semantisches Noise: Lautstärke + Geschwindigkeit
			gain = 1 + self.config.get('semantic_gain', 0.1) * (np.random.rand() - 0.5)
			speed = 1 + self.config.get('semantic_speed', 0.05) * (np.random.rand() - 0.5)
			noisy = librosa.effects.time_stretch(audio, rate=speed)[:len(audio)] * gain
			noise = noisy - audio
		
		else:
			# Fallback – FGSM
			grad = np.sign(np.gradient(audio))
			noise = adv_epsilon * grad
		
		# Bandpass-Filter nur auf bestimmte Noise-Typen anwenden
		# if noise_type in filter_noise_types and order > 0 and min_freq < max_freq:
		if noise_type in filter_noise_types:
			try:
				noise = bandpass_filter(noise, sr, min_freq, max_freq, order)
			except Exception as e:
				print(f"Bandpass filter failed: {e}, using unfiltered noise")
		
		# Gefiltertes/Ungefiltertes Noise zum Audio hinzufügen
		noisy_audio = audio + noise
		
		return np.clip(noisy_audio, -1, 1), noise
	
	
	def calculate_lufs(self, audio, sr):
		"""Berechnet LUFS"""
		meter = pyln.Meter(sr)
		
		if len(audio.shape) == 1:
			audio_for_lufs = np.stack([audio, audio], axis=1)
		elif audio.shape[1] == 1:
			audio_for_lufs = np.hstack([audio, audio])
		else:
			audio_for_lufs = audio
		
		try:
			loudness = meter.integrated_loudness(audio_for_lufs)
		except:
			loudness = -70.0
		
		return {
			'integrated': loudness,
			'peak': np.max(np.abs(audio))
		}
	
	def calculate_statistics(self, audio, sr):
		"""Berechnet Statistiken"""
		if len(audio.shape) == 2:
			if audio.shape[1] == 2:
				audio_mono = np.mean(audio, axis=1)
			elif audio.shape[0] == 2:
				audio_mono = np.mean(audio, axis=0)
			else:
				audio_mono = audio.flatten()
		else:
			audio_mono = audio
		
		rms = np.sqrt(np.mean(audio_mono**2))
		db_avg = 20 * np.log10(rms + 1e-10)
		db_max = 20 * np.log10(np.max(np.abs(audio_mono)) + 1e-10)
		
		return {
			'avg_db': db_avg,
			'max_db': db_max
		}
	
	def save_audio(self, audio, sr, output_path, format_ext, bitrate, bitdepth="24Bit", sample_freq="default"):
		"""Speichert Audio"""
		format_ext = format_ext.lower()

		# ------------------------------------------------------------
		# SAMPLE FREQUENCY PARSEN
		# ------------------------------------------------------------
		target_sr = sr

		if sample_freq != "default":
			# Strings wie "48k", "48kHz", "48000" unterstützen
			if isinstance(sample_freq, str):
				s = sample_freq.lower().replace("khz", "").replace("k", "")
				target_sr = int(float(s) * 1000) if float(s) < 1000 else int(s)
			elif isinstance(sample_freq, (int, float)):
				target_sr = int(sample_freq)

			# Resampling nur wenn notwendig
			if target_sr != sr:
				num_samples = int(len(audio) * target_sr / sr)
				audio = resample(audio, num_samples)
				sr = target_sr

		# ------------------------------------------------------------
		# MP3-EXPORT
		# ------------------------------------------------------------
		if format_ext == '.mp3':
			# Mono -> Stereo
			if len(audio.shape) == 1:
				audio = np.stack([audio, audio], axis=1)
			elif audio.shape[0] < audio.shape[1]:
				audio = audio.T

			# Skalieren für int16 WAV
			audio_int = (audio * 32767).astype(np.int16)

			temp_wav = output_path.with_suffix('.temp.wav')
			sf.write(temp_wav, audio_int, sr)

			sound = AudioSegment.from_wav(temp_wav)

			# --- BITRATE LOGIK ---
			if bitrate == "highest":
				sound.export(output_path, format='mp3', bitrate='320k')
			else:
				sound.export(output_path, format='mp3', bitrate=bitrate)

			temp_wav.unlink()
			return
			
		# ------------------------------------------------------------
		# WAV / LOSSLESS EXPORT
		# ------------------------------------------------------------
		bitdepth_map = {
			"16Bit": "PCM_16",
			"24Bit": "PCM_24",
			"32Bit": "PCM_32",
			"64Bit": "DOUBLE",
		}
		subtype = bitdepth_map.get(bitdepth, "PCM_24")

		# Normalisieren falls nötig
		if np.max(np.abs(audio)) > 1:
			audio = audio / np.max(np.abs(audio))

		sf.write(output_path, audio, sr, subtype=subtype)
	
	def plot_spectrogram(self, audio, sr, output_path):
		"""Erstellt Spektrogramm"""
		plt.figure(figsize=(32, 9))
		
		if len(audio.shape) == 2:
			if audio.shape[1] == 2:
				audio_mono = np.mean(audio, axis=1)
			elif audio.shape[0] == 2:
				audio_mono = np.mean(audio, axis=0)
			else:
				audio_mono = audio.flatten()
		else:
			audio_mono = audio
		
		D = librosa.amplitude_to_db(np.abs(librosa.stft(audio_mono)), ref=np.max)
		
		ax = plt.gca()
		img = librosa.display.specshow(D, sr=sr, x_axis='time', y_axis='hz', ax=ax)
		
		cbar = plt.colorbar(img, format='%+2.0f dB')
		cbar.ax.tick_params(labelsize=30)
		
		plt.title(f'{output_path.stem}', fontsize=30)
		plt.xlabel("Time (s)", fontsize=30)
		plt.ylabel("Frequency (Hz)", fontsize=30)
		
		ax.tick_params(axis='x', labelsize=30)
		ax.tick_params(axis='y', labelsize=30)
		
		plt.tight_layout(pad=1.0)
		plt.savefig(output_path, dpi=150)
		plt.close()
	
	def process_file(self, input_file, postprocess_settings=None):
		"""Verarbeitet eine Datei"""
		global postprocess_vars, postprocess_entries
		
		start_time = time.time()
		
		try:
			audio, sr = self.load_audio(input_file)
			original_size = os.path.getsize(input_file)
			
			lufs_original = self.calculate_lufs(audio, sr)
			stats_original = self.calculate_statistics(audio, sr)
			
			processed_audio = audio.copy()
			removed_noise = None
			added_noise = None
			
			if self.config['remove_noise']:
				processed_audio, removed_noise = self.remove_noise(
					processed_audio, 
					sr, 
					self.config['noise_removal_strength'], 
					# self.config['order_remove'],
					# self.config['denoise_min_freq'],
					# self.config['denoise_max_freq']
				)
			
			if self.config['add_noise']:
				processed_audio, added_noise = self.add_adaptive_noise(
					processed_audio, sr, 
					self.config['noise_addition_strength'],
					self.config['noise_type'],
					# self.config['order_add_noise'],
					# self.config['min_freq_add_noise'],
					# self.config['max_freq_add_noise']
				)
			
			# POST-PROCESSING: Settings zusammenstellen
			if self.config.get('postprocess_enabled', False):
				# Versuche aus postprocess_vars/entries zu lesen
				try:
					postprocess_settings = {
						'highpass_enabled': postprocess_vars.get('highpass_enabled', tk.BooleanVar(value=False)).get(),
						'highpass_freq': float(postprocess_entries.get('highpass_freq', ttk.Entry()).get() or 50),
						'lowpass_enabled': postprocess_vars.get('lowpass_enabled', tk.BooleanVar(value=False)).get(),
						'lowpass_freq': float(postprocess_entries.get('lowpass_freq', ttk.Entry()).get() or 16000),
						'preemphasis_enabled': postprocess_vars.get('preemphasis_enabled', tk.BooleanVar(value=False)).get(),
						'preemphasis_coef': float(postprocess_entries.get('preemphasis_coef', ttk.Entry()).get() or 0.85),
						'deemphasis_enabled': postprocess_vars.get('deemphasis_enabled', tk.BooleanVar(value=False)).get(),
						'deemphasis_coef': float(postprocess_entries.get('deemphasis_coef', ttk.Entry()).get() or 0.8),
						'eq_enabled': postprocess_vars.get('eq_enabled', tk.BooleanVar(value=False)).get(),
						'eq_freq': float(postprocess_entries.get('eq_freq', ttk.Entry()).get() or 8000),
						'eq_gain': float(postprocess_entries.get('eq_gain', ttk.Entry()).get() or 2.0),
						'compression_enabled': postprocess_vars.get('compression_enabled', tk.BooleanVar(value=False)).get(),
						'compression_ratio': float(postprocess_entries.get('compression_ratio', ttk.Entry()).get() or 1.2),
						'limiter_enabled': postprocess_vars.get('limiter_enabled', tk.BooleanVar(value=False)).get(),
						'limiter_threshold': float(postprocess_entries.get('limiter_threshold', ttk.Entry()).get() or 0.95),
						'normalize_enabled': postprocess_vars.get('normalize_enabled', tk.BooleanVar(value=False)).get(),
						'normalize_target': float(postprocess_entries.get('normalize_target', ttk.Entry()).get() or 0.989),
						'bass_mono_enabled': postprocess_vars.get('bass_mono_enabled', tk.BooleanVar(value=False)).get(),
						'bass_mono_freq': float(postprocess_entries.get('bass_mono_freq', ttk.Entry()).get() or 300)
					}
				except:
					# Fallback: leere Settings
					postprocess_settings = {}
				
				processed_audio = apply_postprocess_custom(processed_audio, sr, postprocess_settings)
			
			lufs_processed = self.calculate_lufs(processed_audio, sr)
			stats_processed = self.calculate_statistics(processed_audio, sr)
			
			
			# Unterstützung für mehrere Ausgabeformate
			output_ext = self.config['output_format'].lower()
			output_ext = [fmt.strip() for fmt in output_ext.split("&")]

			saved_paths = []

			for fmt in output_ext:
				if not fmt.startswith("."):
					fmt = "." + fmt

				output_name = input_file.stem + fmt
				output_path = self.output_folder / output_name

				self.save_audio(
					processed_audio, 
					sr, 
					output_path, 
					fmt,
					self.config['bitrate'], 
					self.config['bitdepth']
				)

				saved_paths.append(output_path)
			
			
			# REMOVED NOISE EXPORTER
			if self.config['write_remove_file'] and removed_noise is not None:
				for fmt in output_ext:
					if not fmt.startswith("."):
						fmt = "." + fmt

					noise_path = self.output_folder / f"{input_file.stem}_removed_noise{fmt}"
					self.save_audio(
						removed_noise,
						sr,
						noise_path,
						fmt,
						self.config['bitrate'],
						self.config['bitdepth']
					)

			# ADDED NOISE EXPORTER
			if self.config['write_add_file'] and added_noise is not None:
				for fmt in output_ext:
					if not fmt.startswith("."):
						fmt = "." + fmt

					noise_path = self.output_folder / f"{input_file.stem}_added_noise{fmt}"
					self.save_audio(
						added_noise,
						sr,
						noise_path,
						fmt,
						self.config['bitrate'],
						self.config['bitdepth']
					)
			
			plot_path = self.plots_folder / f"{input_file.stem}_spectrum.png"
			if self.config['export_plots']:
				self.plot_spectrogram(processed_audio, sr, plot_path)
			
			processing_time = time.time() - start_time
			output_size = os.path.getsize(output_path)
			
			if len(audio.shape) == 1:
				duration = len(audio) / sr
			else:
				duration = audio.shape[0] / sr

			file_stats = {
				'filename': input_file.name,
				'input_format': input_file.suffix,
				'output_format': output_ext,
				'input_size_mb': original_size / (1024**2),
				'output_size_mb': output_size / (1024**2),
				'sample_rate': sr,
				'duration_sec': duration,
				'bitrate': self.config['bitrate'],
				'bitdepth': self.config['bitdepth'],
				'noise_removal': self.config['remove_noise'],
				'noise_removal_strength': self.config['noise_removal_strength'] if self.config['remove_noise'] else 0,
				'noise_addition': self.config['add_noise'],
				'noise_addition_strength': self.config['noise_addition_strength'] if self.config['add_noise'] else 0,
				'noise_type': self.config['noise_type'] if self.config['add_noise'] else 'none',
				'lufs_integrated_original': lufs_original['integrated'],
				'lufs_integrated_processed': lufs_processed['integrated'],
				'peak_original': lufs_original['peak'],
				'peak_processed': lufs_processed['peak'],
				'avg_db_original': stats_original['avg_db'],
				'avg_db_processed': stats_processed['avg_db'],
				'max_db_original': stats_original['max_db'],
				'max_db_processed': stats_processed['max_db'],
				'processing_time_sec': processing_time,
				'postprocess_settings': postprocess_settings
			}

			self.stats.append(file_stats)

			return {
				"input_file": input_file,
				"original": audio,
				"processed": processed_audio,
				"removed": removed_noise,
				"added": added_noise,
				"sr": sr,
				"stats": file_stats,
				"plot_path": plot_path
			}
			
		except Exception as e:
			import traceback
			print(f"ERROR in process_file for {input_file.name}:")
			print(traceback.format_exc())
			raise
	
	def save_report(self):
		"""Speichert den vollständigen Processing Report"""
		report_path = self.output_folder / 'processing_report.txt'

		with open(report_path, 'w', encoding='utf-8') as f:
			f.write("=" * 80 + "\n")
			f.write("SCHNITTY'S NOISE MANAGER REPORT\n")
			f.write(f"Date time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
			f.write("=" * 80 + "\n\n")

			f.write(f"Processing files: {len(self.stats)}\n\n")

			for i, stat in enumerate(self.stats, 1):
				f.write(f"File #{i}: {stat['filename']}\n")
				f.write("-" * 80 + "\n")

				# --- File Info ---
				f.write("File Info:\n")
				f.write(f"  Format: {stat['input_format']} -> {stat['output_format']}\n")
				f.write(f"  Size: {stat['input_size_mb']:.2f} MB -> {stat['output_size_mb']:.2f} MB\n")
				f.write(f"  Sample Rate: {stat['sample_rate']} Hz\n")
				f.write(f"  Duration: {stat['duration_sec']:.2f} sec\n")
				f.write(f"  Bitrate: {stat.get('bitrate', 'N/A')}\n")
				f.write(f"  Bit Depth: {stat.get('bitdepth', 'N/A')}\n\n")

				# --- Noise Settings ---
				f.write("Noise Settings:\n")
				f.write(f"  Noise Removal: {'Yes' if stat.get('noise_removal') else 'No'}\n")
				f.write(f"  Noise Removal Strength: {stat.get('noise_removal_strength', 0):.2f}\n")
				f.write(f"  Noise Addition: {'Yes' if stat.get('noise_addition') else 'No'}\n")
				f.write(f"  Noise Addition Strength: {stat.get('noise_addition_strength', 0):.2f}\n")
				f.write(f"  Noise Type: {stat.get('noise_type', 'none')}\n\n")

				# --- Loudness & Levels ---
				f.write("Loudness & Levels:\n")
				f.write(f"  LUFS Integrated: Original {stat.get('lufs_integrated_original', 0):.2f} | Processed {stat.get('lufs_integrated_processed', 0):.2f}\n")
				f.write(f"  Peak Level: Original {stat.get('peak_original', 0):.2f} | Processed {stat.get('peak_processed', 0):.2f}\n")
				f.write(f"  Average dB: Original {stat.get('avg_db_original', 0):.2f} | Processed {stat.get('avg_db_processed', 0):.2f}\n")
				f.write(f"  Max dB: Original {stat.get('max_db_original', 0):.2f} | Processed {stat.get('max_db_processed', 0):.2f}\n\n")

				# --- Processing ---
				f.write("Processing:\n")
				f.write(f"  Processing Time: {stat.get('processing_time_sec', 0):.2f} sec\n")
				f.write(f"  Postprocess Settings: {stat.get('postprocess_settings', {})}\n")

				f.write("\n" + "=" * 80 + "\n\n")

		print(f"\nReport saved: {report_path}")



def start_gui():
	global notebook_plots, notebook_fileinfo
	global postprocess_vars, postprocess_entries, pause_flag, stop_flag
	
	selected_files = []
	checkbox_vars = []
	plot_tabs = {}
	remove_entries = {}
	add_entries = {}
	
	# Initialisiere postprocess Dictionaries HIER
	postprocess_vars = {}
	postprocess_entries = {}
		
	def update_file_info_tabs(stats_list):
		"""Erzeugt Datei-Info Tabs"""
		# Alle alten Tabs entfernen
		for tab_id in notebook_fileinfo.tabs():
			notebook_fileinfo.forget(tab_id)

		if not stats_list:
			frame_empty = ttk.Frame(notebook_fileinfo)
			lbl_empty = ttk.Label(
				frame_empty,
				text="No file information available yet.",
				justify="center",
				font=("Segoe UI", 11)
			)
			lbl_empty.pack(expand=True, pady=50)
			notebook_fileinfo.add(frame_empty, text="No Data")
			return

		for stat in stats_list:
			frame_stat = ttk.Frame(notebook_fileinfo)
			notebook_fileinfo.add(frame_stat, text=stat['filename'])

			# Text + Scrollbar sauber in einem Frame packen
			text_frame = ttk.Frame(frame_stat)
			text_frame.pack(fill="both", expand=True, padx=5, pady=5)

			text_widget = tk.Text(text_frame, wrap="word", height=25, font=("Consolas", 10))
			text_widget.pack(side="left", fill="both", expand=True)

			scrollbar = ttk.Scrollbar(text_frame, orient="vertical", command=text_widget.yview)
			scrollbar.pack(side="right", fill="y")
			text_widget.configure(yscrollcommand=scrollbar.set)

			# Infos einfügen

			info_text = (
				f"File Name: {stat['filename']}\n"
				f"Format: {stat['input_format']} -> {stat['output_format']}\n"
				f"Input Size: {stat['input_size_mb']:.2f} MB\n"
				f"Output Size: {stat['output_size_mb']:.2f} MB\n"
				f"Sample Rate: {stat['sample_rate']} Hz\n"
				f"Duration: {stat['duration_sec']:.2f} sec\n"
				f"Bitrate: {stat.get('bitrate', 'N/A')}\n"
				f"Bit Depth: {stat.get('bitdepth', 'N/A')}\n"
				f"\nNoise Removal: {'Yes' if stat.get('noise_removal') else 'No'}\n"
				f"Noise Removal Strength: {stat.get('noise_removal_strength', 0):.2f}\n"
				f"Noise Addition: {'Yes' if stat.get('noise_addition') else 'No'}\n"
				f"Noise Addition Strength: {stat.get('noise_addition_strength', 0):.2f}\n"
				f"Noise Type: {stat.get('noise_type', 'none')}\n"
				f"\nLUFS Integrated: Original {stat.get('lufs_integrated_original', 0):.2f} | Processed {stat.get('lufs_integrated_processed', 0):.2f}\n"
				f"Peak Level: Original {stat.get('peak_original', 0):.2f} | Processed {stat.get('peak_processed', 0):.2f}\n"
				f"Average dB: Original {stat.get('avg_db_original', 0):.2f} | Processed {stat.get('avg_db_processed', 0):.2f}\n"
				f"Max dB: Original {stat.get('max_db_original', 0):.2f} | Processed {stat.get('max_db_processed', 0):.2f}\n"
				f"\nProcessing Time: {stat.get('processing_time_sec', 0):.2f} sec\n"
				f"Postprocess Settings: {stat.get('postprocess_settings', {})}\n"
			)

			text_widget.insert("1.0", info_text)
			text_widget.configure(state="disabled")

	def add_audio_tab(name, images, metrics):
		"""Fügt Audio-Tab mit Spektrogrammen hinzu"""
		import matplotlib.image as mpimg
		
		tab = ttk.Frame(notebook_plots)
		notebook_plots.add(tab, text=name)
		notebook_plots.select(tab)

		fig, axes = plt.subplots(2, 2, figsize=(12, 9))
		canvas = FigureCanvasTkAgg(fig, master=tab)
		canvas.get_tk_widget().pack(fill="both", expand=True)

		def show_single_image(ax, img_path, title, metric_data):
			if img_path is None or not Path(img_path).exists():
				ax.text(0.5, 0.5, "No data", ha="center", va="center", fontsize=12)
				ax.set_title(title, fontsize=14)
				ax.axis("off")
				return
			
			try:
				img = mpimg.imread(img_path)
				ax.imshow(img, aspect='auto')
				ax.axis("off")
				
				lufs = metric_data.get("LUFS", "–")
				rms = metric_data.get("RMS", "–")
				maxv = metric_data.get("MAX", "–")
				title_text = f"{title}\nLUFS: {lufs} | RMS: {rms} dB | MAX: {maxv} dB"
				ax.set_title(title_text, fontsize=10, pad=10)
			except Exception as e:
				ax.text(0.5, 0.5, f"Error loading image:\n{str(e)}", 
						ha="center", va="center", fontsize=10)
				ax.set_title(title, fontsize=14)
				ax.axis("off")

		show_single_image(axes[0, 0], images.get("original"), "Original", metrics.get("original", {}))
		show_single_image(axes[0, 1], images.get("removed"), "Removed Noise", metrics.get("removed", {}))
		show_single_image(axes[1, 0], images.get("added"), "Added Noise", metrics.get("added", {}))
		show_single_image(axes[1, 1], images.get("processed"), "Processed", metrics.get("processed", {}))

		plt.tight_layout()
		canvas.draw()

		plot_tabs[name] = (fig, axes, canvas, tab)

	def browse_folder():
		"""Ordner auswählen und Audio-Dateien laden"""
		folder_selected = filedialog.askdirectory()
		if folder_selected:
			entry_input_folder.delete(0, tk.END)
			entry_input_folder.insert(0, folder_selected)
			load_audio_files(folder_selected)
	
	
	def load_audio_files(folder):
		"""Lädt Audio-Dateien"""
		for widget in scrollable_frame_cb.winfo_children():
			widget.destroy()
		checkbox_vars.clear()
		
		audio_extensions = ('.mp3', '.wav', '.flac', '.ogg', '.m4a', '.aiff', '.aif')
		
		try:
			files = [f for f in os.listdir(folder) if f.lower().endswith(audio_extensions)]
		except:
			files = []
		
		if not files:
			ttk.Label(scrollable_frame_cb, text="No audio files found.").pack(anchor="w", padx=10)
			return
		
		for file in sorted(files):
			var = tk.BooleanVar(value=True)
			chk = ttk.Checkbutton(scrollable_frame_cb, text=file, variable=var, 
								 command=update_selected_files)
			chk.pack(anchor="w", padx=10)
			checkbox_vars.append((file, var))

		update_selected_files()
	
	def update_selected_files():
		"""Aktualisiert ausgewählte Dateien"""
		nonlocal selected_files
		selected_files = [f for f, var in checkbox_vars if var.get()]
		print(f"DEBUG: {len(selected_files)} files selected")

	def toggle_all_checkboxes(state=True):
		"""Alle Checkboxen an/aus"""
		for _, var in checkbox_vars:
			var.set(state)
		update_selected_files()

	def clean_processing():
		"""Löscht generierte Ordner"""
		global pause_flag, stop_flag
		pause_flag = False
		stop_flag = False  

		folder = entry_input_folder.get().strip()
		if not folder:
			messagebox.showwarning("Warning", "No input folder specified!")
			return

		base_path = Path(folder)
		if not base_path.exists():
			messagebox.showwarning("Warning", f"Folder does not exist:\n{folder}")
			return

		confirm = messagebox.askyesno(
			"Confirmation",
			f"Really delete all generated subfolders in\n\n{folder}\n\n"
			"('plots' and 'converted')?"
		)
		if not confirm:
			return

		deleted_any = False
		for subdir in ["plots", "converted"]:
			target_dir = base_path / subdir
			if target_dir.exists():
				try:
					shutil.rmtree(target_dir)
					print(f"Deleted: {target_dir}")
					deleted_any = True
				except Exception as e:
					print(f"Could not delete {target_dir}: {e}")

		if not deleted_any:
			messagebox.showinfo("Info", "No generated folders found.")
		else:
			messagebox.showinfo("Done", "All generated folders deleted.")

		btn_pause.config(text="Pause", state="disabled")
		btn_start.config(state="normal")
		btn_clean.config(state="normal")
		progress_var.set(0)
		lbl_status.config(text="Cleaned and ready.")
		
		for tab_name in list(plot_tabs.keys()):
			fig, axes, canvas, tab = plot_tabs[tab_name]
			plt.close(fig)
			notebook_plots.forget(tab)
			del plot_tabs[tab_name]

	def pause_processing():
		global pause_flag
		pause_flag = not pause_flag

		if pause_flag:
			btn_pause.config(text="Resume")
			lbl_status.config(text="Paused...")
		else:
			btn_pause.config(text="Pause")
			lbl_status.config(text="Resuming...")
			
	def stop_processing():
		"""Stop processing"""
		global stop_flag
		stop_flag = True

		# UI anpassen
		btn_stop.config(state="disabled")
		btn_pause.config(state="disabled")
		btn_start.config(state="normal")
		lbl_status.config(text="Stopping...")

	def start_processing():
		"""Startet Verarbeitung"""
		global pause_flag, stop_flag

		stop_flag = False
		pause_flag = False

		btn_pause.config(text="Pause", state="normal")
		btn_stop.config(state="normal")
		btn_start.config(state="disabled")
		btn_clean.config(state="disabled")

		progress_var.set(0)
		lbl_status.config(text="Starting...")

		folder = entry_input_folder.get().strip()
		if not folder:
			messagebox.showerror("Error", "Choose an input folder.")
			return

		update_selected_files()

		if not selected_files:
			messagebox.showwarning("Warning", "No files selected!")
			return

		# --- Config sammeln ---
		config = {
			"input_folder": folder,
			"output_format": combo_format.get(),
			"bitrate": combo_bitrate.get(),
			"bitdepth": combo_bitdepth.get(),
			"sample_freq": combo_sample_freq.get(),
			"remove_noise": var_remove_noise.get(),
			"noise_removal_model": combo_noise_model.get(),
			"noise_removal_strength": float(remove_entries.get("strength", ttk.Entry()).get() or 0.1),
			"denoise_min_freq": float(remove_entries.get("min_freq", ttk.Entry()).get() or 1000),
			"denoise_max_freq": float(remove_entries.get("max_freq", ttk.Entry()).get() or 20000),
			"order_remove": float(remove_entries.get("order", ttk.Entry()).get() or 1),
			# "filter_type_remove": string(remove_entries.get("filter_type", ttk.Entry()).get() or "butterworth"),
			"denoise_percentile": float(remove_entries.get("percentile", ttk.Entry()).get() or 10),
			"denoise_epsilon": float(remove_entries.get("epsilon", ttk.Entry()).get() or 0.01),
			"gate_width": float(remove_entries.get("gate_width", ttk.Entry()).get() or 1.0),
			"denoise_min_freq_add": float(remove_entries.get("min_freq", ttk.Entry()).get() or 1000),
			"denoise_max_freq_add": float(remove_entries.get("max_freq", ttk.Entry()).get() or 20000),
			"order_add_noise": float(remove_entries.get("order_add", ttk.Entry()).get() or 1),
			"min_freq_add_noise": float(remove_entries.get("min_freq_add", ttk.Entry()).get() or 1000),
			"max_freq_add_noise":float(remove_entries.get("max_freq_add", ttk.Entry()).get() or 20000),			
			"add_noise": var_add_noise.get(),
			"noise_addition_strength": float(add_entries.get("strength", ttk.Entry()).get() or 0.1),
			"noise_type": combo_noise_type.get(),
			"export_plots": var_export_plots.get(),
			"write_remove_file": var_write_remove.get(),
			"write_add_file": var_write_add.get(),
			"postprocess_enabled": postprocess_vars.get('enabled', tk.BooleanVar(value=False)).get(),
			"noise_profile_path": entry_noise_profile.get() if var_use_noise_profile.get() else None,
		}

		# -----------------------------  
		# WORKER THREAD
		# -----------------------------
		def run_processing():
			global stop_flag, pause_flag

			try:
				processor = AudioProcessor(config, selected_files=selected_files)
				files = processor.get_audio_files()
				total_files = len(files)

				if total_files == 0:
					lbl_status.config(text="No files found.")
					return

				for i, file in enumerate(files, start=1):

					# --- STOP sofort ---
					if stop_flag:
						lbl_status.config(text="Processing stopped.")
						break

					# --- Pause Schleife ---
					while pause_flag:
						if stop_flag:
							lbl_status.config(text="Processing stopped.")
							break
						root.update()
						time.sleep(0.1)

					if stop_flag:
						break

					# --- Datei ---
					lbl_status.config(text=f"Processing {i}/{total_files}: {file.name}")
					root.update_idletasks()

					result = processor.process_file(file)

					if stop_flag:
						lbl_status.config(text="Processing stopped.")
						break

					progress_var.set(int(i / total_files * 100))
					root.update_idletasks()

					# ---------------------------------
					# EXPORT PLOTS (falls aktiviert)
					# ---------------------------------
					if config["export_plots"]:
						stem = file.stem
						plots_dir = Path(folder) / "plots"
						plots_dir.mkdir(exist_ok=True, parents=True)

						def _save_spec(arr, sr, outp):
							try:
								if arr is None:
									return None
								processor.plot_spectrogram(arr, sr, outp)
								return outp
							except:
								return None

						img_original  = _save_spec(result["original"],  result["sr"], plots_dir / f"{stem}_original.png")
						img_removed   = _save_spec(result["removed"],   result["sr"], plots_dir / f"{stem}_removed.png")
						img_added     = _save_spec(result["added"],     result["sr"], plots_dir / f"{stem}_added.png")
						img_processed = _save_spec(result["processed"], result["sr"], plots_dir / f"{stem}_processed.png")

						# kleine Werte
						def _metric(arr, sr):
							if arr is None:
								return {"LUFS": "–", "RMS": "–", "MAX": "–"}
							LUFS = processor.calculate_lufs(arr, sr)["integrated"]
							stats = processor.calculate_statistics(arr, sr)
							return {
								"LUFS": f"{LUFS:.2f}",
								"RMS": f"{stats['avg_db']:.2f}",
								"MAX": f"{stats['max_db']:.2f}",
							}

						metrics = {
							"original":  _metric(result["original"],  result["sr"]),
							"removed":   _metric(result["removed"],   result["sr"]),
							"added":     _metric(result["added"],     result["sr"]),
							"processed": _metric(result["processed"], result["sr"]),
						}

						add_audio_tab(
							file.name,
							images={
								"original":  str(img_original),
								"removed":   str(img_removed),
								"added":     str(img_added),
								"processed": str(img_processed),
							},
							metrics=metrics
						)

				# --- Report + Info Tabs ---
				if not stop_flag:
					processor.save_report()
					update_file_info_tabs(processor.stats)

			except Exception as e:
				import traceback
				messagebox.showerror("Error", f"{e}\n\n{traceback.format_exc()}")

			finally:
				btn_start.config(state="normal")
				btn_pause.config(state="disabled")
				btn_stop.config(state="disabled")
				btn_clean.config(state="normal")

				if stop_flag:
					lbl_status.config(text="Stopped by user.")
				else:
					lbl_status.config(text="Finished.")
					messagebox.showinfo("Finished", f"{total_files} file(s) processed.")

		# THREAD starten
		threading.Thread(target=run_processing, daemon=True).start()


	# GUI Setup
	root = tk.Tk()
	root.title("Schnitty's Noise Manager")
	try:
		root.iconbitmap("icon.ico")
	except:
		pass
	root.geometry("1000x1200")
	root.minsize(800, 500)
	
	main_paned = ttk.PanedWindow(root, orient=tk.VERTICAL)
	main_paned.pack(fill="both", expand=True)

	style = ttk.Style()
	style.theme_use("clam")


	frame_top = ttk.Frame(main_paned, padding=5)
	main_paned.add(frame_top, weight=1)
	
	progress_var = tk.IntVar()
	progress = ttk.Progressbar(frame_top, orient="horizontal", mode="determinate", variable=progress_var)
	progress.pack(fill="x", pady=5)
	lbl_status = ttk.Label(frame_top, text="Waiting for start...")
	lbl_status.pack()


	frame_top_controls = ttk.Frame(frame_top)
	frame_top_controls.pack(fill="x", pady=5)
	
	# Farbe Erstellen des Buttons
	# Grün
	style.configure("Green.TButton", background="#28a745", foreground="white")
	style.map("Green.TButton",
			  background=[("active", "#218838")])

	# Rot
	style.configure("Red.TButton", background="#dc3545", foreground="white")
	style.map("Red.TButton",
			  background=[("active", "#c82333")])

	# Orange
	style.configure("Orange.TButton", background="#fd7e14", foreground="white")
	style.map("Orange.TButton",
			  background=[("active", "#e8590c")])

	# Magenta
	style.configure("Magenta.TButton", background="#d63384", foreground="white")
	style.map("Magenta.TButton",
			  background=[("active", "#c2256d")])
			  


	btn_start = ttk.Button(frame_top_controls, text="Start", command=start_processing, style="Green.TButton")
	btn_start.pack(side="left", padx=5)

	btn_pause = ttk.Button(frame_top_controls, text="Pause", command=pause_processing, state="disabled")
	btn_pause.pack(side="left", padx=5)
	
	
	btn_stop = ttk.Button(frame_top_controls, text="Stop", command=stop_processing, state="disabled", style="Red.TButton")
	btn_stop.pack(side="left", padx=5)


	btn_clean = ttk.Button(frame_top_controls, text="Clean Folder", command=clean_processing, style="Orange.TButton")
	btn_clean.pack(side="left", padx=5)

	
	var_export_plots = tk.BooleanVar(value=False)
	chk_export_plots = ttk.Checkbutton(frame_top_controls, text="Spectrum analyis", variable=var_export_plots)
	chk_export_plots.pack(side="left", padx=5)
	
	
	notebook_noise = ttk.Notebook(frame_top)
	notebook_noise.pack(fill="both", expand=True, pady=12)

	frame_setup_main = ttk.Frame(notebook_noise)
	notebook_noise.add(frame_setup_main, text="Setup")

	
	#Start Setup
	#Sheets
	notebook_setup = ttk.Notebook(frame_setup_main)
	notebook_setup.pack(fill="both", expand=True, padx=5, pady=5)
	
	frame_general_tab = ttk.Frame(notebook_setup)
	notebook_setup.add(frame_general_tab, text="General Setup")

	frame_remove_tab = ttk.Frame(notebook_setup)
	notebook_setup.add(frame_remove_tab, text="Remove Noise")

	frame_add_tab = ttk.Frame(notebook_setup)
	notebook_setup.add(frame_add_tab, text="Add Noise")

	frame_post_tab = ttk.Frame(notebook_setup)
	notebook_setup.add(frame_post_tab, text="Post-Processing")
	
	
	frame_folder = ttk.Frame(frame_general_tab)
	frame_folder.pack(fill="x", pady=2)
	ttk.Label(frame_folder, text="Select File:").pack(anchor="w")
	# ttk.Label(frame_remove, text="Select File:").grid(row=3, column=0, sticky="w", pady=2)
	entry_input_folder = ttk.Entry(frame_folder)
	entry_input_folder.pack(side="left", fill="x", expand=True)
	ttk.Button(frame_folder, text="Browse...", command=browse_folder).pack(side="left", padx=5)

	frame_checkboxes = ttk.LabelFrame(frame_general_tab, text="Available Files")
	frame_checkboxes.pack(fill="both", expand=False, padx=5, pady=5)

	canvas_cb = tk.Canvas(frame_checkboxes, height=80)
	
	def _on_mousewheel(event):
		canvas_cb.yview_scroll(int(-1*(event.delta/120)), "units")
	
	scrollbar_cb = ttk.Scrollbar(frame_checkboxes, orient="vertical", command=canvas_cb.yview)
	scrollable_frame_cb = ttk.Frame(canvas_cb)

	scrollable_frame_cb.bind("<Configure>", lambda e: canvas_cb.configure(scrollregion=canvas_cb.bbox("all")))
	canvas_cb.create_window((0, 0), window=scrollable_frame_cb, anchor="nw")
	canvas_cb.configure(yscrollcommand=scrollbar_cb.set)
	canvas_cb.bind_all("<MouseWheel>", _on_mousewheel)
	
	canvas_cb.pack(side="left", fill="both", expand=True)
	scrollbar_cb.pack(side="right", fill="y")

	frame_select_buttons = ttk.Frame(frame_general_tab)
	frame_select_buttons.pack(fill="x", pady=5)
	btn_select_all = ttk.Button(frame_select_buttons, text="Select all", command=lambda: toggle_all_checkboxes(True))
	btn_select_all.pack(side="left", padx=5)
	btn_deselect_all = ttk.Button(frame_select_buttons, text="Deselect all", command=lambda: toggle_all_checkboxes(False))
	btn_deselect_all.pack(side="left", padx=5)
	
	# def open_output_folder():
		# folder = output_dir  # oder wie dein Ordner heißt

		# if platform.startswith("win"):
			# os.startfile(folder)
		# elif platform == "darwin":  # macOS
			# subprocess.Popen(["open", folder])
		# else:  # Linux
			# subprocess.Popen(["xdg-open", folder])
	
	# btn_open_folder = ttk.Button(frame_select_buttons, text="Open Folder", command=open_output_folder)
	# btn_open_folder.pack(side="left", padx=5)
	


	ttk.Label(frame_general_tab, text="Output format:").pack(anchor="w")
	combo_format = ttk.Combobox(frame_general_tab, values=[".wav", ".mp3", ".flac", ".aiff", ".mp3&.wav"], state="readonly")
	combo_format.set(".wav")
	combo_format.pack(fill="x", pady=2)

	ttk.Label(frame_general_tab, text="Bitrate:").pack(anchor="w")
	combo_bitrate = ttk.Combobox(frame_general_tab, values=["64k","96k","128k","160k","192k","224k","256k","320k","highest"], state="readonly")
	combo_bitrate.set("highest")
	combo_bitrate.pack(fill="x", pady=2)
	
	ttk.Label(frame_general_tab, text="Bitdepth:").pack(anchor="w")
	combo_bitdepth = ttk.Combobox(frame_general_tab, values=["16Bit","24Bit","32Bit","64Bit", "default"], state="readonly")
	combo_bitdepth.set("default")
	combo_bitdepth.pack(fill="x", pady=2)
	
	ttk.Label(frame_general_tab, text="Sample Frequency:").pack(anchor="w")
	combo_sample_freq = ttk.Combobox(frame_general_tab, values=["44.1kHz", "96kHz", "192kHz","default"], state="readonly")
	combo_sample_freq.set("default")
	combo_sample_freq.pack(fill="x", pady=2)
	

	# === Remove Noise Section ===
	frame_remove = ttk.LabelFrame(frame_remove_tab, text="Remove Noise", padding=10)
	frame_remove.pack(side="left", fill="both", expand=True, padx=(0, 5))

	# --- Noise Profile Section ---
	frame_noise_profile = ttk.LabelFrame(frame_remove, text="Noise Profile (Optional)", padding=5)
	frame_noise_profile.grid(row=12, column=0, columnspan=3, sticky="ew", pady=10)

	var_use_noise_profile = tk.BooleanVar(value=False)
	chk_use_profile = ttk.Checkbutton(
		frame_noise_profile, 
		text="Use external noise profile file", 
		variable=var_use_noise_profile
	)
	chk_use_profile.grid(row=0, column=0, columnspan=3, sticky="w", pady=5)

	ttk.Label(
		frame_noise_profile, 
		text="Select a short audio file (2-5s) containing ONLY the noise you want to remove.\nThis profile will be used for all processed files.",
		foreground="gray",
		wraplength=450,
		justify="left"
	).grid(row=1, column=0, columnspan=3, sticky="w", padx=5)

	entry_noise_profile = ttk.Entry(frame_noise_profile, width=40)
	entry_noise_profile.grid(row=2, column=0, padx=5, pady=5, sticky="ew")

	def browse_noise_profile():
		file_path = filedialog.askopenfilename(
			title="Select Noise Profile File",
			filetypes=[
				("Audio Files", "*.wav *.mp3 *.flac *.aiff *.aif"),
				("All Files", "*.*")
			]
		)
		if file_path:
			entry_noise_profile.delete(0, tk.END)
			entry_noise_profile.insert(0, file_path)
			var_use_noise_profile.set(True)

	btn_browse_profile = ttk.Button(frame_noise_profile, text="Browse...", command=browse_noise_profile)
	btn_browse_profile.grid(row=2, column=1, padx=5)

	btn_clear_profile = ttk.Button(frame_noise_profile, text="Clear", command=lambda: [entry_noise_profile.delete(0, tk.END), var_use_noise_profile.set(False)])
	btn_clear_profile.grid(row=2, column=2, padx=5)

	# --- Enable/Write ---
	var_remove_noise = tk.BooleanVar()
	chk_remove_noise = ttk.Checkbutton(frame_remove, text="Enable", variable=var_remove_noise)
	chk_remove_noise.grid(row=0, column=0, sticky="w", pady=(0, 5))

	var_write_remove = tk.BooleanVar()
	chk_write_remove = ttk.Checkbutton(frame_remove, text="Write File", variable=var_write_remove)
	chk_write_remove.grid(row=0, column=1, sticky="w", pady=(0, 5))

	# --- Model Selection ---
	ttk.Label(frame_remove, text="Model:").grid(row=2, column=0, sticky="w", pady=2)
	available_models = ["spectral gating", "deepfilternet", "speechbrain", "demucs", "asteroid"]
	combo_noise_model = ttk.Combobox(frame_remove, values=available_models, state="readonly", width=20)
	combo_noise_model.set(available_models[0])
	combo_noise_model.grid(row=2, column=1, padx=5, pady=2, sticky="w")
	
	# Filter Type
	# ttk.Label(frame_remove, text="Filter Type:").grid(row=3, column=0, sticky="w", pady=2)
	# combo_filter_type = ttk.Combobox(frame_remove, values=[
		# "butterworth", "chebyshev i", "chebyshev ii", "elliptic"
	# ], state="readonly", width=20)
	# combo_filter_type.set("butterworth")
	# combo_filter_type.grid(row=3, column=1, padx=5, pady=2, sticky="w")
	# ttk.Label(frame_remove, text="Select bandpass filter type. Butterworth(soft cutofff)-> Elliptic(hard cutoff).", foreground="gray", wraplength=240, justify="left").grid(row=3, column=2, sticky="w", padx=5)


	# --- Remove Noise Parameters ---
	remove_entries = {}
	params_remove = [
		("order", "Filter Order (0..6):", "3", "Higher orders leads to a sharper cutoffs."),
		("min_freq", "Min Frequency (Hz):", "2000", ""),
		("max_freq", "Max Frequency (Hz):", "20000", ""),
		("gate_width", "Gate Width (1–100):", "3", "Smoothing window for noise gating. Higher values make gating more stable, \nlower values more precise."),
		("percentile", "Percentile (1–100):", "4", "Threshold for noise detection. Higher values remove more quiet content."),
		("epsilon", "Re. Epsilon (0–1):", "0.007", "Sensitivity factor. Larger values apply stronger filtering but may remove details."),
		("strength", "Strength (0–1):", ".4", "Overall denoising strength."),
	]

	# def on_manual_change(*args):
		# if combo_presets_remove.get() != "Custom":
			# combo_presets_remove.set("Custom")

	for i, (key, label_text, default, desc) in enumerate(params_remove, start=3):
		ttk.Label(frame_remove, text=label_text).grid(row=i, column=0, sticky="w", pady=2)
		entry = ttk.Entry(frame_remove, width=8)
		entry.insert(0, default)
		entry.grid(row=i, column=1, padx=5, pady=2, sticky="w")
		# entry.bind("<KeyRelease>", on_manual_change)
		ttk.Label(frame_remove, text=desc, foreground="gray", wraplength=240, justify="left").grid(row=i, column=2, sticky="w", padx=5)
		remove_entries[key] = entry

	# --- Frames für Modell-spezifische Einstellungen ---
	model_frames = {}

	# Spectral Gating
	frame_spectral = ttk.Frame(frame_remove)
	model_frames["spectral gating"] = frame_spectral
	# Hier könnten die Filterparameter aus remove_entries wieder eingebunden werden

	# DeepFilterNet
	frame_df = ttk.Frame(frame_remove)
	model_frames["deepfilternet"] = frame_df
	ttk.Label(frame_df, text="DF Gain (0–1):").grid(row=0, column=0, sticky="w")
	entry_df_gain = ttk.Entry(frame_df, width=8); entry_df_gain.insert(0, "0.8"); entry_df_gain.grid(row=0, column=1, padx=5)
	ttk.Label(frame_df, text="Strength of the neural denoising. Higher values remove more noise but can make speech sound artificial.", foreground="gray", wraplength=240, justify="left").grid(row=0, column=2, sticky="w", padx=5)

	ttk.Label(frame_df, text="Residual (Original Mix 0–1):").grid(row=1, column=0, sticky="w")
	entry_df_residual = ttk.Entry(frame_df, width=8); entry_df_residual.insert(0, "0.2"); entry_df_residual.grid(row=1, column=1, padx=5)
	ttk.Label(frame_df, text="How much of the original signal is mixed back in. Higher values retain natural sound but leave more noise.", foreground="gray", wraplength=240, justify="left").grid(row=1, column=2, sticky="w", padx=5)

	ttk.Label(frame_df, text="Smoothing (0–1):").grid(row=2, column=0, sticky="w")
	entry_df_smooth = ttk.Entry(frame_df, width=8); entry_df_smooth.insert(0, "0.5"); entry_df_smooth.grid(row=2, column=1, padx=5)
	ttk.Label(frame_df, text="Temporal smoothing between frames. Higher values prevent artifacts but may blur fast details.", foreground="gray", wraplength=240, justify="left").grid(row=2, column=2, sticky="w", padx=5)

	# SpeechBrain
	frame_sb = ttk.Frame(frame_remove)
	model_frames["speechbrain"] = frame_sb
	ttk.Label(frame_sb, text="Noise Reduction Level (0–1):").grid(row=0, column=0, sticky="w")
	entry_sb_level = ttk.Entry(frame_sb, width=8); entry_sb_level.insert(0, "0.6"); entry_sb_level.grid(row=0, column=1, padx=5)
	ttk.Label(frame_sb, text="Controls how aggressively background noise is removed. Higher values = stronger suppression.", foreground="gray", wraplength=240, justify="left").grid(row=0, column=2, sticky="w", padx=5)

	ttk.Label(frame_sb, text="Clarity Boost (0–1):").grid(row=1, column=0, sticky="w")
	entry_sb_clarity = ttk.Entry(frame_sb, width=8); entry_sb_clarity.insert(0, "0.5"); entry_sb_clarity.grid(row=1, column=1, padx=5)
	ttk.Label(frame_sb, text="Enhances high-frequency details to restore speech brightness. Too high -> harsh sound.", foreground="gray", wraplength=240, justify="left").grid(row=1, column=2, sticky="w", padx=5)

	ttk.Label(frame_sb, text="Voice Emphasis (0–1):").grid(row=2, column=0, sticky="w")
	entry_sb_voice = ttk.Entry(frame_sb, width=8); entry_sb_voice.insert(0, "0.7"); entry_sb_voice.grid(row=2, column=1, padx=5)
	ttk.Label(frame_sb, text="Increases separation focus on speech components. Higher values make voices stand out more.", foreground="gray", wraplength=240, justify="left").grid(row=2, column=2, sticky="w", padx=5)

	# Demucs
	frame_demucs = ttk.Frame(frame_remove)
	model_frames["demucs"] = frame_demucs
	ttk.Label(frame_demucs, text="Quality Mode:").grid(row=0, column=0, sticky="w")
	combo_demucs_quality = ttk.Combobox(frame_demucs, values=["fast","balanced","best"], state="readonly", width=12)
	combo_demucs_quality.set("balanced"); combo_demucs_quality.grid(row=0, column=1, padx=5)
	ttk.Label(frame_demucs, text="Changes model complexity and processing speed. 'best' provides highest clarity but is slowest.", foreground="gray", wraplength=240, justify="left").grid(row=0, column=2, sticky="w", padx=5)

	ttk.Label(frame_demucs, text="Dry/Wet Mix (0–1):").grid(row=1, column=0, sticky="w")
	entry_demucs_mix = ttk.Entry(frame_demucs, width=8); entry_demucs_mix.insert(0, "0.85"); entry_demucs_mix.grid(row=1, column=1, padx=5)
	ttk.Label(frame_demucs, text="0 = original sound only (no processing), 1 = full denoised signal. Adjust to retain natural tone.", foreground="gray", wraplength=240, justify="left").grid(row=1, column=2, sticky="w", padx=5)

	ttk.Label(frame_demucs, text="Wiener Denoise:").grid(row=2, column=0, sticky="w")
	var_demucs_wiener = tk.BooleanVar(value=True)
	chk_demucs_wiener = ttk.Checkbutton(frame_demucs, variable=var_demucs_wiener); chk_demucs_wiener.grid(row=2, column=1, padx=5)
	ttk.Label(frame_demucs, text="Applies Wiener filtering after separation to reduce residual artifacts. May smooth speech slightly.", foreground="gray", wraplength=240, justify="left").grid(row=2, column=2, sticky="w", padx=5)

	# Asteroid
	frame_asteroid = ttk.Frame(frame_remove)
	model_frames["asteroid"] = frame_asteroid
	ttk.Label(frame_asteroid, text="Asteroid works fully automatically. It does not provide internal control parameters.", foreground="gray").grid(row=0, column=0, sticky="w")

	# Noise Type Descriptions
	remove_noise_modes = {
		"spectral gating": "Removes noise by analyzing spectral statistics and suppressing quiet components. Best for constant/steady background noise (hiss, hum, fan). Provides precise control via frequency range, percentile threshold, gate width and filter sharpness.",
		"deepfilternet": "Learns speech and noise characteristics and filters noise frame-by-frame. Produces clean results with minimal artifacts at moderate settings. Ideal for voice recordings with non-stationary noise, chatter or street noise.",
		"speechbrain": "Strong at isolating human speech from complex environment noise. Can enhance clarity and boost important vocal frequencies. Works best for speech-only material; may color music slightly.",
		"demucs": "Advanced source separation using deep learning. Separates audio into clean components (speech/vocals, noise, etc.). Offers quality modes and mixing controls for flexible processing.",
		"asteroid": "Fully automatic neural separation system requiring no tuning. Optimized for speech restoration in high-noise conditions. Very robust but offers no internal parameters to adjust.",
	}

	lbl_remove_noise_desc = ttk.Label(frame_remove, text=remove_noise_modes["spectral gating"], foreground="gray", wraplength=500, justify="left")
	lbl_remove_noise_desc.grid(row=99, column=0, columnspan=3, sticky="w", pady=(10,0))

	# Initial alle Frames verstecken
	for f in model_frames.values():
		f.grid(row=11, column=0, columnspan=3, sticky="w", pady=5)
		f.grid_remove()

	# Update Funktion
	def update_remove_noise_description(event=None):
		selected_model = combo_noise_model.get()
		lbl_remove_noise_desc.config(text=remove_noise_modes.get(selected_model, f"Keine Beschreibung für '{selected_model}' gefunden."))
		for frame in model_frames.values():
			frame.grid_remove()
		if selected_model in model_frames:
			model_frames[selected_model].grid()  # Frame einblenden

	# Event-Binding
	combo_noise_model.bind("<<ComboboxSelected>>", update_remove_noise_description)

	# Initial aufrufen
	update_remove_noise_description()



	### Add Noise Section
	#######################	
	noise_presets = {
		"Music Protection I": {
			"noise_type": "additive_adversarial",
			"adv_epsilon": 0.0001,
			"strength": 0.005
		},
		"Music Protection II": {
			"noise_type": "pgd",
			"adv_epsilon": 0.001,
			"iterations": 4,
			"strength": 0.001
		},		
		"Music Protection III – Psychoacoustic FGSM": {
			"noise_type": "additive_adversarial",
			"adv_epsilon": 0.0004,
			"strength": 0.0002
		},		
		"Music Protection IV – Soft Universal": {
			"noise_type": "universal",
			"strength": 0.0015
		},
		"Music Protection V – Anti-Fingerprint": {
			"noise_type": "semantic",
			"strength": 0.002,
			"semantic_gain": 0.03,
			"semantic_speed": 0.015
		},
		
		"Music Protection VI – Separation Patch": {
			"noise_type": "adversarial_patch",
			"patch_size": 4096,
			"strength": 0.004
		},
		
		"Music Protection VII – Demucs PGD": {
			"noise_type": "pgd",
			"adv_epsilon": 0.002,
			"iterations": 8,
			"strength": 0.0015
		},
		
		"Music Protection VIII – Analog Drift": {
			"noise_type": "spatial",
			"pitch_shift": 0.05,
			"time_shift": 2.0
		},
		
		"Music Protection IX – Physical Blur": {
			"noise_type": "physical",
			"strength": 0.004
		},
		
		"Music Protection X – Hybrid Defense": {
			"noise_type": "pgd",
			"adv_epsilon": 0.0015,
			"iterations": 3,
			"strength": 0.002,
			"patch_size": 2048
		},
		
		"Music Protection XI – Whisper Guard": {
			"noise_type": "universal",
			"strength": 0.0008
		},
		
		"Music Protection XII – Diffusion Confuser": {
			"noise_type": "semantic",
			"semantic_gain": 0.05,
			"semantic_speed": 0.02,
			"strength": 0.003
		},
		
		"Music Protection XIII – Vinyl Warp": {
			"noise_type": "spatial",
			"pitch_shift": 0.12,
			"time_shift": 4.5
		},
		
		"Tape Wow/Flutter (vintage)": {
			"noise_type": "spatial",
			"pitch_shift": 0.08,
			"time_shift": 3.0
		},
		
		"Tape Wow/Flutter (light)": {
			"noise_type": "spatial",
			"pitch_shift": 0.03,   # 0.03 semitones (extremely subtle)
			"time_shift": 1.5	 # small ms drift
		},
	}

	def apply_preset(*args):
		preset_name = combo_preset.get()
		if not preset_name:
			return

		preset = noise_presets[preset_name]

		# Noise Type setzen
		if "noise_type" in preset:
			combo_noise_type.set(preset["noise_type"])
			on_noise_type_change()  # UI aktualisieren

		# FGSM / PGD Parameter
		if "adv_epsilon" in preset:
			add_entries["adv_epsilon"].delete(0, "end")
			add_entries["adv_epsilon"].insert(0, preset["adv_epsilon"])

		if "iterations" in preset:
			add_entries["iterations"].delete(0, "end")
			add_entries["iterations"].insert(0, preset["iterations"])

		if "patch_size" in preset:
			add_entries["patch_size"].delete(0, "end")
			add_entries["patch_size"].insert(0, preset["patch_size"])

		if "strength" in preset:
			add_entries["strength"].delete(0, "end")
			add_entries["strength"].insert(0, preset["strength"])

		# Spatial / Tape params
		if combo_noise_type.get() == "spatial":
			if "pitch_shift" in preset:
				entry_pitch_shift.delete(0, "end")
				entry_pitch_shift.insert(0, preset["pitch_shift"])

			if "time_shift" in preset:
				entry_time_shift.delete(0, "end")
				entry_time_shift.insert(0, preset["time_shift"])
	
		update_visible_add_params(combo_noise_type.get())
	
	frame_add = ttk.LabelFrame(frame_add_tab, text="Add Noise", padding=10)
	frame_add.pack(side="left", fill="both", expand=True, padx=(5, 0))


	var_add_noise = tk.BooleanVar()
	chk_add_noise = ttk.Checkbutton(frame_add, text="Enable", variable=var_add_noise)
	chk_add_noise.grid(row=0, column=0, sticky="w", pady=(0, 5))
	
				
	var_write_add = tk.BooleanVar()
	chk_write_add = ttk.Checkbutton(frame_add, text="Write File", variable=var_write_add)
	chk_write_add.grid(row=0, column=1, sticky="w", pady=(0, 5))

	ttk.Label(frame_add, text="Preset:").grid(row=1, column=0, sticky="w")

	combo_preset = ttk.Combobox(
		frame_add, 
		values=list(noise_presets.keys()),
		state="readonly",
		width=40
	)
	combo_preset.grid(row=1, column=1, sticky="w", padx=5, pady=2)
		
	combo_preset.bind("<<ComboboxSelected>>", apply_preset)



	ttk.Label(frame_add, text="Noise Type:").grid(row=2, column=0, sticky="w")
	combo_noise_type = ttk.Combobox(frame_add, values=[
		"additive_adversarial", "pgd", "spatial", "adversarial_patch", 
		"universal", "physical", "semantic"
	], state="readonly", width=40)
	combo_noise_type.set("additive_adversarial")
	combo_noise_type.grid(row=2, column=1, padx=5, pady=2, sticky="w")

	# Add Noise Parameters
	param_frames = {}   # <-- neues dict

	params_add = [
		("order_add", "Filter Order (0..6):", "3", "Higher orders lead to sharper cutoffs."),
		("min_freq_add", "Min Frequency (Hz):", "2000", "Lower frequency boundary for filtering."),
		("max_freq_add", "Max Frequency (Hz):", "20000", "Upper frequency boundary for filtering."),
		("adv_epsilon", "Adv. Epsilon (0–1):", "0.01", "Maximum perturbation strength."),
		("iterations", "Iterations (1–20):", "3", "Number of optimization steps."),
		("patch_size", "Patch Size (512-65k):", "2048", "Length of audio patch."),
		("strength", "Strength (0–1):", "0.01", "Overall noise intensity."),
	]

	add_entries = {}	 # key -> Entry widget
	param_widgets = {}   # key -> (label_widget, entry_widget, desc_widget)

	row_index = 3
	for key, label_text, default, desc in params_add:
		# Label in column 0
		lbl = ttk.Label(frame_add, text=label_text)
		lbl.grid(row=row_index, column=0, sticky="w", pady=2)

		# Entry in column 1
		entry = ttk.Entry(frame_add, width=8)
		entry.insert(0, default)
		entry.grid(row=row_index, column=1, padx=5, pady=2, sticky="w")

		# Description in column 2
		lbl_desc = ttk.Label(frame_add, text=desc, foreground="gray", wraplength=240, justify="left")
		lbl_desc.grid(row=row_index, column=2, sticky="w", padx=5)

		add_entries[key] = entry
		param_widgets[key] = (lbl, entry, lbl_desc)

		row_index += 1


	# Spatial Parameters
	frame_spatial_params = ttk.Frame(frame_add)
	frame_spatial_params.grid(row=7, column=0, columnspan=3, sticky="w", pady=(5, 0))
	frame_spatial_params.grid_remove()

	ttk.Label(frame_spatial_params, text="Pitch Shift (1–12):").grid(row=0, column=0, sticky="w", pady=2)
	entry_pitch_shift = ttk.Entry(frame_spatial_params, width=8)
	entry_pitch_shift.insert(0, "1")
	entry_pitch_shift.grid(row=0, column=1, padx=5, pady=2)

	ttk.Label(frame_spatial_params, text="Time Shift (ms):").grid(row=1, column=0, sticky="w", pady=2)
	entry_time_shift = ttk.Entry(frame_spatial_params, width=8)
	entry_time_shift.insert(0, "3")
	entry_time_shift.grid(row=1, column=1, padx=5, pady=2)

	# Noise Type Descriptions
	noise_type_descriptions = {
		"additive_adversarial": "FGSM: Single-step attack. Adds noise in direction of gradient. Fast but less powerful. Good for basic robustness testing.",
		"pgd": "PGD: Multi-step iterative attack. Stronger than FGSM. Better adversarial examples.",
		"spatial": "Spatial distortion: Combines pitch and time shifting to create perceptually similar but digitally different audio.",
		"adversarial_patch": "Localized patch attack: Injects adversarial noise into random segments of the audio.",
		"universal": "Universal perturbation: Same noise pattern applied to all samples.",
		"physical": "Physical-world noise: Combines reverb simulation and environmental noise.",
		"semantic": "Semantic-level attack: Modifies gain and playback speed randomly. Preserves speech content but alters acoustic properties."
	}

	# Label für Beschreibung am unteren Ende
	lbl_noise_desc = ttk.Label(frame_add, text=noise_type_descriptions[combo_noise_type.get()],
							   foreground="gray", wraplength=500, justify="left")
	lbl_noise_desc.grid(row=99, column=0, columnspan=3, sticky="w", pady=(10, 0))

	visible_params_by_type = {
		"additive_adversarial": ["adv_epsilon", "strength", "order_add", "min_freq_add", "max_freq_add"],
		"pgd": ["adv_epsilon", "iterations", "strength", "order_add", "min_freq_add", "max_freq_add"],
		"spatial": ["pitch_shift", "time_shift"],
		"adversarial_patch": ["patch_size", "strength", "order_add", "min_freq_add", "max_freq_add"],
		"universal": ["strength", "order_add", "min_freq_add", "max_freq_add"],
		"physical": ["strength", "order_add", "min_freq_add", "max_freq_add"],
		"semantic": ["strength", "order_add", "min_freq_add", "max_freq_add"],
	}
	
	def update_visible_add_params(noise_type):
		visible = set(visible_params_by_type.get(noise_type, []))

		# Für jeden Parameter die drei Widgets zeigen oder verbergen
		for key, widgets in param_widgets.items():
			lbl, entry, desc = widgets
			if key in visible:
				lbl.grid()   # grid() ohne args nutzt die gespeicherte Position
				entry.grid()
				desc.grid()
			else:
				lbl.grid_remove()
				entry.grid_remove()
				desc.grid_remove()

		# Spatial Spezialfall: eigene Frame für spatial-Parameter
		if noise_type == "spatial":
			frame_spatial_params.grid()
		else:
			frame_spatial_params.grid_remove()

	def on_noise_type_change(event=None):
		selected = combo_noise_type.get()
		update_visible_add_params(selected)
		lbl_noise_desc.config(text=noise_type_descriptions.get(selected, ""))

	combo_noise_type.bind("<<ComboboxSelected>>", on_noise_type_change)


	# Post-Processing Section
	frame_postprocess = ttk.LabelFrame(frame_post_tab, text="Post-Processing", padding=10)
	frame_postprocess.pack(fill="both", expand=True, padx=5, pady=10)

	postprocess_vars['enabled'] = tk.BooleanVar(value=False)
	ttk.Checkbutton(frame_postprocess, text="Enable", 
					variable=postprocess_vars['enabled']).grid(row=0, column=0, columnspan=4, sticky="w", pady=(0,10))

	# row = 1
	
	# Preset Selection
	ttk.Label(frame_postprocess, text="Preset:").grid(row=1, column=0, sticky="w", pady=5)
	combo_postprocess_preset = ttk.Combobox(
		frame_postprocess,
		values=["Custom", "Podcast", "Vocal", "Hiss-Red. post-processing", "Vinyl prepare"],
		state="readonly",
		width=25
	)
	combo_postprocess_preset.set("Custom")
	combo_postprocess_preset.grid(row=1, column=1, columnspan=2, padx=5, pady=5, sticky="w")

	# Storage for entries
	postprocess_entries = {}
	# postprocess_vars = {}  ← GELÖSCHT! Dictionary existiert bereits

	# Parameter rows
	row = 2

	# HIGH-PASS
	postprocess_vars['highpass_enabled'] = tk.BooleanVar(value=False)
	ttk.Checkbutton(frame_postprocess, text="High-Pass Filter", 
					variable=postprocess_vars['highpass_enabled']).grid(row=row, column=0, sticky="w", pady=2)
	ttk.Label(frame_postprocess, text="Cutoff (Hz):").grid(row=row, column=1, sticky="e", padx=5)
	postprocess_entries['highpass_freq'] = ttk.Entry(frame_postprocess, width=8)
	postprocess_entries['highpass_freq'].insert(0, "50")
	postprocess_entries['highpass_freq'].grid(row=row, column=2, sticky="w")
	row += 1

	# LOW-PASS
	postprocess_vars['lowpass_enabled'] = tk.BooleanVar(value=False)
	ttk.Checkbutton(frame_postprocess, text="Low-Pass Filter", 
					variable=postprocess_vars['lowpass_enabled']).grid(row=row, column=0, sticky="w", pady=2)
	ttk.Label(frame_postprocess, text="Cutoff (Hz):").grid(row=row, column=1, sticky="e", padx=5)
	postprocess_entries['lowpass_freq'] = ttk.Entry(frame_postprocess, width=8)
	postprocess_entries['lowpass_freq'].insert(0, "16000")
	postprocess_entries['lowpass_freq'].grid(row=row, column=2, sticky="w")
	row += 1

	# PRE-EMPHASIS
	postprocess_vars['preemphasis_enabled'] = tk.BooleanVar(value=False)
	ttk.Checkbutton(frame_postprocess, text="Pre-Emphasis", 
					variable=postprocess_vars['preemphasis_enabled']).grid(row=row, column=0, sticky="w", pady=2)
	ttk.Label(frame_postprocess, text="Coef (0-1):").grid(row=row, column=1, sticky="e", padx=5)
	postprocess_entries['preemphasis_coef'] = ttk.Entry(frame_postprocess, width=8)
	postprocess_entries['preemphasis_coef'].insert(0, "0.85")
	postprocess_entries['preemphasis_coef'].grid(row=row, column=2, sticky="w")
	row += 1

	# DE-EMPHASIS
	postprocess_vars['deemphasis_enabled'] = tk.BooleanVar(value=False)
	ttk.Checkbutton(frame_postprocess, text="De-Emphasis", 
					variable=postprocess_vars['deemphasis_enabled']).grid(row=row, column=0, sticky="w", pady=2)
	ttk.Label(frame_postprocess, text="Coef (0-1):").grid(row=row, column=1, sticky="e", padx=5)
	postprocess_entries['deemphasis_coef'] = ttk.Entry(frame_postprocess, width=8)
	postprocess_entries['deemphasis_coef'].insert(0, "0.8")
	postprocess_entries['deemphasis_coef'].grid(row=row, column=2, sticky="w")
	row += 1

	# EQ
	postprocess_vars['eq_enabled'] = tk.BooleanVar(value=False)
	ttk.Checkbutton(frame_postprocess, text="EQ (High-Shelf)", 
					variable=postprocess_vars['eq_enabled']).grid(row=row, column=0, sticky="w", pady=2)
	ttk.Label(frame_postprocess, text="Freq (Hz):").grid(row=row, column=1, sticky="e", padx=5)
	postprocess_entries['eq_freq'] = ttk.Entry(frame_postprocess, width=8)
	postprocess_entries['eq_freq'].insert(0, "8000")
	postprocess_entries['eq_freq'].grid(row=row, column=2, sticky="w")
	row += 1
	
	ttk.Label(frame_postprocess, text="Gain (dB):").grid(row=row, column=1, sticky="e", padx=5)
	postprocess_entries['eq_gain'] = ttk.Entry(frame_postprocess, width=8)
	postprocess_entries['eq_gain'].insert(0, "2.0")
	postprocess_entries['eq_gain'].grid(row=row, column=2, sticky="w")
	row += 1

	# COMPRESSION
	postprocess_vars['compression_enabled'] = tk.BooleanVar(value=False)
	ttk.Checkbutton(frame_postprocess, text="Compression", 
					variable=postprocess_vars['compression_enabled']).grid(row=row, column=0, sticky="w", pady=2)
	ttk.Label(frame_postprocess, text="Ratio:").grid(row=row, column=1, sticky="e", padx=5)
	postprocess_entries['compression_ratio'] = ttk.Entry(frame_postprocess, width=8)
	postprocess_entries['compression_ratio'].insert(0, "1.2")
	postprocess_entries['compression_ratio'].grid(row=row, column=2, sticky="w")
	row += 1

	# LIMITER
	postprocess_vars['limiter_enabled'] = tk.BooleanVar(value=False)
	ttk.Checkbutton(frame_postprocess, text="Limiter", 
					variable=postprocess_vars['limiter_enabled']).grid(row=row, column=0, sticky="w", pady=2)
	ttk.Label(frame_postprocess, text="Threshold:").grid(row=row, column=1, sticky="e", padx=5)
	postprocess_entries['limiter_threshold'] = ttk.Entry(frame_postprocess, width=8)
	postprocess_entries['limiter_threshold'].insert(0, "0.95")
	postprocess_entries['limiter_threshold'].grid(row=row, column=2, sticky="w")
	row += 1

	# NORMALIZATION
	postprocess_vars['normalize_enabled'] = tk.BooleanVar(value=True)
	ttk.Checkbutton(frame_postprocess, text="Normalize", 
					variable=postprocess_vars['normalize_enabled']).grid(row=row, column=0, sticky="w", pady=2)
	ttk.Label(frame_postprocess, text="Target:").grid(row=row, column=1, sticky="e", padx=5)
	postprocess_entries['normalize_target'] = ttk.Entry(frame_postprocess, width=8)
	postprocess_entries['normalize_target'].insert(0, "0.989")
	postprocess_entries['normalize_target'].grid(row=row, column=2, sticky="w")
	row += 1

	# BASS MONO
	postprocess_vars['bass_mono_enabled'] = tk.BooleanVar(value=False)
	ttk.Checkbutton(frame_postprocess, text="Bass Mono (Vinyl)", 
					variable=postprocess_vars['bass_mono_enabled']).grid(row=row, column=0, sticky="w", pady=2)
	ttk.Label(frame_postprocess, text="Cutoff (Hz):").grid(row=row, column=1, sticky="e", padx=5)
	postprocess_entries['bass_mono_freq'] = ttk.Entry(frame_postprocess, width=8)
	postprocess_entries['bass_mono_freq'].insert(0, "100")
	postprocess_entries['bass_mono_freq'].grid(row=row, column=2, sticky="w")
	row += 1

	postprocess_settings = {
		'highpass_enabled': postprocess_vars['highpass_enabled'].get(),
		'highpass_freq': float(postprocess_entries['highpass_freq'].get()),
		'lowpass_enabled': postprocess_vars['lowpass_enabled'].get(),
		'lowpass_freq': float(postprocess_entries['lowpass_freq'].get()),
		'preemphasis_enabled': postprocess_vars['preemphasis_enabled'].get(),
		'preemphasis_coef': float(postprocess_entries['preemphasis_coef'].get()),
		'deemphasis_enabled': postprocess_vars['deemphasis_enabled'].get(),
		'deemphasis_coef': float(postprocess_entries['deemphasis_coef'].get()),
		'eq_enabled': postprocess_vars['eq_enabled'].get(),
		'eq_freq': float(postprocess_entries['eq_freq'].get()),
		'eq_gain': float(postprocess_entries['eq_gain'].get()),
		'compression_enabled': postprocess_vars['compression_enabled'].get(),
		'compression_ratio': float(postprocess_entries['compression_ratio'].get()),
		'limiter_enabled': postprocess_vars['limiter_enabled'].get(),
		'limiter_threshold': float(postprocess_entries['limiter_threshold'].get()),
		'normalize_enabled': postprocess_vars['normalize_enabled'].get(),
		'normalize_target': float(postprocess_entries['normalize_target'].get()),
		'bass_mono_enabled': postprocess_vars['bass_mono_enabled'].get(),
		'bass_mono_freq': float(postprocess_entries['bass_mono_freq'].get())
	}

	# === PRESET LOADER ===
	def load_postprocess_preset(event=None):
		preset = combo_postprocess_preset.get()
		
		if preset == "Custom":
			return
		
		# Temporär alle Traces deaktivieren
		for entry in postprocess_entries.values():
			entry.unbind("<KeyRelease>")
		
		trace_ids = {}
		for key, var in postprocess_vars.items():
			if key != 'enabled':
				try:
					# Alle Traces sammeln und entfernen
					traces = var.trace_info()
					for trace in traces:
						var.trace_remove(trace[0], trace[1])
					trace_ids[key] = traces
				except:
					pass
		
		# Reset all (außer 'enabled')
		for key, var in postprocess_vars.items():
			if key != 'enabled':
				var.set(False)
		
		if preset == "Podcast":
			postprocess_vars['preemphasis_enabled'].set(True)
			postprocess_entries['preemphasis_coef'].delete(0, tk.END)
			postprocess_entries['preemphasis_coef'].insert(0, "0.85")
			postprocess_vars['compression_enabled'].set(True)
			postprocess_entries['compression_ratio'].delete(0, tk.END)
			postprocess_entries['compression_ratio'].insert(0, "1.2")
			postprocess_vars['normalize_enabled'].set(True)
			postprocess_vars['bass_mono_enabled'].set(False)
			
		elif preset == "Vocal":
			postprocess_vars['highpass_enabled'].set(True)
			postprocess_entries['highpass_freq'].delete(0, tk.END)
			postprocess_entries['highpass_freq'].insert(0, "80")
			postprocess_vars['preemphasis_enabled'].set(True)
			postprocess_entries['preemphasis_coef'].delete(0, tk.END)
			postprocess_entries['preemphasis_coef'].insert(0, "0.9")
			postprocess_vars['eq_enabled'].set(True)
			postprocess_entries['eq_freq'].delete(0, tk.END)
			postprocess_entries['eq_freq'].insert(0, "5000")
			postprocess_entries['eq_gain'].delete(0, tk.END)
			postprocess_entries['eq_gain'].insert(0, "3.0")
			postprocess_vars['normalize_enabled'].set(True)
			postprocess_vars['bass_mono_enabled'].set(False)
			
		elif preset == "Hiss-Red. post-processing":
			postprocess_vars['lowpass_enabled'].set(True)
			postprocess_entries['lowpass_freq'].delete(0, tk.END)
			postprocess_entries['lowpass_freq'].insert(0, "18000")
			postprocess_vars['eq_enabled'].set(True)
			postprocess_entries['eq_freq'].delete(0, tk.END)
			postprocess_entries['eq_freq'].insert(0, "8000")
			postprocess_entries['eq_gain'].delete(0, tk.END)
			postprocess_entries['eq_gain'].insert(0, "2.0")
			postprocess_vars['limiter_enabled'].set(True)
			postprocess_entries['limiter_threshold'].delete(0, tk.END)
			postprocess_entries['limiter_threshold'].insert(0, "0.95")
			postprocess_vars['normalize_enabled'].set(True)
			postprocess_vars['bass_mono_enabled'].set(False)

			
		elif preset == "Vinyl prepare":
			postprocess_vars['bass_mono_enabled'].set(True) 
			postprocess_entries['bass_mono_freq'].delete(0, tk.END)
			postprocess_entries['bass_mono_freq'].insert(0, "300")
			postprocess_vars['highpass_enabled'].set(True)
			postprocess_entries['highpass_freq'].delete(0, tk.END)
			postprocess_entries['highpass_freq'].insert(0, "20")
			postprocess_vars['lowpass_enabled'].set(True)
			postprocess_entries['lowpass_freq'].delete(0, tk.END)
			postprocess_entries['lowpass_freq'].insert(0, "16000")
			
		# Traces wieder aktivieren
		for entry in postprocess_entries.values():
			entry.bind("<KeyRelease>", on_postprocess_manual_change)
		
		for key, var in postprocess_vars.items():
			if key != 'enabled':
				var.trace_add("write", on_postprocess_manual_change)

	combo_postprocess_preset.bind("<<ComboboxSelected>>", load_postprocess_preset)

	# Manual change -> set to Custom
	def on_postprocess_manual_change(*args):
		if combo_postprocess_preset.get() != "Custom":
			combo_postprocess_preset.set("Custom")

	for entry in postprocess_entries.values():
		entry.bind("<KeyRelease>", on_postprocess_manual_change)

	# Trace nur für Checkboxen (nicht 'enabled')
	for key, var in postprocess_vars.items():
		if key != 'enabled':
			var.trace_add("write", on_postprocess_manual_change)

	# root.mainloop()

	# Plots Tab
	frame_plot = ttk.Frame(notebook_noise)
	notebook_noise.add(frame_plot, text="Analyis")

	notebook_plots = ttk.Notebook(frame_plot)
	notebook_plots.pack(fill="both", expand=True)

	# File Info Tab
	frame_fileinfo = ttk.Frame(notebook_noise)
	notebook_noise.add(frame_fileinfo, text="Info")

	notebook_fileinfo = ttk.Notebook(frame_fileinfo)
	notebook_fileinfo.pack(fill="both", expand=True)
	

	# About Tab
	frame_about = ttk.Frame(notebook_noise)
	notebook_noise.add(frame_about, text="About")

	about_text = (
		"Schnitty's Noise Manager\n\n"
		"This open-source tool allows you to convert audio files into multiple formats "
		"and manage (to remove or add) noise using a variety of modern methods.\n\n"
		"Choose between the traditional filtering method Spectral Gating "
		"or advanced AI-based denoising models such as DeepFilterNet, SpeechBrain, "
		"Demucs, or Asteroid. Each model offers fine-grained control parameters.\n\n"
		"The tool also supports several types of noise manipulation, including additive adversarial, "
		"PGD, spatial, adversarial patch, universal, physical, and semantic noises — "
		"useful for audio-file protection, robustness testing, data augmentation, or creative sound manipulation.\n\n"
		"Key Features:\n"
		" • Multiple noise reduction models\n"
		" • Adjustable filter and denoising parameters\n"
		" • Add synthetic or adversarial noise\n"
		" • Export to common audio formats (.wav, .mp3, .flac, .aiff)\n"
		" • Spectrum analysis and plot export\n\n"
		"Developed by Max Hieke\n\n"
		"If you enjoy this tool or find it useful, you can support future updates and open-source work:\n"
		"Paypal: madguineapig@googlemail.com\n\n"
		"---\n"
		"Sources & References:\n"
		" • DeepFilterNet: https://github.com/facebookresearch/DeepFilterNet (R)\n"
		" • SpeechBrain: https://speechbrain.github.io/ (R)\n"
		" • Demucs: https://github.com/facebookresearch/demucs (R)\n"
		" • Asteroid: https://github.com/asteroid-team/asteroid (R)\n"
		" • Spectral Gating: standard audio denoising approach (C)\n"
		" • Adversarial / Universal / PGD / Spatial / Patch / Physical / Semantic Noise techniques: see published research and implementations (C/R) \n"
		"---\n"
		"\n\nDisclaimer:\n"
		"This software is provided 'as-is' for educational, research, and creative purposes only. "
		"The developer assumes no responsibility for misuse, illegal activities, or damage caused by "
		"the use of this tool.\n\n"
		"By using this software, you agree not to apply it for malicious purposes, including attacks "
		"on third-party systems, bypassing security measures, or unauthorized manipulation of data.\n\n"
		"All third-party models and libraries are used under their respective open-source licenses. "
		"Please respect those licenses when using, distributing, or modifying this software."
	)


	lbl_about = tk.Label(
		frame_about,
		text=about_text,
		justify="left",
		wraplength=800,
		bg="white",      # Hintergrundfarbe
		fg="black"       # Textfarbe
	)
	lbl_about.grid(row=0, column=0, sticky="nw", padx=10, pady=10)

	root.mainloop()


def ladeprogramm():
	"""Simuliert Ladeprozess"""
	global pause_flag
	pause_flag = False
	time.sleep(2)
	splash.quit()


def center_window(window, width, height):
	"""Zentriert Fenster"""
	window.update_idletasks()
	screen_width = window.winfo_screenwidth()
	screen_height = window.winfo_screenheight()
	x = (screen_width // 2) - (width // 2)
	y = (screen_height // 2) - (height // 2)
	window.geometry(f"{width}x{height}+{x}+{y}")


if __name__ == "__main__":
    # Splash Screen
    splash = tk.Tk()
    splash.overrideredirect(True)
    splash.configure(bg="#2e3f4f")
    center_window(splash, 800, 452)
    
    try:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        image_path = os.path.join(script_dir, "Start_Screen.png")
        image = Image.open(image_path)
        image = image.resize((800, 452))
        logo = ImageTk.PhotoImage(image)
        label_logo = tk.Label(splash, image=logo, bg="black")
        label_logo.pack(expand=True)
    except Exception as e:
        print(f"Error loading logo: {e}")
    
    # Flag für Thread-Kommunikation
    loading_done = threading.Event()
    
    def ladeprogramm_wrapper():
        ladeprogramm()
        loading_done.set()
    
    def check_loading():
        if loading_done.is_set():
            splash.destroy()
        else:
            splash.after(100, check_loading)  # Alle 100ms prüfen
    
    # Thread starten und Checking beginnen
    threading.Thread(target=ladeprogramm_wrapper, daemon=True).start()
    splash.after(100, check_loading)
    
    splash.mainloop()
    start_gui()